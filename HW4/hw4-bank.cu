#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>

#define TileSize 32

void input(char *input_filename);
void output(char *output_filename);

__global__ void QKDotAndScalar(float *out, float *q, float *k, int br, int bc, float scalar, int d);
__global__ void RowMax(float *out, float *in, int br, int bc);
__global__ void MinusMaxAndExp(float *out, float *in, float *mx, int br, int bc);
__global__ void RowSum(float *out, float *in, int br, int bc);
__global__ void UpdateMiLiOi(float *mi, float *li, float *oi, float *mij, float *lij, float *pij, float *vj, int br, int bc, int d);
void flash_attention(float *q, float *k, float *v, float *o, float* d_l, float* d_m, int batchIdx);
__global__ void flash_attention_kernel(float* d_kj, float* d_vj, float *q, float *o, float* d_l, float* d_m, int br, int bc, int d, int tr, float scalar);

__device__ float *d_l, *d_m;

double getTimeStamp() {
    struct timeval tv;
    gettimeofday( &tv, NULL );
    return (double) tv.tv_usec/1000000 + tv.tv_sec;
}

int B, N, d;
float *Q, *K, *V, *O;

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <input_filename> <output_filename>\n", argv[0]);
        return 1;
    }
    input(argv[1]);
    int br = TileSize, bc = TileSize;
    int tr = N / br, tc = N / bc;


    // 測量時間
    double start, end;
    start = getTimeStamp();

    float *d_O, *d_Q, *d_K, *d_V;
    cudaMalloc(&d_O, B * N * d * sizeof(float));
    cudaMalloc(&d_Q, B * N * d * sizeof(float));
    cudaMalloc(&d_K, B * N * d * sizeof(float));
    cudaMalloc(&d_V, B * N * d * sizeof(float));

    cudaMemcpy(d_O, O, B * N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Q, Q, B * N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K, B * N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V, B * N * d * sizeof(float), cudaMemcpyHostToDevice);


    // float *h_l = (float *)malloc(N * sizeof(float));
    // float *h_m = (float *)malloc(N * sizeof(float));
    
    cudaMalloc(&d_l, N * sizeof(float));
    cudaMalloc(&d_m, N * sizeof(float));

    for (int batchIdx = 0; batchIdx < B; batchIdx++) {
        flash_attention(
            d_Q + (batchIdx * N * d),
            d_K + (batchIdx * N * d),
            d_V + (batchIdx * N * d),
            d_O + (batchIdx * N * d), d_l, d_m, batchIdx
        );
    }
    cudaMemcpy(O, d_O, B * N * d * sizeof(float), cudaMemcpyDeviceToHost);

    end = getTimeStamp();

    printf("(B, N, d): (%d, %d, %d)\n", B, N, d);
    printf("Time: %.3f seconds\n", end - start);

    output(argv[2]);

    return 0;
}

void flash_attention(float *q, float *k, float *v, float *o, float* d_l, float* d_m, int batchIdx){
    int br = TileSize, bc = TileSize;
    int tr = N / br, tc = N / bc;

    cudaMemset(d_l, 0x00, N * sizeof(float));
    cudaMemset(d_m, FLT_MIN, N * sizeof(float));

    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid(1, tr);
    // Outer Loop
    for (int outerIdx = 0; outerIdx < tc; outerIdx++) {
        flash_attention_kernel<<<blocksPerGrid, threadsPerBlock>>>(k + outerIdx * bc * d, v + outerIdx * bc * d, q, o, d_l, d_m, br, bc, d, tr, 1.0 / sqrt(d));
    }
    // cudaMemcpy(O+batchIdx*N*d, o, N * d * sizeof(float), cudaMemcpyDeviceToHost);
}

__global__ void flash_attention_kernel(float* d_kj, float* d_vj, float *q, float *o, float* d_l, float* d_m, int br, int bc, int d, int tr, float scalar){
    int bcIdx = threadIdx.x;
    int brIdx = threadIdx.y;
    int globalBrIdx = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ float shared_sij[TileSize * (TileSize+1)];
    __shared__ float shared_pij[TileSize * (TileSize+1)];
    __shared__ float shared_mij[TileSize];
    __shared__ float shared_lij[TileSize];
    __shared__ float qi[TileSize * 65];
    __shared__ float kj[TileSize * 65];
    __shared__ float vj[TileSize * 65];
    __shared__ float mi[TileSize];
    __shared__ float li[TileSize];
    __shared__ float oi[TileSize * 65];

    if(bcIdx == 0){
        for(int t = 0;t < d;++t){
            qi[brIdx * (d+1) + t] = q[globalBrIdx * d + t];
        }
    }
    if(brIdx == 0){
        for(int t = 0;t < d;++t){
            kj[bcIdx * (d+1) + t] = d_kj[bcIdx * d + t];
        }
    }
    if(bcIdx == 0){
        mi[brIdx] = d_m[globalBrIdx];
        li[brIdx] = d_l[globalBrIdx];
        for(int t = 0;t < d;++t){
            oi[brIdx * (d+1) + t] = o[globalBrIdx * d + t];                
        }
    }
    if(brIdx == 0){
        for(int t = 0;t < d;++t){
            vj[bcIdx * (d+1) + t] = d_vj[bcIdx * d + t];
        }
    }
    __syncthreads();

    // Query Loop
        // QKDotAndScalar
        if(brIdx < br && bcIdx < bc){
            shared_sij[brIdx * (bc+1) + bcIdx] = 0.0F;
            for (int t = 0; t < d; t++) {
                shared_sij[brIdx * (bc+1) + bcIdx] += qi[brIdx * (d+1) + t] * kj[bcIdx * (d+1) + t];
            }
            shared_sij[brIdx * (bc+1) + bcIdx] *= scalar;
        }
        __syncthreads();
        // RowMax
        if(bcIdx == 0 && brIdx < br){
            shared_mij[brIdx] = shared_sij[brIdx * (bc+1)];
            for (int j = 0; j < bc; j++) {
                shared_mij[brIdx] = fmax(shared_mij[brIdx], shared_sij[brIdx * (bc+1) + j]);
            }
        }
        __syncthreads();
        // MinusMaxAndExp
        if(bcIdx < bc && brIdx < br){
            shared_pij[brIdx * (bc+1) + bcIdx] = expf(shared_sij[brIdx * (bc+1) + bcIdx] - shared_mij[brIdx]);
        }
        __syncthreads();
        // RowSum
        if(bcIdx == 0 && brIdx < br){
            shared_lij[brIdx] = 0.0F;
            for (int j = 0; j < bc; j++) {
                shared_lij[brIdx] += shared_pij[brIdx * (bc+1) + j];
            }
        }
        __syncthreads();        

        // UpdateMiLiOi
        if(bcIdx == 0 && brIdx < br){
            float mi_new = fmax(mi[brIdx], shared_mij[brIdx]);
            float li_new = expf(mi[brIdx] - mi_new) * li[brIdx] + expf(shared_mij[brIdx] - mi_new) * shared_lij[brIdx];
            for (int j = 0; j < d; j++) {
                float pv = 0.0F;
                for (int t = 0; t < bc; t++) {
                    pv += shared_pij[brIdx * (bc+1) + t] * vj[t * (d+1) + j];
                }
                o[globalBrIdx * d + j] = (li[brIdx] * expf(mi[brIdx] - mi_new) * oi[brIdx * (d+1) + j] + expf(shared_mij[brIdx] - mi_new) * pv) / li_new;
            }
            d_m[globalBrIdx] = mi_new;
            d_l[globalBrIdx] = li_new;
        }
        __syncthreads();
}

void input(char *input_filename) {
    FILE *file = fopen(input_filename, "rb");

    fread(&B, sizeof(int), 1, file);
    fread(&N, sizeof(int), 1, file);
    fread(&d, sizeof(int), 1, file);

    Q = (float *)malloc(B * N * d * sizeof(float));
    K = (float *)malloc(B * N * d * sizeof(float));
    V = (float *)malloc(B * N * d * sizeof(float));
    O = (float *)malloc(B * N * d * sizeof(float));

    for (int i = 0; i < B; i++) {
        fread(Q + (i * N * d), sizeof(float), N * d, file);
        fread(K + (i * N * d), sizeof(float), N * d, file);
        fread(V + (i * N * d), sizeof(float), N * d, file);
    }
    memset(O, 0x00, B * N * d * sizeof(float));

    fclose(file);
}

void output(char *output_filename) {
    FILE *file = fopen(output_filename, "wb");

    fwrite(O, sizeof(float), B * N * d, file);

    free(Q);
    free(K);
    free(V);
    free(O);

    fclose(file);
}