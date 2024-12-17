#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>

#define TileSize 32

void input(char *input_filename);
void output(char *output_filename);

void flash_attention(float *q, float *k, float *v, float *o, float* d_l, float* d_m);
__global__ void flash_attention_kernel(float* d_kj, float* d_vj, float *q, float *o, float* d_l, float* d_m, int d, float scalar);
__device__ float _max(float a, float b) { return a > b ? a : b; }
__device__ float _min(float a, float b) { return a < b ? a : b; }

__device__ float *d_l, *d_m;

int br = TileSize, bc = TileSize;
int tr, tc;

double getTimeStamp() {
    struct timeval tv;
    gettimeofday( &tv, NULL );
    return (double) tv.tv_usec/1000000 + tv.tv_sec;
}

int B, N, d;
float *Q, *K, *V, *O;
float *d_O, *d_Q, *d_K, *d_V;
size_t pitchO, pitchQ, pitchK, pitchV, pitchL, pitchM;
cudaStream_t stream;

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <input_filename> <output_filename>\n", argv[0]);
        return 1;
    }
    input(argv[1]);
    cudaStreamCreate(&stream);
    tr = N / br, tc = N / bc;

    // 測量時間
    double start, end;
    start = getTimeStamp();
    
    cudaMallocPitch(&d_O, &pitchO, N * d * sizeof(float), B);
    cudaMallocPitch(&d_Q, &pitchQ, N * d * sizeof(float), B);
    cudaMallocPitch(&d_K, &pitchK, N * d * sizeof(float), B);
    cudaMallocPitch(&d_V, &pitchV, N * d * sizeof(float), B);
    cudaMallocPitch(&d_l, &pitchL, sizeof(float), N);
    cudaMallocPitch(&d_m, &pitchM, sizeof(float), N);

    cudaMemcpy2DAsync(d_O, pitchO, O, N * d * sizeof(float), N * d * sizeof(float), B, cudaMemcpyHostToDevice, stream);
    cudaMemcpy2DAsync(d_Q, pitchQ, Q, N * d * sizeof(float), N * d * sizeof(float), B, cudaMemcpyHostToDevice, stream);
    cudaMemcpy2DAsync(d_K, pitchK, K, N * d * sizeof(float), N * d * sizeof(float), B, cudaMemcpyHostToDevice, stream);
    cudaMemcpy2DAsync(d_V, pitchV, V, N * d * sizeof(float), N * d * sizeof(float), B, cudaMemcpyHostToDevice, stream);
    

    #pragma unroll 320
    for (int batchIdx = 0; batchIdx < B; batchIdx++) {
        flash_attention(
            d_Q + (batchIdx * N * d),
            d_K + (batchIdx * N * d),
            d_V + (batchIdx * N * d),
            d_O + (batchIdx * N * d), d_l, d_m
        );
    }
    cudaMemcpy2DAsync(O, N * d * sizeof(float), d_O, pitchO, N * d * sizeof(float), B, cudaMemcpyDeviceToHost, stream);

    end = getTimeStamp();

    printf("(B, N, d): (%d, %d, %d)\n", B, N, d);
    printf("Time: %.3f seconds\n", end - start);

    output(argv[2]);

    return 0;
}

void flash_attention(float *q, float *k, float *v, float *o, float* d_l, float* d_m){
    cudaMemset(d_l, 0x00, N * sizeof(float));
    cudaMemset(d_m, FLT_MIN, N * sizeof(float));

    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid(tr, 1);
    // Outer Loop
    for (int outerIdx = 0; outerIdx < tc; outerIdx++) {
        flash_attention_kernel<<<blocksPerGrid, threadsPerBlock>>>(k + outerIdx * bc * d, v + outerIdx * bc * d, q, o, d_l, d_m, d, 1.0 / sqrt(d));
    }
}

__global__ void flash_attention_kernel(float* d_kj, float* d_vj, float *q, float *o, float* d_l, float* d_m, int d, float scalar){
    int bcIdx = threadIdx.y;
    int brIdx = threadIdx.x;
    int globalBrIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int bc = blockDim.y;

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
        float sij = 0.0F;
        for (int t = 0; t < d; t++) {
            sij += qi[brIdx * (d+1) + t] * kj[bcIdx * (d+1) + t];
        }
        shared_sij[brIdx * (bc+1) + bcIdx] = sij * scalar;
        __syncthreads();
        // RowMax
        if(bcIdx == 0){
            float mij = shared_sij[brIdx * (bc+1)];
            for (int j = 0; j < bc; j++) {
                mij = _max(mij, shared_sij[brIdx * (bc+1) + j]);
            }
            shared_mij[brIdx] = mij;
        }
        __syncthreads();
        // MinusMaxAndExp
        shared_pij[brIdx * (bc+1) + bcIdx] = __expf(shared_sij[brIdx * (bc+1) + bcIdx] - shared_mij[brIdx]);
        __syncthreads();
        // RowSum
        if(bcIdx == 0){
            float lij = 0.0F;
            for (int j = 0; j < bc; j++) {
                lij += shared_pij[brIdx * (bc+1) + j];
            }
            shared_lij[brIdx] = lij;
        }
        __syncthreads();        

        // UpdateMiLiOi
        if(bcIdx == 0){
            float mi_new = _max(mi[brIdx], shared_mij[brIdx]);
            float li_new = __expf(mi[brIdx] - mi_new) * li[brIdx] + __expf(shared_mij[brIdx] - mi_new) * shared_lij[brIdx];
            for (int j = 0; j < d; j++) {
                float pv = 0.0F;
                for (int t = 0; t < bc; t++) {
                    pv += shared_pij[brIdx * (bc+1) + t] * vj[t * (d+1) + j];
                }
                o[globalBrIdx * d + j] = (li[brIdx] * __expf(mi[brIdx] - mi_new) * oi[brIdx * (d+1) + j] + __expf(shared_mij[brIdx] - mi_new) * pv) / li_new;
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
    // for(int i=0;i<B;++i){
    //     for(int j=0;j<N;++j){
    //         for(int k=0;k<d;++k){
    //             printf("%f ", O[i*N*d + j*d + k]);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }

    fwrite(O, sizeof(float), B * N * d, file);

    free(Q);
    free(K);
    free(V);
    free(O);

    fclose(file);
}