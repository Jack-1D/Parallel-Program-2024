#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>

void input(char *input_filename);
void output(char *output_filename);

__global__ void QKDotAndScalar(float *out, float *q, float *k, int br, int bc, float scalar, int d);
__global__ void RowMax(float *out, float *in, int br, int bc);
__global__ void MinusMaxAndExp(float *out, float *in, float *mx, int br, int bc);
__global__ void RowSum(float *out, float *in, int br, int bc);
__global__ void UpdateMiLiOi(float *mi, float *li, float *oi, float *mij, float *lij, float *pij, float *vj, int br, int bc, int d);


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
    int br = 32, bc = 32;
    int tr = N / br, tc = N / bc;


    // 測量時間
    double start, end;
    start = getTimeStamp();

    float *h_l = (float *)malloc(N * sizeof(float));
    float *h_m = (float *)malloc(N * sizeof(float));
    float *d_l, *d_m;
    cudaMalloc(&d_l, N * sizeof(float));
    cudaMalloc(&d_m, N * sizeof(float));

    float *h_kj = (float *)malloc(bc * d * sizeof(float));
    float *h_vj = (float *)malloc(bc * d * sizeof(float));
    float *h_qi = (float *)malloc(br * d * sizeof(float));
    float *h_oi = (float *)malloc(br * d * sizeof(float));
    float *h_li = (float *)malloc(br * sizeof(float));
    float *h_mi = (float *)malloc(br * sizeof(float));

    float *h_sij = (float *)malloc(br * bc * sizeof(float));
    float *h_pij = (float *)malloc(br * bc * sizeof(float));
    float *h_mij = (float *)malloc(br * sizeof(float));
    float *h_lij = (float *)malloc(br * sizeof(float));

    float *d_kj, *d_vj, *d_qi, *d_oi, *d_li, *d_mi, *d_sij, *d_pij, *d_mij, *d_lij;
    cudaMalloc(&d_kj, bc * d * sizeof(float));
    cudaMalloc(&d_vj, bc * d * sizeof(float));
    cudaMalloc(&d_qi, br * d * sizeof(float));
    cudaMalloc(&d_oi, br * d * sizeof(float));
    cudaMalloc(&d_li, br * sizeof(float));
    cudaMalloc(&d_mi, br * sizeof(float));

    cudaMalloc(&d_sij, br * bc * sizeof(float));
    cudaMalloc(&d_pij, br * bc * sizeof(float));
    cudaMalloc(&d_mij, br * sizeof(float));
    cudaMalloc(&d_lij, br * sizeof(float));

    for (int batchIdx = 0; batchIdx < B; batchIdx++) {
        memset(h_l, 0x00, N * sizeof(float));
        for (int idx = 0; idx < N; idx++) {
            h_m[idx] = FLT_MIN;
        }
        cudaMemcpy(d_l, h_l, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_m, h_m, N * sizeof(float), cudaMemcpyHostToDevice);

        dim3 threadsPerBlock(32, 32);
        for (int j = 0; j < tc; j++) {
            cudaMemcpy(d_kj, (K + (batchIdx * N * d)) + j * bc * d, bc * d * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_vj, (V + (batchIdx * N * d)) + j * bc * d, bc * d * sizeof(float), cudaMemcpyHostToDevice);
            // Query Loop
            for (int i = 0; i < tr; i++) {
                cudaMemcpy(d_qi, (Q + (batchIdx * N * d)) + i * br * d, br * d * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(d_oi, (O + (batchIdx * N * d)) + i * br * d, br * d * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(d_li, d_l + i * br, br * sizeof(float), cudaMemcpyDeviceToDevice);
                cudaMemcpy(d_mi, d_m + i * br, br * sizeof(float), cudaMemcpyDeviceToDevice);

                QKDotAndScalar<<<((bc + threadsPerBlock.x - 1) / threadsPerBlock.x, (br + threadsPerBlock.y - 1) / threadsPerBlock.y), threadsPerBlock>>>(d_sij, d_qi, d_kj, br, bc, 1.0 / sqrt(d), d);
                RowMax<<<(br + threadsPerBlock.x - 1) / threadsPerBlock.x, threadsPerBlock.x>>>(d_mij, d_sij, br, bc);
                MinusMaxAndExp<<<((bc + threadsPerBlock.x - 1) / threadsPerBlock.x, (br + threadsPerBlock.y - 1) / threadsPerBlock.y), threadsPerBlock>>>(d_pij, d_sij, d_mij, br, bc);
                RowSum<<<(br + threadsPerBlock.x - 1) / threadsPerBlock.x, threadsPerBlock.x>>>(d_lij, d_pij, br, bc);
                UpdateMiLiOi<<<(br + threadsPerBlock.x - 1) / threadsPerBlock.x, threadsPerBlock.x>>>(d_mi, d_li, d_oi, d_mij, d_lij, d_pij, d_vj, br, bc, d);
                cudaMemcpy((O + (batchIdx * N * d)) + i * br * d, d_oi, br * d * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(d_l + i * br, d_li, br * sizeof(float), cudaMemcpyDeviceToDevice);
                cudaMemcpy(d_m + i * br, d_mi, br * sizeof(float), cudaMemcpyDeviceToDevice);
            }
        }
    }

    end = getTimeStamp();

    printf("(B, N, d): (%d, %d, %d)\n", B, N, d);
    printf("Time: %.3f seconds\n", end - start);

    output(argv[2]);

    free(h_sij);
    free(h_pij);
    free(h_mij);
    free(h_lij);

    free(h_kj);
    free(h_vj);
    free(h_qi);
    free(h_oi);
    free(h_li);
    free(h_mi);

    free(h_l);
    free(h_m);

    cudaFree(d_sij);
    cudaFree(d_pij);
    cudaFree(d_mij);
    cudaFree(d_lij);

    cudaFree(d_kj);
    cudaFree(d_vj);
    cudaFree(d_qi);
    cudaFree(d_oi);
    cudaFree(d_li);
    cudaFree(d_mi);

    cudaFree(d_l);
    cudaFree(d_m);

    return 0;
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

__global__ void QKDotAndScalar(float *out, float *q, float *k, int br, int bc, float scalar, int d) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < br && col < bc) {
        float sum = 0.0F;
        for (int t = 0; t < d; t++) {
            sum += q[row * d + t] * k[col * d + t];
        }
        out[row * bc + col] = sum * scalar;
    }
}

__global__ void RowMax(float *out, float *in, int br, int bc) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < br) {
        float max_val = in[row * bc];
        for (int j = 1; j < bc; j++) {
            max_val = fmaxf(max_val, in[row * bc + j]);
        }
        out[row] = max_val;
    }
}

__global__ void MinusMaxAndExp(float *out, float *in, float *mx, int br, int bc) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < br && col < bc) {
        out[row * bc + col] = expf(in[row * bc + col] - mx[row]);
    }
}

__global__ void RowSum(float *out, float *in, int br, int bc) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < br) {
        float sum = 0.0F;
        for (int j = 0; j < bc; j++) {
            sum += in[row * bc + j];
        }
        out[row] = sum;
    }
}

__global__ void UpdateMiLiOi(
    float *mi, float *li, float *oi,
    float *mij, float *lij, float *pij,
    float *vj, int br, int bc, int d
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    // int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < br) {
        float mi_new = fmaxf(mi[row], mij[row]);
        float li_new = expf(mi[row] - mi_new) * li[row] + expf(mij[row] - mi_new) * lij[row];

        // if (col < d) {
        //     float pv = 0.0F;
        //     for (int t = 0; t < bc; t++) {
        //         pv += pij[row * bc + t] * vj[t * d + col];
        //     }
        //     oi[row * d + col] = (li[row] * expf(mi[row] - mi_new) * oi[row * d + col] +
        //                          expf(mij[row] - mi_new) * pv) / li_new;
        // }
        for (int j = 0; j < d; j++) {
            float pv = 0.0F;
            for (int t = 0; t < bc; t++) {
                pv += pij[row * bc + t] * vj[t * d + j];
            }
            oi[row * d + j] = (li[row] * expf(mi[row] - mi_new) * oi[row * d + j] + expf(mij[row] - mi_new) * pv) / li_new;
        }

        // if (col == 0) {
            mi[row] = mi_new;
            li[row] = li_new;
        // }
    }
}
