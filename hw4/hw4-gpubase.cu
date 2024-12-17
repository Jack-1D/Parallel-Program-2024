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
void flash_attention(float *q, float *k, float *v, float *o, float* d_l, float* d_m);
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

    // float *d_kj, *d_vj, *d_qi, *d_oi, *d_li, *d_mi, *d_sij, *d_pij, *d_mij, *d_lij;
    // cudaMalloc(&d_kj, bc * d * sizeof(float));
    // cudaMalloc(&d_vj, bc * d * sizeof(float));
    // cudaMalloc(&d_qi, br * d * sizeof(float));
    // cudaMalloc(&d_oi, br * d * sizeof(float));
    // cudaMalloc(&d_li, br * sizeof(float));
    // cudaMalloc(&d_mi, br * sizeof(float));

    // cudaMalloc(&d_sij, br * bc * sizeof(float));
    // cudaMalloc(&d_pij, br * bc * sizeof(float));
    // cudaMalloc(&d_mij, br * sizeof(float));
    // cudaMalloc(&d_lij, br * sizeof(float));

    for (int batchIdx = 0; batchIdx < B; batchIdx++) {
        flash_attention(
            d_Q + (batchIdx * N * d),
            d_K + (batchIdx * N * d),
            d_V + (batchIdx * N * d),
            d_O + (batchIdx * N * d), d_l, d_m
        );
    }
    cudaMemcpy(O, d_O, B * N * d * sizeof(float), cudaMemcpyDeviceToHost);

    end = getTimeStamp();

    printf("(B, N, d): (%d, %d, %d)\n", B, N, d);
    printf("Time: %.3f seconds\n", end - start);

    output(argv[2]);

    // cudaFree(d_sij);
    // cudaFree(d_pij);
    // cudaFree(d_mij);
    // cudaFree(d_lij);

    // cudaFree(d_kj);
    // cudaFree(d_vj);
    // cudaFree(d_qi);
    // cudaFree(d_oi);
    // cudaFree(d_li);
    // cudaFree(d_mi);

    cudaFree(d_l);
    cudaFree(d_m);

    return 0;
}

void flash_attention(float *q, float *k, float *v, float *o, float* d_l, float* d_m){
    int br = TileSize, bc = TileSize;
    int tr = N / br, tc = N / bc;

    cudaMemset(d_l, 0x00, N * sizeof(float));
    cudaMemset(d_m, FLT_MIN, N * sizeof(float));
    // for (int idx = 0; idx < N; idx++) {
    //     d_m[idx] = FLT_MIN;
    // }
    // cudaMemcpy(d_l, h_l, N * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_m, h_m, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid(1, tr);
    // Outer Loop
    for (int outerIdx = 0; outerIdx < tc; outerIdx++) {
        // cudaMemcpy(d_kj, k + j * bc * d, bc * d * sizeof(float), cudaMemcpyDeviceToDevice);
        // cudaMemcpy(d_vj, v + j * bc * d, bc * d * sizeof(float), cudaMemcpyDeviceToDevice);
        // flash_attention_kernel<<<blocksPerGrid, threadsPerBlock>>>(k + j * bc * d, v + j * bc * d, q, o, d_l, d_m, br, bc, d, d_qi, d_oi, d_li, d_mi);
        flash_attention_kernel<<<1, 1>>>(k + outerIdx * bc * d, v + outerIdx * bc * d, q, o, d_l, d_m, br, bc, d, tr, 1.0 / sqrt(d));
        cudaDeviceSynchronize();
    }
}

__global__ void flash_attention_kernel(float* d_kj, float* d_vj, float *q, float *o, float* d_l, float* d_m, int br, int bc, int d, int tr, float scalar){
    // int tid = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float shared_sij[TileSize * TileSize];
    __shared__ float shared_pij[TileSize * TileSize];
    __shared__ float shared_mij[TileSize];
    __shared__ float shared_lij[TileSize];

    // Query Loop
    for (int innerIdx = 0; innerIdx < tr; innerIdx++) {
        // int i = tid;
        // memcpy(d_qi, q + trIdx * br * d, br * d * sizeof(float));
        // memcpy(d_oi, o + trIdx * br * d, br * d * sizeof(float));
        // memcpy(d_li, d_l + trIdx * br, br * sizeof(float));
        // memcpy(d_mi, d_m + trIdx * br, br * sizeof(float));

        // QKDotAndScalar<<<1, 1>>>(shared_sij, d_qi, d_kj, br, bc, 1.0 / sqrt(d), d);
        // RowMax<<<1, 1>>>(shared_mij, shared_sij, br, bc);
        // MinusMaxAndExp<<<1, 1>>>(shared_pij, shared_sij, shared_mij, br, bc);
        // RowSum<<<1, 1>>>(shared_lij, shared_pij, br, bc);
        // UpdateMiLiOi<<<1, 1>>>(d_mi, d_li, d_oi, shared_mij, shared_lij, shared_pij, d_vj, br, bc, d);
        
        // QKDotAndScalar
        for (int i = 0; i < br; i++) {
            for (int j = 0; j < bc; j++) {
                shared_sij[i * bc + j] = 0.0F;
                for (int t = 0; t < d; t++) {
                    shared_sij[i * bc + j] += *(q + innerIdx * br * d + i * d + t) * d_kj[j * d + t];
                }
                shared_sij[i * bc + j] *= scalar;
            }
        }
        // RowMax
        for (int i = 0; i < br; i++) {
            shared_mij[i] = shared_sij[i * bc];
            for (int j = 0; j < bc; j++) {
                shared_mij[i] = fmax(shared_mij[i], shared_sij[i * bc + j]);
            }
        }
        // MinusMaxAndExp
        for (int i = 0; i < br; i++) {
            for (int j = 0; j < bc; j++) {
                shared_pij[i * bc + j] = expf(shared_sij[i * bc + j] - shared_mij[i]);
            }
        }
        // RowSum
        for (int i = 0; i < br; i++) {
            shared_lij[i] = 0.0F;
            for (int j = 0; j < bc; j++) {
                shared_lij[i] += shared_pij[i * bc + j];
            }
        }
        // UpdateMiLiOi
        float *mi_new = (float *)malloc(br * sizeof(float));
        float *li_new = (float *)malloc(br * sizeof(float));

        for (int i = 0; i < br; i++) {
            mi_new[i] = fmax(*(d_m + innerIdx * br + i), shared_mij[i]);
            li_new[i] = expf(*(d_m + innerIdx * br + i) - mi_new[i]) * *(d_l + innerIdx * br + i) + exp(shared_mij[i] - mi_new[i]) * shared_lij[i];
        }

        for (int i = 0; i < br; i++) {
            for (int j = 0; j < d; j++) {
                float pv = 0.0F;
                for (int t = 0; t < bc; t++) {
                    pv += shared_pij[i * bc + t] * d_vj[t * d + j];
                }
                *(o + innerIdx * br * d + i * d + j) = (*(d_l + innerIdx * br + i) * expf(*(d_m + innerIdx * br + i) - mi_new[i]) * *(o + innerIdx * br * d + i * d + j) + expf(shared_mij[i] - mi_new[i]) * pv) / li_new[i];
            }
        }
        // for(int i=0;i<br;++i){
        //     for(int j=0;j<d;++j){
        //         printf("%f ", *(o + innerIdx * br * d + i * d + j));
        //     }
        //     printf("\n");
        // }
        memcpy(d_m + innerIdx * br, mi_new, br * sizeof(float));
        memcpy(d_l + innerIdx * br, li_new, br * sizeof(float));
        free(mi_new);
        free(li_new);

        // memcpy(o + trIdx * br * d, d_oi, br * d * sizeof(float));
        // memcpy(d_l + trIdx * br, d_li, br * sizeof(float));
        // memcpy(d_m + trIdx * br, d_mi, br * sizeof(float));
    }
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
    for(int i=0;i<B;++i){
        for(int j=0;j<N;++j){
            for(int k=0;k<d;++k){
                printf("%f ", O[i*N*d + j*d + k]);
            }
            printf("\n");
        }
        printf("\n");
    }

    fwrite(O, sizeof(float), B * N * d, file);

    free(Q);
    free(K);
    free(V);
    free(O);

    fclose(file);
}



__global__ void QKDotAndScalar(float *out, float *q, float *k, int br, int bc, float scalar, int d) {
    for (int i = 0; i < br; i++) {
        for (int j = 0; j < bc; j++) {
            out[i * bc + j] = 0.0F;
            for (int t = 0; t < d; t++) {
                out[i * bc + j] += q[i * d + t] * k[j * d + t];
            }
            out[i * bc + j] *= scalar;
        }
    }
}

__global__ void RowMax(float *out, float *in, int br, int bc) {
    for (int i = 0; i < br; i++) {
        out[i] = in[i * bc];
        for (int j = 0; j < bc; j++) {
            out[i] = fmax(out[i], in[i * bc + j]);
        }
    }
}

__global__ void MinusMaxAndExp(float *out, float *in, float *mx, int br, int bc) {
    for (int i = 0; i < br; i++) {
        for (int j = 0; j < bc; j++) {
            out[i * bc + j] = expf(in[i * bc + j] - mx[i]);
        }
    }
}

__global__ void RowSum(float *out, float *in, int br, int bc) {
    for (int i = 0; i < br; i++) {
        out[i] = 0.0F;
        for (int j = 0; j < bc; j++) {
            out[i] += in[i * bc + j];
        }
    }
}

__global__ void UpdateMiLiOi(float *mi, float *li, float *oi, float *mij, float *lij, float *pij, float *vj, int br, int bc, int d) {
    float *mi_new = (float *)malloc(br * sizeof(float));
    float *li_new = (float *)malloc(br * sizeof(float));

    for (int i = 0; i < br; i++) {
        mi_new[i] = fmax(mi[i], mij[i]);
        li_new[i] = expf(mi[i] - mi_new[i]) * li[i] + exp(mij[i] - mi_new[i]) * lij[i];
    }

    for (int i = 0; i < br; i++) {
        for (int j = 0; j < d; j++) {
            float pv = 0.0F;
            for (int t = 0; t < bc; t++) {
                pv += pij[i * bc + t] * vj[t * d + j];
            }
            oi[i * d + j] = (li[i] * expf(mi[i] - mi_new[i]) * oi[i * d + j] + expf(mij[i] - mi_new[i]) * pv) / li_new[i];
        }
    }

    memcpy(mi, mi_new, br * sizeof(float));
    memcpy(li, li_new, br * sizeof(float));
    
    free(mi_new);
    free(li_new);
}

// __global__ void QKDotAndScalar(float *out, float *q, float *k, int br, int bc, float scalar, int d) {
//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     int col = blockIdx.x * blockDim.x + threadIdx.x;

//     if (row < br && col < bc) {
//         float sum = 0.0F;
//         for (int t = 0; t < d; t++) {
//             sum += q[row * d + t] * k[col * d + t];
//         }
//         out[row * bc + col] = sum * scalar;
//     }
// }

// __global__ void RowMax(float *out, float *in, int br, int bc) {
//     int row = blockIdx.x * blockDim.x + threadIdx.x;

//     if (row < br) {
//         float max_val = in[row * bc];
//         for (int j = 1; j < bc; j++) {
//             max_val = fmaxf(max_val, in[row * bc + j]);
//         }
//         out[row] = max_val;
//     }
// }

// __global__ void MinusMaxAndExp(float *out, float *in, float *mx, int br, int bc) {
//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     int col = blockIdx.x * blockDim.x + threadIdx.x;

//     if (row < br && col < bc) {
//         out[row * bc + col] = expf(in[row * bc + col] - mx[row]);
//     }
// }

// __global__ void RowSum(float *out, float *in, int br, int bc) {
//     int row = blockIdx.x * blockDim.x + threadIdx.x;

//     if (row < br) {
//         float sum = 0.0F;
//         for (int j = 0; j < bc; j++) {
//             sum += in[row * bc + j];
//         }
//         out[row] = sum;
//     }
// }

// __global__ void UpdateMiLiOi(
//     float *mi, float *li, float *oi,
//     float *mij, float *lij, float *pij,
//     float *vj, int br, int bc, int d
// ) {
//     int row = blockIdx.x * blockDim.x + threadIdx.x;
//     // int col = blockIdx.y * blockDim.y + threadIdx.y;

//     if (row < br) {
//         float mi_new = fmaxf(mi[row], mij[row]);
//         float li_new = expf(mi[row] - mi_new) * li[row] + expf(mij[row] - mi_new) * lij[row];

//         // if (col < d) {
//         //     float pv = 0.0F;
//         //     for (int t = 0; t < bc; t++) {
//         //         pv += pij[row * bc + t] * vj[t * d + col];
//         //     }
//         //     oi[row * d + col] = (li[row] * expf(mi[row] - mi_new) * oi[row * d + col] +
//         //                          expf(mij[row] - mi_new) * pv) / li_new;
//         // }
//         for (int j = 0; j < d; j++) {
//             float pv = 0.0F;
//             for (int t = 0; t < bc; t++) {
//                 pv += pij[row * bc + t] * vj[t * d + j];
//             }
//             oi[row * d + j] = (li[row] * expf(mi[row] - mi_new) * oi[row * d + j] + expf(mij[row] - mi_new) * pv) / li_new;
//         }

//         // if (col == 0) {
//             mi[row] = mi_new;
//             li[row] = li_new;
//         // }
//     }
// }
