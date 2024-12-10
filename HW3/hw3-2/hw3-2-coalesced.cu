#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define DEV_NO 0
cudaDeviceProp prop;

const int INF = ((1 << 30) - 1);
const int B = 32;

int n, m;
int *hostDist;
int *deviceDist;

void block_FW(int B, int *deviceDist);

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    hostDist = (int*)malloc(n * n * sizeof(int));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                hostDist[i*n+j] = 0;
            } else {
                hostDist[i*n+j] = INF;
            }
        }
    }

    int pair[3];
    for (int i = 0; i < m; ++i) {
        fread(pair, sizeof(int), 3, file);
        hostDist[pair[0]*n+pair[1]] = pair[2];
    }
    fclose(file);
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (hostDist[i*n+j] >= INF) hostDist[i*n+j] = INF;
        }
        fwrite(hostDist + i * n, sizeof(int), n, outfile);
    }
    fclose(outfile);
}

int ceil(int a, int b) { return (a + b - 1) / b; }


__global__ void cal(
    int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height, int *deviceDist, int n) {

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    // GPU block idx平移到要計算的block上(單位r)
    int block_x = blockIdx.x + block_start_x;
    int block_y = blockIdx.y + block_start_y;

    // GPU block idx平移到要計算的block上(單位cell)
    int block_internal_start_x = block_x * B;
    int block_internal_start_y = block_y * B;

    int i = block_internal_start_x + ty;
    int j = block_internal_start_y + tx;

    if (i < n && j < n) {
        for (int k = Round * B; k < (Round + 1) * B && k < n; ++k) {
            // deviceDist[i * n + j] = min(deviceDist[i * n + j], deviceDist[i * n + k] + deviceDist[k * n + j]);
            if (deviceDist[i * n + k] + deviceDist[k * n + j] < deviceDist[i * n + j]) {
                deviceDist[i * n + j] = deviceDist[i * n + k] + deviceDist[k * n + j];
            }
        }
    }
}

int main(int argc, char* argv[]) {
    input(argv[1]);
    cudaMalloc(&deviceDist, n * n * sizeof(int));
    cudaMemcpy(deviceDist, hostDist, n * n * sizeof(int), cudaMemcpyHostToDevice);

    cudaGetDeviceProperties(&prop, DEV_NO);
    printf("maxThreadsPerBlock = %d, sharedMemPerBlock = %d\n", prop.maxThreadsPerBlock, prop.sharedMemPerBlock);

    block_FW(B, deviceDist);
    cudaMemcpy(hostDist, deviceDist, n * n * sizeof(int), cudaMemcpyDeviceToHost);
    output(argv[2]);
    return 0;
}

void block_FW(int B, int *deviceDist) {
    int round = ceil(n, B);
    dim3 threadsPerBlock(32, 32);
    dim3 gridsPerBlock1(1, 1);
    dim3 gridsPerBlock2(1, ceil(n, B));
    dim3 gridsPerBlock3(ceil(n, B), ceil(n, B));
    for (int r = 0; r < round; ++r) {
        printf("%d %d\n", r, round);
        fflush(stdout);
        /* Phase 1*/
        cal<<<gridsPerBlock1, threadsPerBlock>>>(B, r, r, r, 1, 1, deviceDist, n);

        /* Phase 2*/
        cal<<<gridsPerBlock2, threadsPerBlock>>>(B, r, r, 0, r, 1, deviceDist, n);
        cal<<<gridsPerBlock2, threadsPerBlock>>>(B, r, r, r + 1, round - r - 1, 1, deviceDist, n);
        cal<<<gridsPerBlock2, threadsPerBlock>>>(B, r, 0, r, 1, r, deviceDist, n);
        cal<<<gridsPerBlock2, threadsPerBlock>>>(B, r, r + 1, r, 1, round - r - 1, deviceDist, n);

        /* Phase 3*/
        cal<<<gridsPerBlock3, threadsPerBlock>>>(B, r, 0, 0, r, r, deviceDist, n);
        cal<<<gridsPerBlock3, threadsPerBlock>>>(B, r, 0, r + 1, round - r - 1, r, deviceDist, n);
        cal<<<gridsPerBlock3, threadsPerBlock>>>(B, r, r + 1, 0, r, round - r - 1, deviceDist, n);
        cal<<<gridsPerBlock3, threadsPerBlock>>>(B, r, r + 1, r + 1, round - r - 1, round - r - 1, deviceDist, n);
    }
}
