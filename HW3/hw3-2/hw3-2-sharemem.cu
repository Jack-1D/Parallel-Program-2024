#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define DEV_NO 0
#define B 32
cudaDeviceProp prop;

const int INF = ((1 << 30) - 1);

int n, m, pad_n;
int *hostDist;
int *deviceDist;

void block_FW(int *deviceDist);

int ceil(int a, int b) { return (a + b - 1) / b; }

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    pad_n = ceil(n, B) * B;

    hostDist = (int*)malloc(pad_n * pad_n * sizeof(int));

    for (int i = 0; i < pad_n; ++i) {
        for (int j = 0; j < pad_n; ++j) {
            if (i == j) {
                hostDist[i*pad_n+j] = 0;
            } else {
                hostDist[i*pad_n+j] = INF;
            }
        }
    }

    int pair[3];
    for (int i = 0; i < m; ++i) {
        fread(pair, sizeof(int), 3, file);
        hostDist[pair[0]*pad_n+pair[1]] = pair[2];
    }
    fclose(file);
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    for (int i = 0; i < pad_n; ++i) {
        for (int j = 0; j < pad_n; ++j) {
            if (hostDist[i*pad_n+j] >= INF) hostDist[i*pad_n+j] = INF;
        }
        // fwrite(hostDist + i * n, sizeof(int), n, outfile);
    }
    for(int i=0; i < n; ++i){
        fwrite(hostDist + i * pad_n, sizeof(int), n, outfile);
    }
    fclose(outfile);
}


__global__ void cal1(
    int Round, int *deviceDist, int n) {

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // GPU block idx平移到要計算的block上(單位cell)
    int block_internal_start_x = Round * B;
    int block_internal_start_y = Round * B;

    int j = block_internal_start_x + tx;
    int i = block_internal_start_y + ty;

    __shared__ int sharedDist[B * B];
    sharedDist[ty * B + tx] = deviceDist[i * n + j];
    __syncthreads();

    for (int k = 0; k < B; ++k) {
        // sharedDist[ty * B + tx] = min(sharedDist[ty * B + tx], sharedDist[ty * B + k] + sharedDist[k * B + tx]);
        if (sharedDist[ty * B + k] + sharedDist[k * B + tx] < sharedDist[ty * B + tx]) {
            sharedDist[ty * B + tx] = sharedDist[ty * B + k] + sharedDist[k * B + tx];
        }
        __syncthreads();
    }
    deviceDist[i * n + j] = sharedDist[ty * B + tx];
}

__global__ void cal2(
    int Round, int *deviceDist, int n) {
    // Center不需更新
    if(blockIdx.y == Round)
        return;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // GPU block idx平移到要計算的block上(單位cell)
    int block_internal_start_x = Round * B;
    int block_internal_start_y = Round * B;

    int centerX = block_internal_start_x + tx;
    int centerY = block_internal_start_y + ty;
    int colX = centerX;
    int colY = blockIdx.y * B + ty;
    int rowX = blockIdx.y * B + tx;
    int rowY = centerY;

    __shared__ int sharedCenter[B * B];
    __shared__ int sharedCol[B * B];
    __shared__ int sharedRow[B * B];

    sharedCenter[ty * B + tx] = deviceDist[centerY * n + centerX];
    sharedCol[ty * B + tx] = deviceDist[colY * n + colX];
    sharedRow[ty * B + tx] = deviceDist[rowY * n + rowX];
    __syncthreads();

    for (int k = 0; k < B; ++k) {
        // sharedCol[ty * B + tx] = min(sharedCol[ty * B + tx], sharedCol[ty * B + k] + sharedCenter[k * B + tx]);
        // sharedRow[ty * B + tx] = min(sharedRow[ty * B + tx], sharedCenter[ty * B + k] + sharedRow[k * B + tx]);
        // Update column
        if (sharedCol[ty * B + k] + sharedCenter[k * B + tx] < sharedCol[ty * B + tx]) {
            sharedCol[ty * B + tx] = sharedCol[ty * B + k] + sharedCenter[k * B + tx];
        }
        // Update row
        if (sharedCenter[ty * B + k] + sharedRow[k * B + tx] < sharedRow[ty * B + tx]) {
            sharedRow[ty * B + tx] = sharedCenter[ty * B + k] + sharedRow[k * B + tx];
        }
        __syncthreads();
    }
    deviceDist[colY * n + colX] = sharedCol[ty * B + tx];
    deviceDist[rowY * n + rowX] = sharedRow[ty * B + tx];
}

__global__ void cal3(
    int Round, int *deviceDist, int n) {
    // 撇除phase1和phase2
    if(blockIdx.x == Round || blockIdx.y == Round)
        return;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // GPU block idx平移到要計算的block上(單位cell)
    int block_internal_start_x = blockIdx.x * B;
    int block_internal_start_y = blockIdx.y * B;

    int centerX = block_internal_start_x + tx;
    int centerY = block_internal_start_y + ty;
    // 只算自己那欄
    int colX = Round * B + tx;
    int colY = centerY;
    int rowX = centerX;
    // 只算自己那列
    int rowY = Round * B + ty;

    __shared__ int sharedCol[B * B];
    __shared__ int sharedRow[B * B];

    sharedCol[ty * B + tx] = deviceDist[colY * n + colX];
    sharedRow[ty * B + tx] = deviceDist[rowY * n + rowX];
    __syncthreads();
    // 每個cell的dependency都是row和column
    int tmpBlock = deviceDist[centerY * n + centerX];

    for (int k = 0; k < B; ++k) {
        // tmpBlock = min(tmpBlock, sharedCol[ty * B + k] + sharedRow[k * B + tx]);
        // Update block
        if (sharedCol[ty * B + k] + sharedRow[k * B + tx] < tmpBlock) {
            tmpBlock = sharedCol[ty * B + k] + sharedRow[k * B + tx];
        }
    }
    deviceDist[centerY * n + centerX] = tmpBlock;
}


int main(int argc, char* argv[]) {
    input(argv[1]);
    cudaMalloc(&deviceDist, pad_n * pad_n * sizeof(int));
    cudaMemcpy(deviceDist, hostDist, pad_n * pad_n * sizeof(int), cudaMemcpyHostToDevice);

    cudaGetDeviceProperties(&prop, DEV_NO);
    printf("maxThreadsPerBlock = %d, sharedMemPerBlock = %d\n", prop.maxThreadsPerBlock, prop.sharedMemPerBlock);

    block_FW(deviceDist);
    cudaMemcpy(hostDist, deviceDist, pad_n * pad_n * sizeof(int), cudaMemcpyDeviceToHost);
    output(argv[2]);
    return 0;
}

void block_FW(int *deviceDist) {
    int round = ceil(pad_n, B);
    dim3 threadsPerBlock(32, 32);
    dim3 gridsPerBlock1(1, 1);
    dim3 gridsPerBlock2(1, ceil(pad_n, B));
    dim3 gridsPerBlock3(ceil(pad_n, B), ceil(pad_n, B));
    for (int r = 0; r < round; ++r) {
        printf("%d %d\n", r, round);
        fflush(stdout);
        /* Phase 1*/
        cal1<<<gridsPerBlock1, threadsPerBlock>>>(r, deviceDist, pad_n);

        /* Phase 2*/
        cal2<<<gridsPerBlock2, threadsPerBlock>>>(r, deviceDist, pad_n);

        /* Phase 3*/
        cal3<<<gridsPerBlock3, threadsPerBlock>>>(r, deviceDist, pad_n);
    }
}
