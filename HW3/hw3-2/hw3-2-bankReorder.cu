#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define DEV_NO 0
#define B 64
#define halfB 32
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

    cudaMallocHost((void**)&hostDist, pad_n * pad_n * sizeof(int), cudaHostAllocDefault);
    // hostDist = (int*)malloc(pad_n * pad_n * sizeof(int));

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
    sharedDist[(ty+halfB) * B + tx] = deviceDist[(i+halfB) * n + j];
    sharedDist[ty * B + (tx+halfB)] = deviceDist[i * n + (j+halfB)];
    sharedDist[(ty+halfB) * B + (tx+halfB)] = deviceDist[(i+halfB) * n + (j+halfB)];
    __syncthreads();

    for (int k = 0; k < B; ++k) {
        // sharedDist[ty * B + tx] = min(sharedDist[ty * B + tx], sharedDist[ty * B + k] + sharedDist[k * B + tx]);
        if (sharedDist[ty * B + k] + sharedDist[k * B + ((tx+ty)%halfB)] < sharedDist[ty * B + ((tx+ty)%halfB)]) {
            sharedDist[ty * B + ((tx+ty)%halfB)] = sharedDist[ty * B + k] + sharedDist[k * B + ((tx+ty)%halfB)];
        }
        if (sharedDist[(ty+halfB) * B + k] + sharedDist[k * B + ((tx+ty)%halfB)] < sharedDist[(ty+halfB) * B + ((tx+ty)%halfB)]) {
            sharedDist[(ty+halfB) * B + ((tx+ty)%halfB)] = sharedDist[(ty+halfB) * B + k] + sharedDist[k * B + ((tx+ty)%halfB)];
        }
        if (sharedDist[ty * B + k] + sharedDist[k * B + ((tx+ty)%halfB+halfB)] < sharedDist[ty * B + ((tx+ty)%halfB+halfB)]) {
            sharedDist[ty * B + ((tx+ty)%halfB+halfB)] = sharedDist[ty * B + k] + sharedDist[k * B + ((tx+ty)%halfB+halfB)];
        }
        if (sharedDist[(ty+halfB) * B + k] + sharedDist[k * B + ((tx+ty)%halfB+halfB)] < sharedDist[(ty+halfB) * B + ((tx+ty)%halfB+halfB)]) {
            sharedDist[(ty+halfB) * B + ((tx+ty)%halfB+halfB)] = sharedDist[(ty+halfB) * B + k] + sharedDist[k * B + ((tx+ty)%halfB+halfB)];
        }
        __syncthreads();
    }
    deviceDist[i * n + j] = sharedDist[ty * B + tx];
    deviceDist[(i+halfB) * n + j] = sharedDist[(ty+halfB) * B + tx];
    deviceDist[i * n + (j+halfB)] = sharedDist[ty * B + (tx+halfB)];
    deviceDist[(i+halfB) * n + (j+halfB)] = sharedDist[(ty+halfB) * B + (tx+halfB)];
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
    sharedCenter[(ty+halfB) * B + tx] = deviceDist[(centerY+halfB) * n + centerX];
    sharedCenter[ty * B + (tx+halfB)] = deviceDist[centerY * n + (centerX+halfB)];
    sharedCenter[(ty+halfB) * B + (tx+halfB)] = deviceDist[(centerY+halfB) * n + (centerX+halfB)];
    sharedCol[ty * B + tx] = deviceDist[colY * n + colX];
    sharedCol[(ty+halfB) * B + tx] = deviceDist[(colY+halfB) * n + colX];
    sharedCol[ty * B + (tx+halfB)] = deviceDist[colY * n + (colX+halfB)];
    sharedCol[(ty+halfB) * B + (tx+halfB)] = deviceDist[(colY+halfB) * n + (colX+halfB)];
    sharedRow[ty * B + tx] = deviceDist[rowY * n + rowX];
    sharedRow[(ty+halfB) * B + tx] = deviceDist[(rowY+halfB) * n + rowX];
    sharedRow[ty * B + (tx+halfB)] = deviceDist[rowY * n + (rowX+halfB)];
    sharedRow[(ty+halfB) * B + (tx+halfB)] = deviceDist[(rowY+halfB) * n + (rowX+halfB)];
    __syncthreads();

    for (int k = 0; k < B; ++k) {
        // sharedCol[ty * B + tx] = min(sharedCol[ty * B + tx], sharedCol[ty * B + k] + sharedCenter[k * B + tx]);
        // sharedRow[ty * B + tx] = min(sharedRow[ty * B + tx], sharedCenter[ty * B + k] + sharedRow[k * B + tx]);
        // Update column
        if (sharedCol[ty * B + k] + sharedCenter[k * B + ((tx+ty)%halfB)] < sharedCol[ty * B + ((tx+ty)%halfB)]) {
            sharedCol[ty * B + ((tx+ty)%halfB)] = sharedCol[ty * B + k] + sharedCenter[k * B + ((tx+ty)%halfB)];
        }
        if (sharedCol[(ty+halfB) * B + k] + sharedCenter[k * B + ((tx+ty)%halfB)] < sharedCol[(ty+halfB) * B + ((tx+ty)%halfB)]) {
            sharedCol[(ty+halfB) * B + ((tx+ty)%halfB)] = sharedCol[(ty+halfB) * B + k] + sharedCenter[k * B + ((tx+ty)%halfB)];
        }
        if (sharedCol[ty * B + k] + sharedCenter[k * B + ((tx+ty)%halfB+halfB)] < sharedCol[ty * B + ((tx+ty)%halfB+halfB)]) {
            sharedCol[ty * B + ((tx+ty)%halfB+halfB)] = sharedCol[ty * B + k] + sharedCenter[k * B + ((tx+ty)%halfB+halfB)];
        }
        if (sharedCol[(ty+halfB) * B + k] + sharedCenter[k * B + ((tx+ty)%halfB+halfB)] < sharedCol[(ty+halfB) * B + ((tx+ty)%halfB+halfB)]) {
            sharedCol[(ty+halfB) * B + ((tx+ty)%halfB+halfB)] = sharedCol[(ty+halfB) * B + k] + sharedCenter[k * B + ((tx+ty)%halfB+halfB)];
        }
        // Update row
        if (sharedCenter[ty * B + k] + sharedRow[k * B + ((tx+ty)%halfB)] < sharedRow[ty * B + ((tx+ty)%halfB)]) {
            sharedRow[ty * B + ((tx+ty)%halfB)] = sharedCenter[ty * B + k] + sharedRow[k * B + ((tx+ty)%halfB)];
        }
        if (sharedCenter[(ty+halfB) * B + k] + sharedRow[k * B + ((tx+ty)%halfB)] < sharedRow[(ty+halfB) * B + ((tx+ty)%halfB)]) {
            sharedRow[(ty+halfB) * B + ((tx+ty)%halfB)] = sharedCenter[(ty+halfB) * B + k] + sharedRow[k * B + ((tx+ty)%halfB)];
        }
        if (sharedCenter[ty * B + k] + sharedRow[k * B + ((tx+ty)%halfB+halfB)] < sharedRow[ty * B + ((tx+ty)%halfB+halfB)]) {
            sharedRow[ty * B + ((tx+ty)%halfB+halfB)] = sharedCenter[ty * B + k] + sharedRow[k * B + ((tx+ty)%halfB+halfB)];
        }
        if (sharedCenter[(ty+halfB) * B + k] + sharedRow[k * B + ((tx+ty)%halfB+halfB)] < sharedRow[(ty+halfB) * B + ((tx+ty)%halfB+halfB)]) {
            sharedRow[(ty+halfB) * B + ((tx+ty)%halfB+halfB)] = sharedCenter[(ty+halfB) * B + k] + sharedRow[k * B + ((tx+ty)%halfB+halfB)];
        }
    }
    deviceDist[colY * n + colX] = sharedCol[ty * B + tx];
    deviceDist[(colY+halfB) * n + colX] = sharedCol[(ty+halfB) * B + tx];
    deviceDist[colY * n + (colX+halfB)] = sharedCol[ty * B + (tx+halfB)];
    deviceDist[(colY+halfB) * n + (colX+halfB)] = sharedCol[(ty+halfB) * B + (tx+halfB)];
    deviceDist[rowY * n + rowX] = sharedRow[ty * B + tx];
    deviceDist[(rowY+halfB) * n + rowX] = sharedRow[(ty+halfB) * B + tx];
    deviceDist[rowY * n + (rowX+halfB)] = sharedRow[ty * B + (tx+halfB)];
    deviceDist[(rowY+halfB) * n + (rowX+halfB)] = sharedRow[(ty+halfB) * B + (tx+halfB)];
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
    sharedCol[(ty+halfB) * B + tx] = deviceDist[(colY+halfB) * n + colX];
    sharedCol[ty * B + (tx+halfB)] = deviceDist[colY * n + (colX+halfB)];
    sharedCol[(ty+halfB) * B + (tx+halfB)] = deviceDist[(colY+halfB) * n + (colX+halfB)];
    sharedRow[ty * B + tx] = deviceDist[rowY * n + rowX];
    sharedRow[(ty+halfB) * B + tx] = deviceDist[(rowY+halfB) * n + rowX];
    sharedRow[ty * B + (tx+halfB)] = deviceDist[rowY * n + (rowX+halfB)];
    sharedRow[(ty+halfB) * B + (tx+halfB)] = deviceDist[(rowY+halfB) * n + (rowX+halfB)];
    __syncthreads();
    // 每個cell的dependency都是row和column
    int tmpBlock1 = deviceDist[centerY * n + centerX];
    int tmpBlock2 = deviceDist[(centerY+halfB) * n + centerX];
    int tmpBlock3 = deviceDist[centerY * n + (centerX+halfB)];
    int tmpBlock4 = deviceDist[(centerY+halfB) * n + (centerX+halfB)];

    for (int k = 0; k < B; ++k) {
        // tmpBlock = min(tmpBlock, sharedCol[ty * B + k] + sharedRow[k * B + tx]);
        // Update block
        if (sharedCol[ty * B + k] + sharedRow[k * B + tx] < tmpBlock1) {
            tmpBlock1 = sharedCol[ty * B + k] + sharedRow[k * B + tx];
        }
        if (sharedCol[(ty+halfB) * B + k] + sharedRow[k * B + tx] < tmpBlock2) {
            tmpBlock2 = sharedCol[(ty+halfB) * B + k] + sharedRow[k * B + tx];
        }
        if (sharedCol[ty * B + k] + sharedRow[k * B + (tx+halfB)] < tmpBlock3) {
            tmpBlock3 = sharedCol[ty * B + k] + sharedRow[k * B + (tx+halfB)];
        }
        if (sharedCol[(ty+halfB) * B + k] + sharedRow[k * B + (tx+halfB)] < tmpBlock4) {
            tmpBlock4 = sharedCol[(ty+halfB) * B + k] + sharedRow[k * B + (tx+halfB)];
        }
    }
    deviceDist[centerY * n + centerX] = tmpBlock1;
    deviceDist[(centerY+halfB) * n + centerX] = tmpBlock2;
    deviceDist[centerY * n + (centerX+halfB)] = tmpBlock3;
    deviceDist[(centerY+halfB) * n + (centerX+halfB)] = tmpBlock4;
}


int main(int argc, char* argv[]) {
    input(argv[1]);
    cudaMalloc(&deviceDist, pad_n * pad_n * sizeof(int));
    cudaMemcpy(deviceDist, hostDist, pad_n * pad_n * sizeof(int), cudaMemcpyHostToDevice);


    // pad_n是B的倍數
    int round = pad_n / B;
    dim3 threadsPerBlock(32, 32);
    dim3 gridsPerBlock1(1, 1);
    dim3 gridsPerBlock2(1, round);
    dim3 gridsPerBlock3(round, round);
    for (int r = 0; r < round; ++r) {
        cal1<<<gridsPerBlock1, threadsPerBlock>>>(r, deviceDist, pad_n);
        cal2<<<gridsPerBlock2, threadsPerBlock>>>(r, deviceDist, pad_n);
        cal3<<<gridsPerBlock3, threadsPerBlock>>>(r, deviceDist, pad_n);
    }

    cudaMemcpy(hostDist, deviceDist, pad_n * pad_n * sizeof(int), cudaMemcpyDeviceToHost);
    output(argv[2]);
    return 0;
}
