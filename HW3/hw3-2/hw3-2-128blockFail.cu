#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define DEV_NO 0
#define B 128
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
    sharedDist[(ty+2*halfB) * B + tx] = deviceDist[(i+2*halfB) * n + j];
    sharedDist[(ty+3*halfB) * B + tx] = deviceDist[(i+3*halfB) * n + j];
    sharedDist[ty * B + (tx+halfB)] = deviceDist[i * n + (j+halfB)];
    sharedDist[(ty+halfB) * B + (tx+halfB)] = deviceDist[(i+halfB) * n + (j+halfB)];
    sharedDist[(ty+2*halfB) * B + (tx+halfB)] = deviceDist[(i+2*halfB) * n + (j+halfB)];
    sharedDist[(ty+3*halfB) * B + (tx+halfB)] = deviceDist[(i+3*halfB) * n + (j+halfB)];
    sharedDist[ty * B + (tx+2*halfB)] = deviceDist[i * n + (j+2*halfB)];
    sharedDist[(ty+halfB) * B + (tx+2*halfB)] = deviceDist[(i+halfB) * n + (j+2*halfB)];
    sharedDist[(ty+2*halfB) * B + (tx+2*halfB)] = deviceDist[(i+2*halfB) * n + (j+2*halfB)];
    sharedDist[(ty+3*halfB) * B + (tx+2*halfB)] = deviceDist[(i+3*halfB) * n + (j+2*halfB)];
    sharedDist[ty * B + (tx+3*halfB)] = deviceDist[i * n + (j+3*halfB)];
    sharedDist[(ty+halfB) * B + (tx+3*halfB)] = deviceDist[(i+halfB) * n + (j+3*halfB)];
    sharedDist[(ty+2*halfB) * B + (tx+3*halfB)] = deviceDist[(i+2*halfB) * n + (j+3*halfB)];
    sharedDist[(ty+3*halfB) * B + (tx+3*halfB)] = deviceDist[(i+3*halfB) * n + (j+3*halfB)];
    __syncthreads();

    for (int k = 0; k < B; ++k) {
        sharedDist[ty * B + tx] = min(sharedDist[ty * B + tx], sharedDist[ty * B + k] + sharedDist[k * B + tx]);
        sharedDist[(ty+halfB) * B + tx] = min(sharedDist[(ty+halfB) * B + tx], sharedDist[(ty+halfB) * B + k] + sharedDist[k * B + tx]);
        sharedDist[(ty+2*halfB) * B + tx] = min(sharedDist[(ty+2*halfB) * B + tx], sharedDist[(ty+2*halfB) * B + k] + sharedDist[k * B + tx]);
        sharedDist[(ty+3*halfB) * B + tx] = min(sharedDist[(ty+3*halfB) * B + tx], sharedDist[(ty+3*halfB) * B + k] + sharedDist[k * B + tx]);
        sharedDist[ty * B + (tx+halfB)] = min(sharedDist[ty * B + (tx+halfB)], sharedDist[ty * B + k] + sharedDist[k * B + (tx+halfB)]);
        sharedDist[(ty+halfB) * B + (tx+halfB)] = min(sharedDist[(ty+halfB) * B + (tx+halfB)], sharedDist[(ty+halfB) * B + k] + sharedDist[k * B + (tx+halfB)]);
        sharedDist[(ty+2*halfB) * B + (tx+halfB)] = min(sharedDist[(ty+2*halfB) * B + (tx+halfB)], sharedDist[(ty+2*halfB) * B + k] + sharedDist[k * B + (tx+halfB)]);
        sharedDist[(ty+3*halfB) * B + (tx+halfB)] = min(sharedDist[(ty+3*halfB) * B + (tx+halfB)], sharedDist[(ty+3*halfB) * B + k] + sharedDist[k * B + (tx+halfB)]);
        sharedDist[ty * B + (tx+2*halfB)] = min(sharedDist[ty * B + (tx+2*halfB)], sharedDist[ty * B + k] + sharedDist[k * B + (tx+2*halfB)]);
        sharedDist[(ty+halfB) * B + (tx+2*halfB)] = min(sharedDist[(ty+halfB) * B + (tx+2*halfB)], sharedDist[(ty+halfB) * B + k] + sharedDist[k * B + (tx+2*halfB)]);
        sharedDist[(ty+2*halfB) * B + (tx+2*halfB)] = min(sharedDist[(ty+2*halfB) * B + (tx+2*halfB)], sharedDist[(ty+2*halfB) * B + k] + sharedDist[k * B + (tx+2*halfB)]);
        sharedDist[(ty+3*halfB) * B + (tx+2*halfB)] = min(sharedDist[(ty+3*halfB) * B + (tx+2*halfB)], sharedDist[(ty+3*halfB) * B + k] + sharedDist[k * B + (tx+2*halfB)]);
        sharedDist[ty * B + (tx+3*halfB)] = min(sharedDist[ty * B + (tx+3*halfB)], sharedDist[ty * B + k] + sharedDist[k * B + (tx+3*halfB)]);
        sharedDist[(ty+halfB) * B + (tx+3*halfB)] = min(sharedDist[(ty+halfB) * B + (tx+3*halfB)], sharedDist[(ty+halfB) * B + k] + sharedDist[k * B + (tx+3*halfB)]);
        sharedDist[(ty+2*halfB) * B + (tx+3*halfB)] = min(sharedDist[(ty+2*halfB) * B + (tx+3*halfB)], sharedDist[(ty+2*halfB) * B + k] + sharedDist[k * B + (tx+3*halfB)]);
        sharedDist[(ty+3*halfB) * B + (tx+3*halfB)] = min(sharedDist[(ty+3*halfB) * B + (tx+3*halfB)], sharedDist[(ty+3*halfB) * B + k] + sharedDist[k * B + (tx+3*halfB)]);
        
        __syncthreads();
    }
    deviceDist[i * n + j] = sharedDist[ty * B + tx];
    deviceDist[(i+halfB) * n + j] = sharedDist[(ty+halfB) * B + tx];
    deviceDist[(i+2*halfB) * n + j] = sharedDist[(ty+2*halfB) * B + tx];
    deviceDist[(i+3*halfB) * n + j] = sharedDist[(ty+3*halfB) * B + tx];
    deviceDist[i * n + (j+halfB)] = sharedDist[ty * B + (tx+halfB)];
    deviceDist[(i+halfB) * n + (j+halfB)] = sharedDist[(ty+halfB) * B + (tx+halfB)];
    deviceDist[(i+2*halfB) * n + (j+halfB)] = sharedDist[(ty+2*halfB) * B + (tx+halfB)];
    deviceDist[(i+3*halfB) * n + (j+halfB)] = sharedDist[(ty+3*halfB) * B + (tx+halfB)];
    deviceDist[i * n + (j+2*halfB)] = sharedDist[ty * B + (tx+2*halfB)];
    deviceDist[(i+halfB) * n + (j+2*halfB)] = sharedDist[(ty+halfB) * B + (tx+2*halfB)];
    deviceDist[(i+2*halfB) * n + (j+2*halfB)] = sharedDist[(ty+2*halfB) * B + (tx+2*halfB)];
    deviceDist[(i+3*halfB) * n + (j+2*halfB)] = sharedDist[(ty+3*halfB) * B + (tx+2*halfB)];
    deviceDist[i * n + (j+3*halfB)] = sharedDist[ty * B + (tx+3*halfB)];
    deviceDist[(i+halfB) * n + (j+3*halfB)] = sharedDist[(ty+halfB) * B + (tx+3*halfB)];
    deviceDist[(i+2*halfB) * n + (j+3*halfB)] = sharedDist[(ty+2*halfB) * B + (tx+3*halfB)];
    deviceDist[(i+3*halfB) * n + (j+3*halfB)] = sharedDist[(ty+3*halfB) * B + (tx+3*halfB)];
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
    sharedCenter[(ty+2*halfB) * B + tx] = deviceDist[(centerY+2*halfB) * n + centerX];
    sharedCenter[(ty+3*halfB) * B + tx] = deviceDist[(centerY+3*halfB) * n + centerX];
    sharedCenter[ty * B + (tx+halfB)] = deviceDist[centerY * n + (centerX+halfB)];
    sharedCenter[(ty+halfB) * B + (tx+halfB)] = deviceDist[(centerY+halfB) * n + (centerX+halfB)];
    sharedCenter[(ty+2*halfB) * B + (tx+halfB)] = deviceDist[(centerY+2*halfB) * n + (centerX+halfB)];
    sharedCenter[(ty+3*halfB) * B + (tx+halfB)] = deviceDist[(centerY+3*halfB) * n + (centerX+halfB)];
    sharedCenter[ty * B + (tx+2*halfB)] = deviceDist[centerY * n + (centerX+2*halfB)];
    sharedCenter[(ty+halfB) * B + (tx+2*halfB)] = deviceDist[(centerY+halfB) * n + (centerX+2*halfB)];
    sharedCenter[(ty+2*halfB) * B + (tx+2*halfB)] = deviceDist[(centerY+2*halfB) * n + (centerX+2*halfB)];
    sharedCenter[(ty+3*halfB) * B + (tx+2*halfB)] = deviceDist[(centerY+3*halfB) * n + (centerX+2*halfB)];
    sharedCenter[ty * B + (tx+3*halfB)] = deviceDist[centerY * n + (centerX+3*halfB)];
    sharedCenter[(ty+halfB) * B + (tx+3*halfB)] = deviceDist[(centerY+halfB) * n + (centerX+3*halfB)];
    sharedCenter[(ty+2*halfB) * B + (tx+3*halfB)] = deviceDist[(centerY+2*halfB) * n + (centerX+3*halfB)];
    sharedCenter[(ty+3*halfB) * B + (tx+3*halfB)] = deviceDist[(centerY+3*halfB) * n + (centerX+3*halfB)];

    sharedCol[ty * B + tx] = deviceDist[colY * n + colX];
    sharedCol[(ty+halfB) * B + tx] = deviceDist[(colY+halfB) * n + colX];
    sharedCol[(ty+2*halfB) * B + tx] = deviceDist[(colY+2*halfB) * n + colX];
    sharedCol[(ty+3*halfB) * B + tx] = deviceDist[(colY+3*halfB) * n + colX];
    sharedCol[ty * B + (tx+halfB)] = deviceDist[colY * n + (colX+halfB)];
    sharedCol[(ty+halfB) * B + (tx+halfB)] = deviceDist[(colY+halfB) * n + (colX+halfB)];
    sharedCol[(ty+2*halfB) * B + (tx+halfB)] = deviceDist[(colY+2*halfB) * n + (colX+halfB)];
    sharedCol[(ty+3*halfB) * B + (tx+halfB)] = deviceDist[(colY+3*halfB) * n + (colX+halfB)];
    sharedCol[ty * B + (tx+2*halfB)] = deviceDist[colY * n + (colX+2*halfB)];
    sharedCol[(ty+halfB) * B + (tx+2*halfB)] = deviceDist[(colY+halfB) * n + (colX+2*halfB)];
    sharedCol[(ty+2*halfB) * B + (tx+2*halfB)] = deviceDist[(colY+2*halfB) * n + (colX+2*halfB)];
    sharedCol[(ty+3*halfB) * B + (tx+2*halfB)] = deviceDist[(colY+3*halfB) * n + (colX+2*halfB)];
    sharedCol[ty * B + (tx+3*halfB)] = deviceDist[colY * n + (colX+3*halfB)];
    sharedCol[(ty+halfB) * B + (tx+3*halfB)] = deviceDist[(colY+halfB) * n + (colX+3*halfB)];
    sharedCol[(ty+2*halfB) * B + (tx+3*halfB)] = deviceDist[(colY+2*halfB) * n + (colX+3*halfB)];
    sharedCol[(ty+3*halfB) * B + (tx+3*halfB)] = deviceDist[(colY+3*halfB) * n + (colX+3*halfB)];

    sharedRow[ty * B + tx] = deviceDist[rowY * n + rowX];
    sharedRow[(ty+halfB) * B + tx] = deviceDist[(rowY+halfB) * n + rowX];
    sharedRow[(ty+2*halfB) * B + tx] = deviceDist[(rowY+2*halfB) * n + rowX];
    sharedRow[(ty+3*halfB) * B + tx] = deviceDist[(rowY+3*halfB) * n + rowX];
    sharedRow[ty * B + (tx+halfB)] = deviceDist[rowY * n + (rowX+halfB)];
    sharedRow[(ty+halfB) * B + (tx+halfB)] = deviceDist[(rowY+halfB) * n + (rowX+halfB)];
    sharedRow[(ty+2*halfB) * B + (tx+halfB)] = deviceDist[(rowY+2*halfB) * n + (rowX+halfB)];
    sharedRow[(ty+3*halfB) * B + (tx+halfB)] = deviceDist[(rowY+3*halfB) * n + (rowX+halfB)];
    sharedRow[ty * B + (tx+2*halfB)] = deviceDist[rowY * n + (rowX+2*halfB)];
    sharedRow[(ty+halfB) * B + (tx+2*halfB)] = deviceDist[(rowY+halfB) * n + (rowX+2*halfB)];
    sharedRow[(ty+2*halfB) * B + (tx+2*halfB)] = deviceDist[(rowY+2*halfB) * n + (rowX+2*halfB)];
    sharedRow[(ty+3*halfB) * B + (tx+2*halfB)] = deviceDist[(rowY+3*halfB) * n + (rowX+2*halfB)];
    sharedRow[ty * B + (tx+3*halfB)] = deviceDist[rowY * n + (rowX+3*halfB)];
    sharedRow[(ty+halfB) * B + (tx+3*halfB)] = deviceDist[(rowY+halfB) * n + (rowX+3*halfB)];
    sharedRow[(ty+2*halfB) * B + (tx+3*halfB)] = deviceDist[(rowY+2*halfB) * n + (rowX+3*halfB)];
    sharedRow[(ty+3*halfB) * B + (tx+3*halfB)] = deviceDist[(rowY+3*halfB) * n + (rowX+3*halfB)];

    __syncthreads();

    for (int k = 0; k < B; ++k) {
        // Update column
        sharedCol[ty * B + tx] = min(sharedCol[ty * B + tx], sharedCol[ty * B + k] + sharedCenter[k * B + tx]);
        sharedCol[(ty+halfB) * B + tx] = min(sharedCol[(ty+halfB) * B + tx], sharedCol[(ty+halfB) * B + k] + sharedCenter[k * B + tx]);
        sharedCol[(ty+2*halfB) * B + tx] = min(sharedCol[(ty+2*halfB) * B + tx], sharedCol[(ty+2*halfB) * B + k] + sharedCenter[k * B + tx]);
        sharedCol[(ty+3*halfB) * B + tx] = min(sharedCol[(ty+3*halfB) * B + tx], sharedCol[(ty+3*halfB) * B + k] + sharedCenter[k * B + tx]);
        sharedCol[ty * B + (tx+halfB)] = min(sharedCol[ty * B + (tx+halfB)], sharedCol[ty * B + k] + sharedCenter[k * B + (tx+halfB)]);
        sharedCol[(ty+halfB) * B + (tx+halfB)] = min(sharedCol[(ty+halfB) * B + (tx+halfB)], sharedCol[(ty+halfB) * B + k] + sharedCenter[k * B + (tx+halfB)]);
        sharedCol[(ty+2*halfB) * B + (tx+halfB)] = min(sharedCol[(ty+2*halfB) * B + (tx+halfB)], sharedCol[(ty+2*halfB) * B + k] + sharedCenter[k * B + (tx+halfB)]);
        sharedCol[(ty+3*halfB) * B + (tx+halfB)] = min(sharedCol[(ty+3*halfB) * B + (tx+halfB)], sharedCol[(ty+3*halfB) * B + k] + sharedCenter[k * B + (tx+halfB)]);
        sharedCol[ty * B + (tx+2*halfB)] = min(sharedCol[ty * B + (tx+2*halfB)], sharedCol[ty * B + k] + sharedCenter[k * B + (tx+2*halfB)]);
        sharedCol[(ty+halfB) * B + (tx+2*halfB)] = min(sharedCol[(ty+halfB) * B + (tx+2*halfB)], sharedCol[(ty+halfB) * B + k] + sharedCenter[k * B + (tx+2*halfB)]);
        sharedCol[(ty+2*halfB) * B + (tx+2*halfB)] = min(sharedCol[(ty+2*halfB) * B + (tx+2*halfB)], sharedCol[(ty+2*halfB) * B + k] + sharedCenter[k * B + (tx+2*halfB)]);
        sharedCol[(ty+3*halfB) * B + (tx+2*halfB)] = min(sharedCol[(ty+3*halfB) * B + (tx+2*halfB)], sharedCol[(ty+3*halfB) * B + k] + sharedCenter[k * B + (tx+2*halfB)]);
        sharedCol[ty * B + (tx+3*halfB)] = min(sharedCol[ty * B + (tx+3*halfB)], sharedCol[ty * B + k] + sharedCenter[k * B + (tx+3*halfB)]);
        sharedCol[(ty+halfB) * B + (tx+3*halfB)] = min(sharedCol[(ty+halfB) * B + (tx+3*halfB)], sharedCol[(ty+halfB) * B + k] + sharedCenter[k * B + (tx+3*halfB)]);
        sharedCol[(ty+2*halfB) * B + (tx+3*halfB)] = min(sharedCol[(ty+2*halfB) * B + (tx+3*halfB)], sharedCol[(ty+2*halfB) * B + k] + sharedCenter[k * B + (tx+3*halfB)]);
        sharedCol[(ty+3*halfB) * B + (tx+3*halfB)] = min(sharedCol[(ty+3*halfB) * B + (tx+3*halfB)], sharedCol[(ty+3*halfB) * B + k] + sharedCenter[k * B + (tx+3*halfB)]);

        // Update row
        sharedRow[ty * B + tx] = min(sharedRow[ty * B + tx], sharedCenter[ty * B + k] + sharedRow[k * B + tx]);
        sharedRow[(ty+halfB) * B + tx] = min(sharedRow[(ty+halfB) * B + tx], sharedCenter[(ty+halfB) * B + k] + sharedRow[k * B + tx]);
        sharedRow[(ty+2*halfB) * B + tx] = min(sharedRow[(ty+2*halfB) * B + tx], sharedCenter[(ty+2*halfB) * B + k] + sharedRow[k * B + tx]);
        sharedRow[(ty+3*halfB) * B + tx] = min(sharedRow[(ty+3*halfB) * B + tx], sharedCenter[(ty+3*halfB) * B + k] + sharedRow[k * B + tx]);
        sharedRow[ty * B + (tx+halfB)] = min(sharedRow[ty * B + (tx+halfB)], sharedCenter[ty * B + k] + sharedRow[k * B + (tx+halfB)]);
        sharedRow[(ty+halfB) * B + (tx+halfB)] = min(sharedRow[(ty+halfB) * B + (tx+halfB)], sharedCenter[(ty+halfB) * B + k] + sharedRow[k * B + (tx+halfB)]);
        sharedRow[(ty+2*halfB) * B + (tx+halfB)] = min(sharedRow[(ty+2*halfB) * B + (tx+halfB)], sharedCenter[(ty+2*halfB) * B + k] + sharedRow[k * B + (tx+halfB)]);
        sharedRow[(ty+3*halfB) * B + (tx+halfB)] = min(sharedRow[(ty+3*halfB) * B + (tx+halfB)], sharedCenter[(ty+3*halfB) * B + k] + sharedRow[k * B + (tx+halfB)]);
        sharedRow[ty * B + (tx+2*halfB)] = min(sharedRow[ty * B + (tx+2*halfB)], sharedCenter[ty * B + k] + sharedRow[k * B + (tx+2*halfB)]);
        sharedRow[(ty+halfB) * B + (tx+2*halfB)] = min(sharedRow[(ty+halfB) * B + (tx+2*halfB)], sharedCenter[(ty+halfB) * B + k] + sharedRow[k * B + (tx+2*halfB)]);
        sharedRow[(ty+2*halfB) * B + (tx+2*halfB)] = min(sharedRow[(ty+2*halfB) * B + (tx+2*halfB)], sharedCenter[(ty+2*halfB) * B + k] + sharedRow[k * B + (tx+2*halfB)]);
        sharedRow[(ty+3*halfB) * B + (tx+2*halfB)] = min(sharedRow[(ty+3*halfB) * B + (tx+2*halfB)], sharedCenter[(ty+3*halfB) * B + k] + sharedRow[k * B + (tx+2*halfB)]);
        sharedRow[ty * B + (tx+3*halfB)] = min(sharedRow[ty * B + (tx+3*halfB)], sharedCenter[ty * B + k] + sharedRow[k * B + (tx+3*halfB)]);
        sharedRow[(ty+halfB) * B + (tx+3*halfB)] = min(sharedRow[(ty+halfB) * B + (tx+3*halfB)], sharedCenter[(ty+halfB) * B + k] + sharedRow[k * B + (tx+3*halfB)]);
        sharedRow[(ty+2*halfB) * B + (tx+3*halfB)] = min(sharedRow[(ty+2*halfB) * B + (tx+3*halfB)], sharedCenter[(ty+2*halfB) * B + k] + sharedRow[k * B + (tx+3*halfB)]);
        sharedRow[(ty+3*halfB) * B + (tx+3*halfB)] = min(sharedRow[(ty+3*halfB) * B + (tx+3*halfB)], sharedCenter[(ty+3*halfB) * B + k] + sharedRow[k * B + (tx+3*halfB)]);

        __syncthreads();
    }
    deviceDist[colY * n + colX] = sharedCol[ty * B + tx];
    deviceDist[(colY+halfB) * n + colX] = sharedCol[(ty+halfB) * B + tx];
    deviceDist[(colY+2*halfB) * n + colX] = sharedCol[(ty+2*halfB) * B + tx];
    deviceDist[(colY+3*halfB) * n + colX] = sharedCol[(ty+3*halfB) * B + tx];
    deviceDist[colY * n + (colX+halfB)] = sharedCol[ty * B + (tx+halfB)];
    deviceDist[(colY+halfB) * n + (colX+halfB)] = sharedCol[(ty+halfB) * B + (tx+halfB)];
    deviceDist[(colY+2*halfB) * n + (colX+halfB)] = sharedCol[(ty+2*halfB) * B + (tx+halfB)];
    deviceDist[(colY+3*halfB) * n + (colX+halfB)] = sharedCol[(ty+3*halfB) * B + (tx+halfB)];
    deviceDist[colY * n + (colX+2*halfB)] = sharedCol[ty * B + (tx+2*halfB)];
    deviceDist[(colY+halfB) * n + (colX+2*halfB)] = sharedCol[(ty+halfB) * B + (tx+2*halfB)];
    deviceDist[(colY+2*halfB) * n + (colX+2*halfB)] = sharedCol[(ty+2*halfB) * B + (tx+2*halfB)];
    deviceDist[(colY+3*halfB) * n + (colX+2*halfB)] = sharedCol[(ty+3*halfB) * B + (tx+2*halfB)];
    deviceDist[colY * n + (colX+3*halfB)] = sharedCol[ty * B + (tx+3*halfB)];
    deviceDist[(colY+halfB) * n + (colX+3*halfB)] = sharedCol[(ty+halfB) * B + (tx+3*halfB)];
    deviceDist[(colY+2*halfB) * n + (colX+3*halfB)] = sharedCol[(ty+2*halfB) * B + (tx+3*halfB)];
    deviceDist[(colY+3*halfB) * n + (colX+3*halfB)] = sharedCol[(ty+3*halfB) * B + (tx+3*halfB)];

    deviceDist[rowY * n + rowX] = sharedRow[ty * B + tx];
    deviceDist[(rowY+halfB) * n + rowX] = sharedRow[(ty+halfB) * B + tx];
    deviceDist[(rowY+2*halfB) * n + rowX] = sharedRow[(ty+2*halfB) * B + tx];
    deviceDist[(rowY+3*halfB) * n + rowX] = sharedRow[(ty+3*halfB) * B + tx];
    deviceDist[rowY * n + (rowX+halfB)] = sharedRow[ty * B + (tx+halfB)];
    deviceDist[(rowY+halfB) * n + (rowX+halfB)] = sharedRow[(ty+halfB) * B + (tx+halfB)];
    deviceDist[(rowY+2*halfB) * n + (rowX+halfB)] = sharedRow[(ty+2*halfB) * B + (tx+halfB)];
    deviceDist[(rowY+3*halfB) * n + (rowX+halfB)] = sharedRow[(ty+3*halfB) * B + (tx+halfB)];
    deviceDist[rowY * n + (rowX+2*halfB)] = sharedRow[ty * B + (tx+2*halfB)];
    deviceDist[(rowY+halfB) * n + (rowX+2*halfB)] = sharedRow[(ty+halfB) * B + (tx+2*halfB)];
    deviceDist[(rowY+2*halfB) * n + (rowX+2*halfB)] = sharedRow[(ty+2*halfB) * B + (tx+2*halfB)];
    deviceDist[(rowY+3*halfB) * n + (rowX+2*halfB)] = sharedRow[(ty+3*halfB) * B + (tx+2*halfB)];
    deviceDist[rowY * n + (rowX+3*halfB)] = sharedRow[ty * B + (tx+3*halfB)];
    deviceDist[(rowY+halfB) * n + (rowX+3*halfB)] = sharedRow[(ty+halfB) * B + (tx+3*halfB)];
    deviceDist[(rowY+2*halfB) * n + (rowX+3*halfB)] = sharedRow[(ty+2*halfB) * B + (tx+3*halfB)];
    deviceDist[(rowY+3*halfB) * n + (rowX+3*halfB)] = sharedRow[(ty+3*halfB) * B + (tx+3*halfB)];
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
    sharedCol[(ty+2*halfB) * B + tx] = deviceDist[(colY+2*halfB) * n + colX];
    sharedCol[(ty+3*halfB) * B + tx] = deviceDist[(colY+3*halfB) * n + colX];
    sharedCol[ty * B + (tx+halfB)] = deviceDist[colY * n + (colX+halfB)];
    sharedCol[(ty+halfB) * B + (tx+halfB)] = deviceDist[(colY+halfB) * n + (colX+halfB)];
    sharedCol[(ty+2*halfB) * B + (tx+halfB)] = deviceDist[(colY+2*halfB) * n + (colX+halfB)];
    sharedCol[(ty+3*halfB) * B + (tx+halfB)] = deviceDist[(colY+3*halfB) * n + (colX+halfB)];
    sharedCol[ty * B + (tx+2*halfB)] = deviceDist[colY * n + (colX+2*halfB)];
    sharedCol[(ty+halfB) * B + (tx+2*halfB)] = deviceDist[(colY+halfB) * n + (colX+2*halfB)];
    sharedCol[(ty+2*halfB) * B + (tx+2*halfB)] = deviceDist[(colY+2*halfB) * n + (colX+2*halfB)];
    sharedCol[(ty+3*halfB) * B + (tx+2*halfB)] = deviceDist[(colY+3*halfB) * n + (colX+2*halfB)];
    sharedCol[ty * B + (tx+3*halfB)] = deviceDist[colY * n + (colX+3*halfB)];
    sharedCol[(ty+halfB) * B + (tx+3*halfB)] = deviceDist[(colY+halfB) * n + (colX+3*halfB)];
    sharedCol[(ty+2*halfB) * B + (tx+3*halfB)] = deviceDist[(colY+2*halfB) * n + (colX+3*halfB)];
    sharedCol[(ty+3*halfB) * B + (tx+3*halfB)] = deviceDist[(colY+3*halfB) * n + (colX+3*halfB)];

    sharedRow[ty * B + tx] = deviceDist[rowY * n + rowX];
    sharedRow[(ty+halfB) * B + tx] = deviceDist[(rowY+halfB) * n + rowX];
    sharedRow[(ty+2*halfB) * B + tx] = deviceDist[(rowY+2*halfB) * n + rowX];
    sharedRow[(ty+3*halfB) * B + tx] = deviceDist[(rowY+3*halfB) * n + rowX];
    sharedRow[ty * B + (tx+halfB)] = deviceDist[rowY * n + (rowX+halfB)];
    sharedRow[(ty+halfB) * B + (tx+halfB)] = deviceDist[(rowY+halfB) * n + (rowX+halfB)];
    sharedRow[(ty+2*halfB) * B + (tx+halfB)] = deviceDist[(rowY+2*halfB) * n + (rowX+halfB)];
    sharedRow[(ty+3*halfB) * B + (tx+halfB)] = deviceDist[(rowY+3*halfB) * n + (rowX+halfB)];
    sharedRow[ty * B + (tx+2*halfB)] = deviceDist[rowY * n + (rowX+2*halfB)];
    sharedRow[(ty+halfB) * B + (tx+2*halfB)] = deviceDist[(rowY+halfB) * n + (rowX+2*halfB)];
    sharedRow[(ty+2*halfB) * B + (tx+2*halfB)] = deviceDist[(rowY+2*halfB) * n + (rowX+2*halfB)];
    sharedRow[(ty+3*halfB) * B + (tx+2*halfB)] = deviceDist[(rowY+3*halfB) * n + (rowX+2*halfB)];
    sharedRow[ty * B + (tx+3*halfB)] = deviceDist[rowY * n + (rowX+3*halfB)];
    sharedRow[(ty+halfB) * B + (tx+3*halfB)] = deviceDist[(rowY+halfB) * n + (rowX+3*halfB)];
    sharedRow[(ty+2*halfB) * B + (tx+3*halfB)] = deviceDist[(rowY+2*halfB) * n + (rowX+3*halfB)];
    sharedRow[(ty+3*halfB) * B + (tx+3*halfB)] = deviceDist[(rowY+3*halfB) * n + (rowX+3*halfB)];
    __syncthreads();
    // 每個cell的dependency都是row和column
    int tmpBlock1 = deviceDist[centerY * n + centerX];
    int tmpBlock2 = deviceDist[(centerY+halfB) * n + centerX];
    int tmpBlock3 = deviceDist[(centerY+2*halfB) * n + centerX];
    int tmpBlock4 = deviceDist[(centerY+3*halfB) * n + centerX];
    int tmpBlock5 = deviceDist[centerY * n + (centerX+halfB)];
    int tmpBlock6 = deviceDist[(centerY+halfB) * n + (centerX+halfB)];
    int tmpBlock7 = deviceDist[(centerY+2*halfB) * n + (centerX+halfB)];
    int tmpBlock8 = deviceDist[(centerY+3*halfB) * n + (centerX+halfB)];
    int tmpBlock9 = deviceDist[centerY * n + (centerX+2*halfB)];
    int tmpBlock10 = deviceDist[(centerY+halfB) * n + (centerX+2*halfB)];
    int tmpBlock11 = deviceDist[(centerY+2*halfB) * n + (centerX+2*halfB)];
    int tmpBlock12 = deviceDist[(centerY+3*halfB) * n + (centerX+2*halfB)];
    int tmpBlock13 = deviceDist[centerY * n + (centerX+3*halfB)];
    int tmpBlock14 = deviceDist[(centerY+halfB) * n + (centerX+3*halfB)];
    int tmpBlock15 = deviceDist[(centerY+2*halfB) * n + (centerX+3*halfB)];
    int tmpBlock16 = deviceDist[(centerY+3*halfB) * n + (centerX+3*halfB)];

    for (int k = 0; k < B; ++k) {
        // Update block
        tmpBlock1 = min(tmpBlock1, sharedCol[ty * B + k] + sharedRow[k * B + tx]);
        tmpBlock2 = min(tmpBlock2, sharedCol[(ty+halfB) * B + k] + sharedRow[k * B + tx]);
        tmpBlock3 = min(tmpBlock3, sharedCol[(ty+2*halfB) * B + k] + sharedRow[k * B + tx]);
        tmpBlock4 = min(tmpBlock4, sharedCol[(ty+3*halfB) * B + k] + sharedRow[k * B + tx]);
        tmpBlock5 = min(tmpBlock5, sharedCol[ty * B + k] + sharedRow[k * B + (tx+halfB)]);
        tmpBlock6 = min(tmpBlock6, sharedCol[(ty+halfB) * B + k] + sharedRow[k * B + (tx+halfB)]);
        tmpBlock7 = min(tmpBlock7, sharedCol[(ty+2*halfB) * B + k] + sharedRow[k * B + (tx+halfB)]);
        tmpBlock8 = min(tmpBlock8, sharedCol[(ty+3*halfB) * B + k] + sharedRow[k * B + (tx+halfB)]);
        tmpBlock9 = min(tmpBlock9, sharedCol[ty * B + k] + sharedRow[k * B + (tx+2*halfB)]);
        tmpBlock10 = min(tmpBlock10, sharedCol[(ty+halfB) * B + k] + sharedRow[k * B + (tx+2*halfB)]);
        tmpBlock11 = min(tmpBlock11, sharedCol[(ty+2*halfB) * B + k] + sharedRow[k * B + (tx+2*halfB)]);
        tmpBlock12 = min(tmpBlock12, sharedCol[(ty+3*halfB) * B + k] + sharedRow[k * B + (tx+2*halfB)]);
        tmpBlock13 = min(tmpBlock13, sharedCol[ty * B + k] + sharedRow[k * B + (tx+3*halfB)]);
        tmpBlock14 = min(tmpBlock14, sharedCol[(ty+halfB) * B + k] + sharedRow[k * B + (tx+3*halfB)]);
        tmpBlock15 = min(tmpBlock15, sharedCol[(ty+2*halfB) * B + k] + sharedRow[k * B + (tx+3*halfB)]);
        tmpBlock16 = min(tmpBlock16, sharedCol[(ty+3*halfB) * B + k] + sharedRow[k * B + (tx+3*halfB)]);
    }
    deviceDist[centerY * n + centerX] = tmpBlock1;
    deviceDist[(centerY+halfB) * n + centerX] = tmpBlock2;
    deviceDist[(centerY+2*halfB) * n + centerX] = tmpBlock3;
    deviceDist[(centerY+3*halfB) * n + centerX] = tmpBlock4;
    deviceDist[centerY * n + (centerX+halfB)] = tmpBlock5;
    deviceDist[(centerY+halfB) * n + (centerX+halfB)] = tmpBlock6;
    deviceDist[(centerY+2*halfB) * n + (centerX+halfB)] = tmpBlock7;
    deviceDist[(centerY+3*halfB) * n + (centerX+halfB)] = tmpBlock8;
    deviceDist[centerY * n + (centerX+2*halfB)] = tmpBlock9;
    deviceDist[(centerY+halfB) * n + (centerX+2*halfB)] = tmpBlock10;
    deviceDist[(centerY+2*halfB) * n + (centerX+2*halfB)] = tmpBlock11;
    deviceDist[(centerY+3*halfB) * n + (centerX+2*halfB)] = tmpBlock12;
    deviceDist[centerY * n + (centerX+3*halfB)] = tmpBlock13;
    deviceDist[(centerY+halfB) * n + (centerX+3*halfB)] = tmpBlock14;
    deviceDist[(centerY+2*halfB) * n + (centerX+3*halfB)] = tmpBlock15;
    deviceDist[(centerY+3*halfB) * n + (centerX+3*halfB)] = tmpBlock16;
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
    // pad_n是B的倍數
    int round = pad_n / B;
    dim3 threadsPerBlock(32, 32);
    dim3 gridsPerBlock1(1, 1);
    dim3 gridsPerBlock2(1, round);
    dim3 gridsPerBlock3(round, round);
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
