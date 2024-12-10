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
size_t pitch;

void block_FW(int *deviceDist);

int ceil(int a, int b) { return (a + b - 1) / b; }

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    pad_n = ceil(n, B) * B;

    // cudaMallocHost((void**)&hostDist, pad_n * pad_n * sizeof(int), cudaHostAllocDefault);
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
    for(int i=0; i < n; ++i){
        fwrite(hostDist + i * pad_n, sizeof(int), n, outfile);
    }
    fclose(outfile);
}


__global__ void cal1(
    int Round, int *deviceDist, int pitch) {

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // GPU block idx平移到要計算的block上(單位cell)
    int block_internal_start_x = Round * B;
    int block_internal_start_y = Round * B;

    int j = block_internal_start_x + tx;
    int i = block_internal_start_y + ty;

    __shared__ int sharedDist[B * (B+1)];
    sharedDist[ty * (B+1) + tx] = deviceDist[i * pitch + j];
    sharedDist[(ty+halfB) * (B+1) + tx] = deviceDist[(i+halfB) * pitch + j];
    sharedDist[ty * (B+1) + (tx+halfB)] = deviceDist[i * pitch + (j+halfB)];
    sharedDist[(ty+halfB) * (B+1) + (tx+halfB)] = deviceDist[(i+halfB) * pitch + (j+halfB)];
    __syncthreads();

    #pragma unroll 64
    for (int k = 0; k < B; ++k) {
        // sharedDist[ty * B + tx] = min(sharedDist[ty * B + tx], sharedDist[ty * B + k] + sharedDist[k * B + tx]);
        if (sharedDist[ty * (B+1) + k] + sharedDist[k * (B+1) + tx] < sharedDist[ty * (B+1) + tx]) {
            sharedDist[ty * (B+1) + tx] = sharedDist[ty * (B+1) + k] + sharedDist[k * (B+1) + tx];
        }
        if (sharedDist[(ty+halfB) * (B+1) + k] + sharedDist[k * (B+1) + tx] < sharedDist[(ty+halfB) * (B+1) + tx]) {
            sharedDist[(ty+halfB) * (B+1) + tx] = sharedDist[(ty+halfB) * (B+1) + k] + sharedDist[k * (B+1) + tx];
        }
        if (sharedDist[ty * (B+1) + k] + sharedDist[k * (B+1) + (tx+halfB)] < sharedDist[ty * (B+1) + (tx+halfB)]) {
            sharedDist[ty * (B+1) + (tx+halfB)] = sharedDist[ty * (B+1) + k] + sharedDist[k * (B+1) + (tx+halfB)];
        }
        if (sharedDist[(ty+halfB) * (B+1) + k] + sharedDist[k * (B+1) + (tx+halfB)] < sharedDist[(ty+halfB) * (B+1) + (tx+halfB)]) {
            sharedDist[(ty+halfB) * (B+1) + (tx+halfB)] = sharedDist[(ty+halfB) * (B+1) + k] + sharedDist[k * (B+1) + (tx+halfB)];
        }
        __syncthreads();
    }
    deviceDist[i * pitch + j] = sharedDist[ty * (B+1) + tx];
    deviceDist[(i+halfB) * pitch + j] = sharedDist[(ty+halfB) * (B+1) + tx];
    deviceDist[i * pitch + (j+halfB)] = sharedDist[ty * (B+1) + (tx+halfB)];
    deviceDist[(i+halfB) * pitch + (j+halfB)] = sharedDist[(ty+halfB) * (B+1) + (tx+halfB)];
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
    int Round, int *deviceDist, int pitch) {
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

    __shared__ int sharedCol[B * (B+1)];
    __shared__ int sharedRow[B * (B+1)];

    sharedCol[ty * (B+1) + tx] = deviceDist[colY * pitch + colX];
    sharedCol[(ty+halfB) * (B+1) + tx] = deviceDist[(colY+halfB) * pitch + colX];
    sharedCol[ty * (B+1) + (tx+halfB)] = deviceDist[colY * pitch + (colX+halfB)];
    sharedCol[(ty+halfB) * (B+1) + (tx+halfB)] = deviceDist[(colY+halfB) * pitch + (colX+halfB)];
    sharedRow[ty * (B+1) + tx] = deviceDist[rowY * pitch + rowX];
    sharedRow[(ty+halfB) * (B+1) + tx] = deviceDist[(rowY+halfB) * pitch + rowX];
    sharedRow[ty * (B+1) + (tx+halfB)] = deviceDist[rowY * pitch + (rowX+halfB)];
    sharedRow[(ty+halfB) * (B+1) + (tx+halfB)] = deviceDist[(rowY+halfB) * pitch + (rowX+halfB)];
    __syncthreads();
    // 每個cell的dependency都是row和column
    int tmpBlock1 = deviceDist[centerY * pitch + centerX];
    int tmpBlock2 = deviceDist[(centerY+halfB) * pitch + centerX];
    int tmpBlock3 = deviceDist[centerY * pitch + (centerX+halfB)];
    int tmpBlock4 = deviceDist[(centerY+halfB) * pitch + (centerX+halfB)];

    #pragma unroll 64
    for (int k = 0; k < B; ++k) {
        // tmpBlock = min(tmpBlock, sharedCol[ty * B + k] + sharedRow[k * B + tx]);
        // Update block
        if (sharedCol[ty * (B+1) + k] + sharedRow[k * (B+1) + tx] < tmpBlock1) {
            tmpBlock1 = sharedCol[ty * (B+1) + k] + sharedRow[k * (B+1) + tx];
        }
        if (sharedCol[(ty+halfB) * (B+1) + k] + sharedRow[k * (B+1) + tx] < tmpBlock2) {
            tmpBlock2 = sharedCol[(ty+halfB) * (B+1) + k] + sharedRow[k * (B+1) + tx];
        }
        if (sharedCol[ty * (B+1) + k] + sharedRow[k * (B+1) + (tx+halfB)] < tmpBlock3) {
            tmpBlock3 = sharedCol[ty * (B+1) + k] + sharedRow[k * (B+1) + (tx+halfB)];
        }
        if (sharedCol[(ty+halfB) * (B+1) + k] + sharedRow[k * (B+1) + (tx+halfB)] < tmpBlock4) {
            tmpBlock4 = sharedCol[(ty+halfB) * (B+1) + k] + sharedRow[k * (B+1) + (tx+halfB)];
        }
    }
    deviceDist[centerY * pitch + centerX] = tmpBlock1;
    deviceDist[(centerY+halfB) * pitch + centerX] = tmpBlock2;
    deviceDist[centerY * pitch + (centerX+halfB)] = tmpBlock3;
    deviceDist[(centerY+halfB) * pitch + (centerX+halfB)] = tmpBlock4;
}


int main(int argc, char* argv[]) {
    input(argv[1]);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaHostRegister(hostDist, pad_n * pad_n * sizeof(int), cudaHostAllocDefault);
    cudaMallocPitch(&deviceDist, &pitch, pad_n * sizeof(int), pad_n);
    cudaMemcpy2DAsync(deviceDist, pitch, hostDist, pad_n * sizeof(int), pad_n * sizeof(int), pad_n, cudaMemcpyHostToDevice, stream);
    // cudaMalloc(&deviceDist, pad_n * pad_n * sizeof(int));
    // cudaMemcpy(deviceDist, hostDist, pad_n * pad_n * sizeof(int), cudaMemcpyHostToDevice);

    cudaGetDeviceProperties(&prop, DEV_NO);
    printf("maxThreadsPerBlock = %d, sharedMemPerBlock = %d\n", prop.maxThreadsPerBlock, prop.sharedMemPerBlock);

    block_FW(deviceDist);
    cudaMemcpy2DAsync(hostDist, pad_n * sizeof(int), deviceDist, pitch, pad_n * sizeof(int), pad_n, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    // cudaMemcpy(hostDist, deviceDist, pad_n * pad_n * sizeof(int), cudaMemcpyDeviceToHost);
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
        cal1<<<gridsPerBlock1, threadsPerBlock>>>(r, deviceDist, pitch / sizeof(int));

        /* Phase 2*/
        cal2<<<gridsPerBlock2, threadsPerBlock>>>(r, deviceDist, pitch / sizeof(int));

        /* Phase 3*/
        cal3<<<gridsPerBlock3, threadsPerBlock>>>(r, deviceDist, pitch / sizeof(int));
    }
}
