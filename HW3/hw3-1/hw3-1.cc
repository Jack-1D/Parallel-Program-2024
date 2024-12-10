#include <stdio.h>
#include <stdlib.h>
#include <sched.h>
#include <omp.h>
#include <pthread.h>
#define CHUNK_SIZE 10

const int INF = ((1 << 30) - 1);
const int V = 50010;
void input(char* inFileName);
void output(char* outFileName);

void block_FW(int B);
int ceil(int a, int b);
int max(int a, int b);
void cal(int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height);

int n, m, NUM_THREADS;
static int Dist[V][V];

pthread_mutex_t lock;
pthread_barrier_t barrier;

int min(int a, int b){
    return a > b ? b : a;
}


// const int jobSize = 50000;
// int src = 0, dst = 0;

// int jobRemain(){
//     int span = dst * n + src, nSquare = n * n;
//     if(span + 10 < nSquare)
//         return jobSize;
//     else if(span >= nSquare)
//         return -1;
//     else
//         return nSquare - span;
// }

// void* floyd_warshall(void*){
//     int tmpSrc, tmpDst, job;
//     for(int k = 0;k < n;++k){
//         while(true){
//             pthread_mutex_lock(&lock);
//             job = jobRemain();
//             // printf("job: %d\n", job);
//             if(job == -1){
//                 pthread_mutex_unlock(&lock);
//                 break;
//             }
//             else{
//                 tmpSrc = src;
//                 tmpDst = dst;
//                 src += job;
//                 while(src >= n){
//                     src -= n;
//                     ++dst;
//                 }
//                 pthread_mutex_unlock(&lock);
//             }
//             for(int i = 0;i < job;++i){
//                 if(Dist[tmpSrc][tmpDst] > Dist[tmpSrc][k] + Dist[k][tmpDst]){
//                     // printf("%d > %d, k: %d\n", Dist[tmpSrc][tmpDst], Dist[tmpSrc][k] + Dist[k][tmpDst], k);
//                     Dist[tmpSrc][tmpDst] = Dist[tmpSrc][k] + Dist[k][tmpDst];
//                 }
//                 ++tmpSrc;
//                 if(tmpSrc == n){
//                     tmpSrc = 0;
//                     ++tmpDst;
//                 }
//             }
//         }
//         pthread_barrier_wait(&barrier);
//         pthread_mutex_lock(&lock);
//         src = 0;
//         dst = 0;
//         pthread_mutex_unlock(&lock);
//         pthread_barrier_wait(&barrier);
//     }
//     pthread_exit(NULL);
// }

int startLine = 0;
int jobRemain(){
    if(startLine >= n)
        return -1;
    else{
        ++startLine;
        return startLine - 1;
    }
}

void* floyd_warshall(void* args){
    int* tid = (int*)args;
    int k, i, j, tmpDist;
    int jobPerThread = ceil(n * n, NUM_THREADS);

    // int startX = *tid * ceil(nSquare, NUM_THREADS) % n;
    // int startY = *tid * ceil(nSquare, NUM_THREADS) / n;
    int curX, curY;
    for(k = 0;k < n;++k){
        for(i = *tid;i < n;i += NUM_THREADS){
            tmpDist = Dist[i][k];
            for(j = 0;j < n;++j){
                if(Dist[i][j] > Dist[i][k] + Dist[k][j]){
                    Dist[i][j] = Dist[i][k] + Dist[k][j];
                }
            }
        }
        pthread_barrier_wait(&barrier);
        // while(true){
        //     pthread_mutex_lock(&lock);
        //     int line = jobRemain();
        //     pthread_mutex_unlock(&lock);
        //     if(line == -1)
        //         break;
        //     tmpDist = Dist[line][k];
        //     for(j = 0;j < n;++j){
        //         if(Dist[line][j] > tmpDist + Dist[k][j]){
        //             Dist[line][j] = tmpDist + Dist[k][j];
        //         }
        //     }
        // }
        // pthread_barrier_wait(&barrier);
        // startLine = 0;

        // curX = *tid * jobPerThread;
        // for(int i = 0; i < jobPerThread; ++i){
        //     if(Dist[curX/n][curX%n] > Dist[curX/n][k] + Dist[k][curX%n])
        //         Dist[curX/n][curX%n] = Dist[curX/n][k] + Dist[k][curX%n];
        //     ++curX;
        // }
        // pthread_barrier_wait(&barrier);
    }
    pthread_exit(NULL);
}

void fork_floyd_warshall(int numThread){
    int pid[numThread];
    pthread_t threadPool[NUM_THREADS];
    pthread_barrier_init(&barrier, NULL, NUM_THREADS);
    for(int i=0;i<numThread;++i){
        pid[i] = i;
        pthread_create(&threadPool[i], NULL, floyd_warshall, (void*)&pid[i]);
    }
    for(int i=0;i<numThread;++i){
        pthread_join(threadPool[i], NULL);
    }
}

int main(int argc, char* argv[]) {
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    NUM_THREADS = CPU_COUNT(&cpu_set);
    printf("ncpus: %d\n", NUM_THREADS);
    input(argv[1]);
    int B = 512;
    fork_floyd_warshall(NUM_THREADS);
    // block_FW(B);
    output(argv[2]);
    return 0;
}

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);
    printf("n: %d, m: %d\n", n, m);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                Dist[i][j] = 0;
            } else {
                Dist[i][j] = INF;
            }
        }
    }

    int pair[3];
    for (int i = 0; i < m; ++i) {
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0]][pair[1]] = pair[2];
    }
    fclose(file);
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (Dist[i][j] >= INF) Dist[i][j] = INF;
        }
        fwrite(Dist[i], sizeof(int), n, outfile);
    }
    fclose(outfile);
}

int ceil(int a, int b) { return (a + b - 1) / b; }
int max(int a, int b) {return a > b ? a : b ;}

void block_FW(int B) {
    int round = ceil(n, B);
    for (int r = 0; r < round; ++r) {
        printf("%d %d\n", r, round);
        fflush(stdout);
        /* Phase 1*/
        cal(B, r, r, r, 1, 1);

        /* Phase 2*/
        // Up
        cal(B, r, r, 0, r, 1);
        // Down
        cal(B, r, r, r + 1, round - r - 1, 1);
        // Left
        cal(B, r, 0, r, 1, r);
        // Right
        cal(B, r, r + 1, r, 1, round - r - 1);

        /* Phase 3*/
        // Upper left
        cal(B, r, 0, 0, r, r);
        // Lower left
        cal(B, r, 0, r + 1, round - r - 1, r);
        // Upper right
        cal(B, r, r + 1, 0, r, round - r - 1);
        // Lower right
        cal(B, r, r + 1, r + 1, round - r - 1, round - r - 1);
    }
}

void cal(
    int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height) {
#pragma omp parallel num_threads(NUM_THREADS)
{
    int block_end_x = block_start_x + block_height;
    int block_end_y = block_start_y + block_width;
    #pragma omp for schedule(dynamic) collapse(2)
    for (int b_i = block_start_x; b_i < block_end_x; ++b_i) {
        for (int b_j = block_start_y; b_j < block_end_y; ++b_j) {
            // To calculate B*B elements in the block (b_i, b_j)
            // For each block, it need to compute B times
            int block_internal_start_x = b_i * B;
            int block_internal_end_x = (b_i + 1) * B > n ? n : (b_i + 1) * B;
            int block_internal_start_y = b_j * B;
            int block_internal_end_y = (b_j + 1) * B > n ? n : (b_j + 1) * B;

            // if (block_internal_end_x > n) block_internal_end_x = n;
            // if (block_internal_end_y > n) block_internal_end_y = n;
            for (int k = Round * B; k < (Round + 1) * B && k < n; ++k) {
                // To calculate original index of elements in the block (b_i, b_j)
                // For instance, original index of (0,0) in block (1,2) is (2,5) for V=6,B=2
                for (int i = block_internal_start_x; i < block_internal_end_x; ++i) {
                    int dik = Dist[i][k];
                    if (dik == INF) continue;
                    #pragma omp reduction(+:Dist)
                    for (int j = block_internal_start_y; j < block_internal_end_y; ++j) {
                        if (Dist[k][j] != INF && dik + Dist[k][j] < Dist[i][j]) {
                            Dist[i][j] = dik + Dist[k][j];
                        }
                    }
                }
            }
        }
    }

}
}