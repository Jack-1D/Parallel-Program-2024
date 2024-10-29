#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <boost/sort/spreadsort/spreadsort.hpp>

float* combine;

int ceil(int a, int b){
    if(a % b == 0)
        return a / b;
    return a / b + 1;
}

int max(int a, int b){
    return a > b ? a : b;
}

int swapper(float* arr, float* res, int dataSize, int nxtSize, bool smallToLarge){
    int nonSwapped = 1;
    
    int arrPtr = 0, resPtr = 0, combinePtr = 0;
    // 小rank排序
    if(!smallToLarge){
        if(arr[dataSize-1] <= res[0])
            return nonSwapped;
        while(arrPtr < dataSize && resPtr < nxtSize && combinePtr < dataSize){
            if(arr[arrPtr] < res[resPtr]){
                combine[combinePtr] = arr[arrPtr];
                ++arrPtr;
            }
            else{
                combine[combinePtr] = res[resPtr];
                ++resPtr;
            }
            ++combinePtr;
        }
        while(combinePtr < dataSize){
            if(arrPtr < dataSize){
                combine[combinePtr] = arr[arrPtr];
                ++arrPtr;
            }
            else{
                combine[combinePtr] = res[resPtr];
                ++resPtr;
            }
            ++combinePtr;
        }
    }
    // 大rank排序
    else{
        if(arr[0] >= res[nxtSize-1])
            return nonSwapped;
        arrPtr = dataSize - 1;
        resPtr = nxtSize - 1;
        combinePtr = dataSize - 1;
        while(arrPtr >= 0 && resPtr >= 0 && combinePtr >= 0){
            if(arr[arrPtr] > res[resPtr]){
                combine[combinePtr] = arr[arrPtr];
                --arrPtr;
            }
            else{
                combine[combinePtr] = res[resPtr];
                --resPtr;
            }
            --combinePtr;
        }
        while(combinePtr >= 0){
            if(arrPtr >= 0){
                combine[combinePtr] = arr[arrPtr];
                --arrPtr;
            }
            else{
                combine[combinePtr] = res[resPtr];
                --resPtr;
            }
            --combinePtr;
        }
    }
    for(int i=0;i<dataSize;++i){
        arr[i] = combine[i];
    }
    return !nonSwapped;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Group world_group, new_group;
    MPI_Comm new_comm;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);

    int arr_size = atoi(argv[1]);
    char *input_filename = argv[2];
    char *output_filename = argv[3];

    // rank-1的資料量
    int preSize = rank - 1 < (arr_size%size) ? arr_size/size + 1 : arr_size/size;
    // rank的資料量
    int dataSize = rank < (arr_size%size) ? arr_size/size + 1 : arr_size/size;
    // rank+1的資料量
    int nxtSize = rank + 1 < (arr_size%size) ? arr_size/size + 1 : arr_size/size;
    // 參與計算的process數量
    int involveSize = arr_size > size ? size : arr_size;
    int processIncluded[involveSize] = {0};
    for(int i=0;i<involveSize;++i)
        processIncluded[i] = i;
    MPI_Group_incl(world_group, involveSize, processIncluded, &new_group);
    MPI_Comm_create(MPI_COMM_WORLD, new_group, &new_comm);
    combine = (float*)malloc(max(preSize, max(dataSize, nxtSize))*sizeof(float));
    float* arr, *res;
    // 是否參與計算
    bool involved = false;
    // 讀檔起始點
    int beginOffset;
    if(arr_size % size == 0)
        beginOffset = rank*(arr_size/size);
    else
        beginOffset = rank <= (arr_size%size) ? rank*ceil(arr_size, size) : rank*ceil(arr_size, size)-((rank-(arr_size%size)+size)%size);
    
    if(dataSize > 0){
        arr = (float*)malloc(dataSize*sizeof(float));
        res = (float*)malloc(max(nxtSize, preSize)*sizeof(float));
        involved = true;
    }
    else{
        beginOffset = 0;
    }

    // double start, end, start1, end1;
    // start = MPI_Wtime();
    
    // Open file
    MPI_File input_file, output_file;
    if(dataSize > 0){
        MPI_File_open(new_comm, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
        MPI_File_read_at(input_file, sizeof(float) * beginOffset, arr, dataSize, MPI_FLOAT, MPI_STATUS_IGNORE);
        MPI_File_close(&input_file);
    }
    // end = MPI_Wtime();
    // if(rank == 1)
    //     printf("Read file: %f sec\n", end - start);
    
    // double computation_time = 0, communication_time = 0;
    // start = MPI_Wtime();
    if(involved)
        boost::sort::spreadsort::spreadsort(arr, arr + dataSize);
        // qsort(arr, dataSize, sizeof(float), cmp);
    // end = MPI_Wtime();
    // if(rank == 1)
    //     computation_time += end - start;

    int swappedTotal;
    int round = 0;
    bool finishOnePhase = false;

    while(involved && size > 1){
        // start = MPI_Wtime();
        int nonSwapped = 1;
        // even phase
        if(!(round & 1)){
            // odd rank
            if(rank & 1){
                // MPI_Send(arr, dataSize, MPI_FLOAT, rank-1, 0, new_comm);
                // MPI_Recv(arr, dataSize, MPI_FLOAT, rank-1, 1, new_comm, MPI_STATUS_IGNORE);
                MPI_Sendrecv(arr, dataSize, MPI_FLOAT, rank-1, 2,
                 res, preSize, MPI_FLOAT, rank-1, 2, new_comm, MPI_STATUS_IGNORE);
                // start1 = MPI_Wtime();
                nonSwapped = swapper(arr, res, dataSize, preSize, true);
                // end1 = MPI_Wtime();
            }
            // even rank
            else{
                if(rank+1 < involveSize){
                    MPI_Sendrecv(arr, dataSize, MPI_FLOAT, rank+1, 2,
                        res, nxtSize, MPI_FLOAT, rank+1, 2, new_comm, MPI_STATUS_IGNORE);
                    nonSwapped = swapper(arr, res, dataSize, nxtSize, false);
                }
            }
        }
        // odd phase
        else{
            // even rank
            if(!(rank & 1)){
                if(rank != 0){
                    MPI_Sendrecv(arr, dataSize, MPI_FLOAT, rank-1, 1,
                    res, preSize, MPI_FLOAT, rank-1, 1, new_comm, MPI_STATUS_IGNORE);
                    nonSwapped = swapper(arr, res, dataSize, preSize, true);
                }
            }
            // odd rank
            else{
                if(rank != involveSize-1){
                    MPI_Sendrecv(arr, dataSize, MPI_FLOAT, rank+1, 1,
                        res, nxtSize, MPI_FLOAT, rank+1, 1, new_comm, MPI_STATUS_IGNORE);
                    // start1 = MPI_Wtime();
                    nonSwapped = swapper(arr, res, dataSize, nxtSize, false);
                    // end1 = MPI_Wtime();
                }
            }
        }
        MPI_Barrier(new_comm);
        MPI_Allreduce(&nonSwapped, &swappedTotal, 1, MPI_INT, MPI_SUM, new_comm);
        // end = MPI_Wtime();
        if(swappedTotal == involveSize){
            if(finishOnePhase)
                break;
            else
                finishOnePhase = true;
        }
        else
            finishOnePhase = false;
        ++round;
        // if(rank == 1){
        //     computation_time += end1 - start1;
        //     communication_time += end - start - (end1 - start1);
        // }
    }
    // if(rank == 1){
    //     printf("Computation time: %f\n", computation_time);
    //     printf("Communication time: %f\n", communication_time);
    // }

    // start = MPI_Wtime();

    if(involved){
        MPI_File_open(new_comm, output_filename, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
        MPI_File_write_at(output_file, sizeof(float) * beginOffset, arr, dataSize, MPI_FLOAT, MPI_STATUS_IGNORE);
        MPI_File_close(&output_file);
    }
    // end = MPI_Wtime();
    // if(rank == 1){
    //     printf("Write file: %f sec\n", end - start);
    // }
    MPI_Finalize();
    return 0;    
}