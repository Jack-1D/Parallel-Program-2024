#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <omp.h>
#include <mpi.h>
#include <immintrin.h>

void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

int main(int argc, char** argv) {
    // struct timespec start, end, tmp;
    // double totalTime;
    // clock_gettime(CLOCK_MONOTONIC, &start);

    int rank, size;
	MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    cpu_set_t cpu_set;
    omp_set_nested(true);
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);

    assert(argc == 9);
    const char* filename = argv[1];
    int iters = strtol(argv[2], 0, 10);
    double left = strtod(argv[3], 0);
    double right = strtod(argv[4], 0);
    double lower = strtod(argv[5], 0);
    double upper = strtod(argv[6], 0);
    int width = strtol(argv[7], 0, 10);
    int height = strtol(argv[8], 0, 10);

    int floorHeight = height / size;
    int extraRows = height % size;
    int rows = rank < extraRows ? floorHeight + 1 : floorHeight;

    int* gatherImage = (int*)malloc(width * height * sizeof(int));
    int* image = (int*)malloc(width * rows * sizeof(int));
    int* displs = (int*)malloc(size * sizeof(int));
    int* recvcounts = (int*)malloc(size * sizeof(int));
    
    // clock_gettime(CLOCK_MONOTONIC, &end);
    // if((end.tv_nsec - start.tv_nsec) < 0){
    //     tmp.tv_sec = end.tv_sec - start.tv_sec - 1;
    //     tmp.tv_nsec = end.tv_nsec - start.tv_nsec + 1e9;
    // }
    // else{
    //     tmp.tv_sec = end.tv_sec - start.tv_sec;
    //     tmp.tv_nsec = end.tv_nsec - start.tv_nsec;
    // }
    // totalTime = tmp.tv_sec + (double)tmp.tv_nsec / (double)1e9;
    // if(rank == 0)
    //     printf("Initialization: %f\n", totalTime);

    // clock_gettime(CLOCK_MONOTONIC, &start);

    #pragma omp parallel num_threads(CPU_COUNT(&cpu_set))
    {
        double startXVec[8], startYVec[8], repeatCpy[8], startYCpy[8], startXCpy[8], xCpy[8], yCpy[8], lengthSquaredCpy[8], x0Cpy[8], y0Cpy[8];
        #pragma omp for schedule(static, 10)
        for(int startY = 0;startY < rows;++startY){
            int startX = 0;
            for(int i=0;i<8;++i){
                startXVec[i] = (double)startX;
                startYVec[i] = (double)startY;
                ++startX;
            }
            __m512d mxVec = _mm512_set1_pd(0.0);
            __m512d myVec = _mm512_set1_pd(0.0);
            __m512d mstartXVec = _mm512_loadu_pd(startXVec);
            __m512d mstartYVec = _mm512_loadu_pd(startYVec);
            __m512d mlengthSquared = _mm512_set1_pd(0.0);
            __m512d mupperVec = _mm512_set1_pd(upper);
            __m512d mlowerVec = _mm512_set1_pd(lower);
            __m512d mrightVec = _mm512_set1_pd(right);
            __m512d mleftVec = _mm512_set1_pd(left);
            __m512d mheightVec = _mm512_set1_pd((double)height);
            __m512d mwidthVec = _mm512_set1_pd((double)width);

            __m512d mitersVec = _mm512_set1_pd((double)iters);
            __m512d mrepeatsVec = _mm512_set1_pd(0.0);

            __m512d mx0Vec = _mm512_fmadd_pd(mstartXVec, _mm512_div_pd(_mm512_sub_pd(mrightVec, mleftVec), mwidthVec), mleftVec);
            __m512d my0Vec = _mm512_fmadd_pd(_mm512_add_pd(_mm512_mul_pd(mstartYVec, _mm512_set1_pd((double)size)), _mm512_set1_pd((double)rank)), _mm512_div_pd(_mm512_sub_pd(mupperVec, mlowerVec), mheightVec), mlowerVec);
            int idx = -1;
            for(int iter = 8;iter < width;++iter){
                while(true){
                    __mmask8 mask4 = _mm512_cmp_pd_mask(mlengthSquared, _mm512_set1_pd(4.0), _CMP_GE_OQ);
                    __mmask8 maskRepeats = _mm512_cmp_pd_mask(mrepeatsVec, mitersVec, _CMP_GE_OQ);
                    bool breakFlag = false;
                    for(idx=0;idx<8;++idx){
                        if((mask4 & (1 << idx)) || (maskRepeats & (1 << idx))){
                            _mm512_store_pd(&xCpy, mxVec);
                            _mm512_store_pd(&yCpy, myVec);
                            _mm512_store_pd(&x0Cpy, mx0Vec);
                            _mm512_store_pd(&y0Cpy, my0Vec);
                            _mm512_store_pd(&repeatCpy, mrepeatsVec);
                            _mm512_store_pd(&startXCpy, mstartXVec);
                            _mm512_store_pd(&startYCpy, mstartYVec);
                            image[(int)startYCpy[idx] * width + (int)startXCpy[idx]] = (int)repeatCpy[idx];
                            _mm512_store_pd(&xCpy, mxVec);
                            _mm512_store_pd(&yCpy, myVec);
                            _mm512_store_pd(&x0Cpy, mx0Vec);
                            _mm512_store_pd(&y0Cpy, my0Vec);
                            _mm512_store_pd(&lengthSquaredCpy, mlengthSquared);
                            repeatCpy[idx] = 0.0;
                            startXCpy[idx] = (double)startX;
                            startYCpy[idx] = (double)startY;
                            xCpy[idx] = 0.0;
                            yCpy[idx] = 0.0;
                            lengthSquaredCpy[idx] = 0.0;
                            x0Cpy[idx] = startX * ((right - left) / width) + left;
                            y0Cpy[idx] = (((double)startY*size) + rank) * ((upper - lower) / height) + lower;
                            mrepeatsVec = _mm512_load_pd(&repeatCpy);
                            mstartXVec = _mm512_load_pd(&startXCpy);
                            mstartYVec = _mm512_load_pd(&startYCpy);
                            mxVec = _mm512_load_pd(&xCpy);
                            myVec = _mm512_load_pd(&yCpy);
                            mlengthSquared = _mm512_load_pd(&lengthSquaredCpy);
                            mx0Vec = _mm512_load_pd(&x0Cpy);
                            my0Vec = _mm512_load_pd(&y0Cpy);
                            ++startX;
                            breakFlag = true;
                            break;
                        }
                    }
                    if(breakFlag)
                        break;
                    __m512d tmpVec = _mm512_add_pd(_mm512_sub_pd(_mm512_mul_pd(mxVec, mxVec), _mm512_mul_pd(myVec, myVec)), mx0Vec);
                    myVec = _mm512_add_pd(_mm512_mul_pd(_mm512_set1_pd(2.0), _mm512_mul_pd(mxVec, myVec)), my0Vec);
                    mxVec = tmpVec;
                    mlengthSquared = _mm512_add_pd(_mm512_mul_pd(mxVec, mxVec), _mm512_mul_pd(myVec, myVec));
                    mrepeatsVec = _mm512_add_pd(mrepeatsVec, _mm512_set1_pd(1.0));
                }
            }
            for(int i=0;i<8;++i){
                int repeats = 0;
                double x = 0;
                double y = 0;
                double length_squared = 0;
                double y0 = ((int)startYCpy[i]*size+rank) * ((upper - lower) / height) + lower;
                double x0 = (int)startXCpy[i] * ((right - left) / width) + left;
                while (repeats < iters && length_squared < 4) {
                    double temp = x * x - y * y + x0;
                    y = 2 * x * y + y0;
                    x = temp;
                    length_squared = x * x + y * y;
                    ++repeats;
                }
                image[(int)startYCpy[i] * width + (int)startXCpy[i]] = repeats;
            }
            startX = 0;
        }
    }
    // clock_gettime(CLOCK_MONOTONIC, &end);
    // if((end.tv_nsec - start.tv_nsec) < 0){
    //     tmp.tv_sec = end.tv_sec - start.tv_sec - 1;
    //     tmp.tv_nsec = end.tv_nsec - start.tv_nsec + 1e9;
    // }
    // else{
    //     tmp.tv_sec = end.tv_sec - start.tv_sec;
    //     tmp.tv_nsec = end.tv_nsec - start.tv_nsec;
    // }
    // totalTime = tmp.tv_sec + (double)tmp.tv_nsec / (double)1e9;
    // if(rank == 0)
    //     printf("OpenMP time: %f\n", totalTime);

    displs[0] = 0;
    for(int i = 0;i < size;i++){
        if(i < extraRows)
            recvcounts[i] = (floorHeight + 1) * width;
        else
            recvcounts[i] = floorHeight * width;

        if(i != 0)
            displs[i] = displs[i-1] + recvcounts[i-1];
    }    

    // clock_gettime(CLOCK_MONOTONIC, &start);

    MPI_Gatherv(image, rows*width, MPI_INT, gatherImage, recvcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);

    // clock_gettime(CLOCK_MONOTONIC, &end);
    // if((end.tv_nsec - start.tv_nsec) < 0){
    //     tmp.tv_sec = end.tv_sec - start.tv_sec - 1;
    //     tmp.tv_nsec = end.tv_nsec - start.tv_nsec + 1e9;
    // }
    // else{
    //     tmp.tv_sec = end.tv_sec - start.tv_sec;
    //     tmp.tv_nsec = end.tv_nsec - start.tv_nsec;
    // }
    // totalTime = tmp.tv_sec + (double)tmp.tv_nsec / (double)1e9;
    // printf("Rank%d: %f\n", rank, totalTime);
    // if(rank == 0)
        // printf("Communication time: %f\n", totalTime);

    MPI_Finalize();

    if(rank==0){
        // clock_gettime(CLOCK_MONOTONIC, &start);
        int* final_image = (int*)malloc(width * height * sizeof(int));
        int index = 0;
        for(int i = 0;i < rows;i++){
            int stride = rows;
            int extraRowLeft = extraRows;
            for(int row = i; row < height && index < height*width; row += stride){
                for(int col = 0; col < width; col++){
                    final_image[index] = gatherImage[row*width + col];
                    index++;
                }
                --extraRowLeft;
                if(extraRowLeft < 0){
                    stride = floorHeight;
                }
            }
        }

        // clock_gettime(CLOCK_MONOTONIC, &end);
        // if((end.tv_nsec - start.tv_nsec) < 0){
        //     tmp.tv_sec = end.tv_sec - start.tv_sec - 1;
        //     tmp.tv_nsec = end.tv_nsec - start.tv_nsec + 1e9;
        // }
        // else{
        //     tmp.tv_sec = end.tv_sec - start.tv_sec;
        //     tmp.tv_nsec = end.tv_nsec - start.tv_nsec;
        // }
        // totalTime = tmp.tv_sec + (double)tmp.tv_nsec / (double)1e9;
        // printf("Gather image time: %f\n", totalTime);

        // clock_gettime(CLOCK_MONOTONIC, &start);
        write_png(filename, iters, width, height, final_image);
        // clock_gettime(CLOCK_MONOTONIC, &end);
        // if((end.tv_nsec - start.tv_nsec) < 0){
        //     tmp.tv_sec = end.tv_sec - start.tv_sec - 1;
        //     tmp.tv_nsec = end.tv_nsec - start.tv_nsec + 1e9;
        // }
        // else{
        //     tmp.tv_sec = end.tv_sec - start.tv_sec;
        //     tmp.tv_nsec = end.tv_nsec - start.tv_nsec;
        // }
        // totalTime = tmp.tv_sec + (double)tmp.tv_nsec / (double)1e9;
        // printf("Write PNG time: %f\n", totalTime);

        free(final_image);
    }
}
