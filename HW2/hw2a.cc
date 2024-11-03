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
#include <immintrin.h>

int min(int a, int b){
    return a > b ? b : a;
}

pthread_mutex_t mutex;

int iters;
int* image;
double left;
double right;
double lower;
double upper;
int width;
int height;

int curX = 0, curY = 0, blockPerThread = 2500;

int jobRemain(){
    // Return available block size
    // Guarantee cur not pass limit
    int totalBlock = width * height;
    int curSpanX = curX + curY * width;
    if(curSpanX + blockPerThread < totalBlock)
        return blockPerThread;
    else if(curSpanX >= totalBlock)
        return -1;
    else
        return totalBlock - curSpanX;
}

void* calculate(void* pid){
    while(true){
        pthread_mutex_lock(&mutex);
        int block = jobRemain();
        int startX, startY;
        if(block == -1){
            pthread_mutex_unlock(&mutex);
            break;
        }
        else{
            startX = curX;
            startY = curY;
            curX += block;
            while(curX >= width){
                curX -= width;
                ++curY;
            }
            pthread_mutex_unlock(&mutex);
        }
        int idx = -1;
        int iter = 0;

        double startXVec[8], startYVec[8], repeatCpy[8], startYCpy[8], startXCpy[8], xCpy[8], yCpy[8], lengthSquaredCpy[8], x0Cpy[8], y0Cpy[8];
        for(int i=0;i<min(8, block);++i){
            // ++iter;
            startXVec[i] = (double)startX;
            startYVec[i] = (double)startY;
            ++startX;
            if(startX >= width){
                startX = 0;
                ++startY;
            }
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
        __m512d my0Vec = _mm512_fmadd_pd(mstartYVec, _mm512_div_pd(_mm512_sub_pd(mupperVec, mlowerVec), mheightVec), mlowerVec);

        while(iter < block){
            __mmask8 mask4 = _mm512_cmp_pd_mask(mlengthSquared, _mm512_set1_pd(4.0), _CMP_GE_OQ);
            __mmask8 maskRepeats = _mm512_cmp_pd_mask(mrepeatsVec, mitersVec, _CMP_GE_OQ);
            // __mmask8 maskEmpty = _mm512_cmp_pd_mask(mrepeatsVec, _mm512_set1_pd(-1.0), _CMP_EQ_OQ);
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
                    if(iter == block)
                        break;
                    else{
                        ++iter;
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
                        y0Cpy[idx] = startY * ((upper - lower) / height) + lower;
                        mrepeatsVec = _mm512_load_pd(&repeatCpy);
                        mstartXVec = _mm512_load_pd(&startXCpy);
                        mstartYVec = _mm512_load_pd(&startYCpy);
                        mxVec = _mm512_load_pd(&xCpy);
                        myVec = _mm512_load_pd(&yCpy);
                        mlengthSquared = _mm512_load_pd(&lengthSquaredCpy);
                        mx0Vec = _mm512_load_pd(&x0Cpy);
                        my0Vec = _mm512_load_pd(&y0Cpy);
                        ++startX;
                        if(startX >= width){
                            startX = 0;
                            ++startY;
                        }
                    }
                }
            }
            __m512d tmpVec = _mm512_add_pd(_mm512_sub_pd(_mm512_mul_pd(mxVec, mxVec), _mm512_mul_pd(myVec, myVec)), mx0Vec);
            myVec = _mm512_add_pd(_mm512_mul_pd(_mm512_set1_pd(2.0), _mm512_mul_pd(mxVec, myVec)), my0Vec);
            mxVec = tmpVec;
            mlengthSquared = _mm512_add_pd(_mm512_mul_pd(mxVec, mxVec), _mm512_mul_pd(myVec, myVec));
            mrepeatsVec = _mm512_add_pd(mrepeatsVec, _mm512_set1_pd(1.0));
        }
        if(block != blockPerThread)
            break;
    }
    pthread_exit(NULL);
}

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

    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    int ncpus = CPU_COUNT(&cpu_set);
    pthread_t thread_pool[ncpus];
    pthread_mutex_init(&mutex, NULL);

    /* argument parsing */
    assert(argc == 9);
    const char* filename = argv[1];
    iters = strtol(argv[2], 0, 10);
    left = strtod(argv[3], 0);
    right = strtod(argv[4], 0);
    lower = strtod(argv[5], 0);
    upper = strtod(argv[6], 0);
    width = strtol(argv[7], 0, 10);
    height = strtol(argv[8], 0, 10);

    /* allocate memory for image */
    image = (int*)malloc(width * height * sizeof(int));
    assert(image);
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
    // printf("Initialization: %f\n", totalTime);

    
    // clock_gettime(CLOCK_MONOTONIC, &start);
    
    for(int i = 0;i < ncpus;++i)
        pthread_create(&thread_pool[i], NULL, calculate, NULL);
    
    for(int i=0;i < ncpus;++i)
        pthread_join(thread_pool[i], NULL);
    
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
    // printf("Thread compute time: %f\n", totalTime);

    /* draw and cleanup */
    // clock_gettime(CLOCK_MONOTONIC, &start);
    write_png(filename, iters, width, height, image);
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

    free(image);
}
