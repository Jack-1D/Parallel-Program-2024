#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>

unsigned long long r, k, ncpus;

void* calculate(void* pid){
	unsigned long long* pId = (unsigned long long*) pid;
	unsigned long long y, pixels = 0;
	for(unsigned long long i=*pId;i<r;i+=ncpus) {
		y = ceil(sqrtl(r*r - i*i));
		pixels += y;
		if(pixels >= k)
			pixels -= k;
	}
	unsigned long long* p_arr = (unsigned long long*)malloc(sizeof(unsigned long long)*1);
	p_arr[0] = pixels;
	pthread_exit((void*)p_arr);
}

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	r = atoll(argv[1]);
	k = atoll(argv[2]);
	unsigned long long pixels = 0;
	cpu_set_t cpuset;
	sched_getaffinity(0, sizeof(cpuset), &cpuset);
	ncpus = CPU_COUNT(&cpuset);

	unsigned long long pid[ncpus];
	pthread_t thread_pool[ncpus];
	for(int i=0;i<ncpus;++i){
		pid[i] = i;
		pthread_create(&thread_pool[i], NULL, calculate, (void*)&pid[i]);
	}
	void* ret;
	for(unsigned long long i=0;i<ncpus;++i){
		pthread_join(thread_pool[i], &ret);
		pixels += *(unsigned long long*)ret;
		if(pixels >= k)
			pixels -= k;
	}
	
	printf("%llu\n", (4 * pixels) % k);
}

