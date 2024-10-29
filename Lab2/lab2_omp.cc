#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long pixels = 0;

#pragma omp parallel
	{
		unsigned long long y = 0;
#pragma omp for schedule(guided)
	for(unsigned long long i=0;i<r;++i) {
		y += ceil(sqrtl(r*r - i*i));
		if(y >= k)
			y -= k;
	}
	#pragma omp critical
		{
			pixels += y;
			if(pixels >= k)
				pixels -= k;
		}
	}

	printf("%llu\n", (4 * pixels) % k);
}

