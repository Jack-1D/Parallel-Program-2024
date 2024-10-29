#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long pixels = 0;
	unsigned long long res;
	int size, rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

#pragma omp parallel
	{
		unsigned long long y = 0;
		#pragma omp for schedule(guided)
			for(unsigned long long i=rank;i<r;i+=size){
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

	if(rank == 0){
		for(unsigned long long i=1;i<size;++i){
			MPI_Recv(&res, 1, MPI_UNSIGNED_LONG_LONG, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			pixels += res;
			if(pixels >= k)
				pixels -= k;
		}
	} else {
		MPI_Request request;
		MPI_Isend(&pixels, 1, MPI_UNSIGNED_LONG_LONG, 0, 1, MPI_COMM_WORLD, &request);
		MPI_Wait(&request, MPI_STATUS_IGNORE);
	}
	MPI_Finalize();

	if(rank == 0){
		printf("%llu\n", (4 * pixels) % k);
	}
}

