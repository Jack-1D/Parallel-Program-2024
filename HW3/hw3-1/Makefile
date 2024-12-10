CC = mpicc
CXX = mpicxx 
LDLIBS = -pthread
CFLAGS = -O3 -fopenmp -fprefetch-loop-arrays
CXXFLAGS = -O3 -fopenmp -fprefetch-loop-arrays
TARGETS = hw3-1

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS) $(TARGETS:=.o)
