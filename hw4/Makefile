CC = gcc
NVCC = nvcc

CFLAGS = -Wall -g
NVFLAGS = -std=c++11 -O3 -Xptxas="-v" -arch=sm_61 -use_fast_math -ftz=true -prec-div=false -prec-sqrt=false -fmad=true
LDFLAGS = -lm

TARGETS = seq-attention seq-flashattention hw4 hw4-cmp

all: $(TARGETS)

clean:
	rm -f $(TARGETS)

.PHONY: all clean

seq-attention: seq-attention.c
	$(CC) $(CFLAGS) -o $@ $? $(LDFLAGS)

seq-flashattention: seq-flashattention.c
	$(CC) $(CFLAGS) -o $@ $? $(LDFLAGS)

hw4: hw4.cu
	$(NVCC) $(NVFLAGS) -o $@ $? $(LDFLAGS)

hw4-cmp: hw4-cmp.cu
	$(NVCC) $(NVFLAGS) -o $@ $? $(LDFLAGS)

