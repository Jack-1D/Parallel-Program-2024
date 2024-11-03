CC = gcc
CXX = g++
LDLIBS = -lpng
# CFLAGS = -fno-signed-zeros -fno-trapping-math -fassociative-math
CFLAGS = -lm -O1
hw2a: CFLAGS += -mavx512f
hw2a: CFLAGS += -pthread
hw2a: CFLAGS += -ftree-vectorize
hw2a: CFLAGS += -march=native
# hw2a: CFLAGS += -Ofast
hw2b: CC = mpicc
hw2b: CXX = mpicxx
hw2b: CFLAGS += -fopenmp
hw2b: CFLAGS += -ftree-vectorize
hw2b: CFLAGS += -march=native
CXXFLAGS = $(CFLAGS)
TARGETS = hw2seq hw2a hw2b

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS) $(TARGETS:=.o)
