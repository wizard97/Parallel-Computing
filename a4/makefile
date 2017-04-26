OBJECTS = daw268_hw4_1_to_4.o naive_mult.o cpu_mult.o matMul_tiled.o matMul_tiled_transposed.o  matMul_cublas.o
EXEC = daw268_hw4_1_to_4

ALL_INCLUDES = -L/usr/local/cuda-8.0/lib64/
ALL_LIBS = -lcudart -fopenmp -lcublas

C_INCLUDES = -I/usr/local/cuda-8.0/include/
C_LIBS = -fopenmp -lcublas -lcudart

COMPUTE_VER ?= sm_52
COMPUTE = -arch=$(COMPUTE_VER)

NVCC= /usr/local/cuda-8.0/bin/nvcc

#COMPUTE = -arch=sm_30
# gcc -std=gnu99 -c -O3 -Wall a4.c -o main.o
# nvcc p1_mult.cu -arch=sm_30 -dc
# nvcc -arch=sm_30 -dlink main.o p1_mult.o -o gpuCode.o
# g++ main.o p1_mult.o gpuCode.o -I/usr/local/cuda-8.0/lib64/ -lcudart -o a4

all: $(OBJECTS) gpuCode.o
	g++ $(OBJECTS) gpuCode.o $(ALL_INCLUDES) $(ALL_LIBS) -o $(EXEC)

%.o: %.c
	gcc -std=gnu99 -c -O3 -Wall $(C_INCLUDES) $(C_LIBS) $< -o $@

%.o: %.cu
	$(NVCC) $(COMPUTE) -O3 -dc -D_FORCE_INLINES $< -o $@

gpuCode.o : $(OBJECTS)
	$(NVCC) $(COMPUTE) -O3 -dlink $^ -o $@

clean:
	rm -f *.o $(EXEC)
