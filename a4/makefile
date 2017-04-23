objects = main.o naive_mult.o cpu_mult.o matMul_tiled.o matMul_tiled_transposed.o  matMul_cublas.o
exec = a4

ALL_INCLUDES = -I/usr/local/cuda-8.0/lib64/
ALL_LIBS = -lcudart -fopenmp -lcublas

C_INCLUDES = -I/usr/local/cuda-8.0/lib64/
C_LIBS = -fopenmp -lcublas -lcudart
# gcc -std=gnu99 -c -O3 -Wall a4.c -o main.o
# nvcc p1_mult.cu -arch=sm_30 -dc
# nvcc -arch=sm_30 -dlink main.o p1_mult.o -o gpuCode.o
# g++ main.o p1_mult.o gpuCode.o -I/usr/local/cuda-8.0/lib64/ -lcudart -o a4

all: $(objects) gpuCode.o
	g++ $(objects) gpuCode.o $(ALL_INCLUDES) $(ALL_LIBS) -o $(exec)

%.o: %.c
	gcc -std=gnu99 -c -O3 -Wall $(C_INCLUDES) $(C_LIBS) $< -o $@

%.o: %.cu
	nvcc -arch=sm_30 -O3 -dc -D_FORCE_INLINES $< -o $@

gpuCode.o : $(objects)
	nvcc -arch=sm_30 -O3 -dlink $^ -o $@

clean:
	rm -f *.o $(exec)
