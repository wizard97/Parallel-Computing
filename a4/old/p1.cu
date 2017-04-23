
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

// Note N must be an even multiple of BLOCK_DIM
#define N (BLOCK_DIM*1024)
#define BLOCK_DIM 16
#define RAND_SEED 97

#define NB_OF_THREADS 4
#define PRINT_MATRIX_OUT 0

#if PRINT_MATRIX_OUT
    #define PRINT_MATRIX(...) print_matrix(__VA_ARGS__)
#else
    #define PRINT_MATRIX(...) do { } while(0)
#endif

// CUDA Functions
__global__ void matrixMul(uint32_t n, float *dev_A, float *dev_B, float *dev_C);

// HOST functions
void matrixMulCPU(uint32_t n, float *A, float *B, float *C);
void print_matrix(uint64_t m, uint64_t n, float *data, bool matlab);
void fill_crap(uint64_t m, uint64_t n, float *data);

int main()
{
    float *dev_A, *dev_B, *dev_C;

    float (*A)[N] = (float (*)[N]) malloc(sizeof(float[N][N])); // NxN
    float (*B)[N] = (float (*)[N]) malloc(sizeof(float[N][N])); // NxN
    float (*C)[N] = (float (*)[N]) malloc(sizeof(float[N][N])); // NxN

    //seed srand
    srand(RAND_SEED);

    fill_crap(N, N, (float*)A);
    fill_crap(N, N, (float*)B);

    printf("A: \n");
    PRINT_MATRIX(N, N, (float*)A, false);

    printf("\n\nB: \n");
    PRINT_MATRIX(N, N, (float*)B, false);

    // copy A
    cudaMalloc(&dev_A, sizeof(float[N][N]));
    cudaMemcpy(dev_A, A, sizeof(float[N][N]), cudaMemcpyHostToDevice);

    // Copy B
    cudaMalloc(&dev_B, sizeof(float[N][N]));
    cudaMemcpy(dev_B, (float*)B, sizeof(float[N][N]), cudaMemcpyHostToDevice);

    // Allocate space for C
    cudaMalloc(&dev_C, sizeof(float[N][N]));

    dim3 Block(BLOCK_DIM, BLOCK_DIM);
    dim3 Grid(N/Block.x, N/Block.y);
    matrixMul<<< Grid, Block>>>(N, dev_A, dev_B, dev_C);
    //matrixMulCPU(N, (float*)A, (float*)B, (float*)C);
    cudaMemcpy(C, dev_C, sizeof(float[N][N]), cudaMemcpyDeviceToHost);

    printf("\nResult:\n");
    PRINT_MATRIX(N, N, (float*)C, false);


    // Clean up memory
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);

    free(A);
    free(B);
    free(C);

    return 0;
}

__global__ void matrixMul(uint32_t n, float *dev_A, float *dev_B, float *dev_C)
{
    float partial = 0.0;
    uint32_t i = blockIdx.y * blockDim.y + threadIdx.y; // Row i of C
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x; // Column j of C

    for (uint32_t k = 0; k < n; k++)
        partial += dev_A[n*i + k]*dev_B[n*k + j];

    dev_C[n*i + j] = partial;
}


// compare to cpu implimentation
void matrixMulCPU(uint32_t n, float *A, float *B, float *C)
{
    #pragma omp parallel for schedule(static) num_threads(NB_OF_CPU_THREADS)
    for (uint32_t i=0; i < n; i++)
    {
        for (uint32_t j=0; j < n; j++)
        {
            C[n*i + j]=0.;
            for (uint32_t k=0; k<n; k++)
            {
                C[n*i + j] += B[n*i + k]*C[n*k + j];
            }
        }
    }
}



void print_matrix(uint64_t m, uint64_t n, float *data, bool matlab)
{
  if (matlab)
    printf("[ ");
  for (uint64_t i=0; i< m; i++) {

    for (uint64_t j=0; j < n-1; j++) {
      printf("%.4f, ", data[n*i+j]);
    }
    printf(matlab ? "%.4f; " : "%.4f\n", data[n*i + n-1]);
  }

  if (matlab)
    printf("]\n");
}




void fill_crap(uint64_t m, uint64_t n, float *data)
{
  for (uint64_t i=0; i<m; i++) {

    for (uint64_t j=0; j < n; j++) {
      data[n*i+j] = ((float)rand())/((float)RAND_MAX);
    }

  }

}
