
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include "matrix_multiply.h"

// Note N must be an even multiple of BLOCK_DIM
#define N (BLOCK_DIM*128)
#define BLOCK_DIM 16
#define RAND_SEED 97

#define PRINT_MATRIX_OUT 0

#if PRINT_MATRIX_OUT
    #define PRINT_MATRIX(...) print_matrix(__VA_ARGS__)
#else
    #define PRINT_MATRIX(...) do { } while(0)
#endif

// HOST functions
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

    //dim3 Block(BLOCK_DIM, BLOCK_DIM);
    //dim3 Grid(N/Block.x, N/Block.y);
    matrixMul(N, dev_A, dev_B, dev_C, block_dim);
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
