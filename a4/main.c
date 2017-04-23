// gcc -std=gnu99 -c -O3 -Wall a4.c
// nvcc p1_mult.cu -arch=sm_30 -dc
//nvcc -arch=sm_30 -dlink main.o p1_mult.o -o gpuCode.o
//g++ main.o p1_mult.o gpuCode.o -I/usr/local/cuda-8.0/lib64/ -lcudart -o a4

// gcc -std=gnu99 -c -O3 -Wall a4.c -o main.o
// nvcc p1_mult.cu -arch=sm_30 -dc
//nvcc -arch=sm_30 -dlink main.o p1_mult.o -o gpuCode.o
//g++ main.o p1_mult.o gpuCode.o -I/usr/local/cuda-8.0/lib64/ -lcudart -o a4

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <omp.h>
#include "matMul.h"


// Note N must be an even multiple of BLOCK_DIM
#define N 8192
#define BLOCK_DIM 16
#define RAND_SEED 97

#define CPU_NUM_THREADS 4 // For use in cpu matMul

// debugging/testing
#define ENABLE_CPU_MATMUL 1 //super slow, so only for small matrices
#define VALIDATE_IMPLIMENTATIONS 1 //tests all implimentations of matMul
#define PRINT_MATRIX_OUT 0




#if PRINT_MATRIX_OUT
    #define PRINT_MATRIX(...) print_matrix(__VA_ARGS__)
#else
    #define PRINT_MATRIX(...) do { } while(0)
#endif

// Prototypes
void print_matrix(uint64_t m, uint64_t n, float *data, bool matlab);
void fill_crap(uint64_t m, uint64_t n, float *data);
void validate_matMul();
bool floats_equal(float *a, float *b, const uint32_t num);

int main()
{

    float (*A)[N] = (float (*)[N]) malloc(sizeof(float[N][N])); // NxN
    float (*B)[N] = (float (*)[N]) malloc(sizeof(float[N][N])); // NxN
    float (*C)[N] = (float (*)[N]) malloc(sizeof(float[N][N])); // NxN

    //seed srand
    srand(RAND_SEED);

    //run tests
    #if VALIDATE_IMPLIMENTATIONS
    validate_matMul();
    #endif

    fill_crap(N, N, (float*)A);
    fill_crap(N, N, (float*)B);

    printf("A: \n");
    PRINT_MATRIX(N, N, (float*)A, false);

    printf("\n\nB: \n");
    PRINT_MATRIX(N, N, (float*)B, false);
    printf("\n\n");

    printf("Testing: 'matMul_naive()'\n");
    matMul_naive(N, (float*)A, (float*)B, (float*)C, BLOCK_DIM);
    printf("Result:\n");
    PRINT_MATRIX(N, N, (float*)C, false);

    printf("\nTesting: 'matMul_tiled()'\n");
    matMul_tiled(N, (float*)A, (float*)B, (float*)C, BLOCK_DIM);
    printf("Result:\n");
    PRINT_MATRIX(N, N, (float*)C, false);

    printf("\nTesting: 'matMul_tiled_transposed()'\n");
    matMul_tiled_transposed(N, (float*)A, (float*)B, (float*)C, BLOCK_DIM);
    printf("Result:\n");
    PRINT_MATRIX(N, N, (float*)C, false);


    printf("\nTesting: 'matMul_cublas()'\n");
    matMul_cublas(N, (float*)A, (float*)B, (float*)C);
    printf("Result:\n");
    PRINT_MATRIX(N, N, (float*)C, false);


    #if ENABLE_CPU_MATMUL
    printf("\nTesting: 'matMul_cpu()'\n");
    matMul_cpu(N, (float*)A, (float*)B, (float*)C, CPU_NUM_THREADS);
    printf("Result:\n");
    PRINT_MATRIX(N, N, (float*)C, false);
    #endif


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

// Checks all the implimentations aginst CPU implimentation
void validate_matMul()
{
    const uint16_t bd = 16;
    const uint16_t n = 2*bd;
    bool ret;

    float (*A)[n] = (float (*)[n]) malloc(sizeof(float[n][n])); // NxN
    float (*B)[n] = (float (*)[n]) malloc(sizeof(float[n][n])); // NxN
    float (*C)[n] = (float (*)[n]) malloc(sizeof(float[n][n])); // NxN
    float (*C_test)[n] = (float (*)[n]) malloc(sizeof(float[n][n])); // NxN

    printf("Testing matMul implimentations...\n");

    fill_crap(n, n, (float*)A);
    fill_crap(n, n, (float*)B);

    // have the cpu compute answer
    matMul_cpu(n, (float*)A, (float*)B, (float*)C, CPU_NUM_THREADS);

    // Test matMul_naive
    matMul_naive(n, (float*)A, (float*)B, (float*)C_test, bd);
    ret = floats_equal((float*)C_test, (float*)C, n*n);
    printf("'matMul_naive()': %s\n", ret ? "Pass" : "Fail");

    // Test matMul_tiled
    matMul_tiled(n, (float*)A, (float*)B, (float*)C_test, bd);
    ret = floats_equal((float*)C_test, (float*)C, n*n);
    printf("'matMul_tiled()': %s\n", ret ? "Pass" : "Fail");

    // Test matMul_tiled_transposed
    matMul_tiled_transposed(n, (float*)A, (float*)B, (float*)C_test, bd);
    ret = floats_equal((float*)C_test, (float*)C, n*n);
    printf("'matMul_tiled_transposed()': %s\n", ret ? "Pass" : "Fail");

    // Test matMul_cublas
    matMul_cublas(n, (float*)A, (float*)B, (float*)C_test);
    ret = floats_equal((float*)C_test, (float*)C, n*n);
    printf("'matMul_cublas()': %s\n", ret ? "Pass" : "Fail");

    free(A);
    free(B);
    free(C);
    free(C_test);
}


bool floats_equal(float *a, float *b, const uint32_t num)
{
    float epsilon = sqrt(num)*FLT_EPSILON;

    for (uint32_t i=0; i < num; i++)
    {
        if (fabs(a[i] - b[i]) > epsilon)
            return false;
    }

    return true;
}
