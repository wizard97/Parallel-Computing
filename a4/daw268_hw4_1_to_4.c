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
#include <time.h>
#include <unistd.h>
#include <math.h>
#include <float.h>
#include <omp.h>
#include "matMul.h"
#define BILLION 1000000000

// Size ranges os tested matrices
// Note N must be an even multiple of BLOCK_DIM
// test between N_START x N_START to N_ENDx N_END
#define N_START BLOCK_DIM
#define N_END 256*BLOCK_DIM

#define BLOCK_DIM 16 //cuda block dim BLOCK_DIM x BLOCK_DIMBLOCK_DIM
#define NUM_RUNS 3 // number of averages for each matrix size
#define RAND_SEED 97 //seed srand()

#define ENABLE_CPU_MATMUL 1 //super slow, so only for small matrices, uses OMP
#define CPU_NUM_THREADS 16 // For use in cpu matMul

// debugging/testing
#define VALIDATE_IMPLIMENTATIONS 1 //tests all implimentations of matMul before benchmarking
#define PRINT_MATRIX_OUT 0 //prints A, B, and C out




#if PRINT_MATRIX_OUT
    #define PRINT_MATRIX(A, ...) do { printf("Matrix "A ":\n"); \
                                print_matrix(__VA_ARGS__); \
                                printf("\n"); } while(0)
#else
    #define PRINT_MATRIX(...) do { } while(0)
#endif

// Prototypes
void print_matrix(uint64_t m, uint64_t n, float *data, bool matlab);
void fill_crap(uint64_t m, uint64_t n, float *data);
void validate_matMul();
bool floats_equal(float *a, float *b, const uint32_t num);
uint64_t get_dt(struct timespec *start, struct timespec *end);

int main(int argc, char **argv)
{
    struct timespec start, end;
    uint64_t rt;

    srand(RAND_SEED);

    char tmp[50];
    sprintf(tmp, "data/a4_data_%u_%u.csv", N_START, N_END);
    FILE *f = fopen(tmp, "w");

    fprintf(f, "N,Naive (ns),Tiled (ns),Transposed (ns),Cublas (ns),CPU %u Thds OMP (ns)\n", CPU_NUM_THREADS);

    //run tests
    #if VALIDATE_IMPLIMENTATIONS
    validate_matMul();
    #endif

    // sweep
    for (uint32_t N= N_START; N <= N_END; N <<= 1)
    {
        uint64_t rt_mn = 0, rt_mt = 0, rt_tt = 0, rt_cb = 0, rt_cp = 0;

        float (*A)[N] = (float (*)[N]) malloc(sizeof(float[N][N])); // NxN
        float (*B)[N] = (float (*)[N]) malloc(sizeof(float[N][N])); // NxN
        float (*C)[N] = (float (*)[N]) malloc(sizeof(float[N][N])); // NxN

        fill_crap(N, N, (float*)A);
        fill_crap(N, N, (float*)B);

        //number of averages
        for (uint32_t i=0; i < NUM_RUNS; i++)
        {
            printf("************** Begin test %ux%u, run %u of %u *****************\n",
                                N, N, i+1, NUM_RUNS);

            PRINT_MATRIX("A", N, N, (float*)A, false);

            PRINT_MATRIX("B", N, N, (float*)B, false);



            printf("Running: 'matMul_naive()'\n");
            clock_gettime(CLOCK_MONOTONIC, &start);
            matMul_naive(N, (float*)A, (float*)B, (float*)C, BLOCK_DIM);
            clock_gettime(CLOCK_MONOTONIC, &end);
            rt = get_dt(&start, &end);
            rt_mn += rt;
            printf("Completed in %lu us\n\n", rt/1000);
            PRINT_MATRIX("C", N, N, (float*)C, false);


            printf("Running: 'matMul_tiled()'\n");
            clock_gettime(CLOCK_MONOTONIC, &start);
            matMul_tiled(N, (float*)A, (float*)B, (float*)C, BLOCK_DIM);
            clock_gettime(CLOCK_MONOTONIC, &end);
            rt = get_dt(&start, &end);
            rt_mt += rt;
            printf("Completed in %lu us\n\n", rt/1000);
            PRINT_MATRIX("C", N, N, (float*)C, false);


            printf("Running: 'matMul_tiled_transposed()'\n");
            clock_gettime(CLOCK_MONOTONIC, &start);
            matMul_tiled_transposed(N, (float*)A, (float*)B, (float*)C, BLOCK_DIM);
            clock_gettime(CLOCK_MONOTONIC, &end);
            rt = get_dt(&start, &end);
            rt_tt += rt;
            printf("Completed in %lu us\n\n", rt/1000);
            PRINT_MATRIX("C", N, N, (float*)C, false);


            printf("Running: 'matMul_cublas()'\n");
            clock_gettime(CLOCK_MONOTONIC, &start);
            matMul_cublas(N, (float*)A, (float*)B, (float*)C);
            clock_gettime(CLOCK_MONOTONIC, &end);
            rt = get_dt(&start, &end);
            rt_cb += rt;
            printf("Completed in %lu us\n\n", rt/1000);
            PRINT_MATRIX("C", N, N, (float*)C, false);


            #if ENABLE_CPU_MATMUL
            printf("Testing: 'matMul_cpu()'\n");
            clock_gettime(CLOCK_MONOTONIC, &start);
            matMul_cpu(N, (float*)A, (float*)B, (float*)C, CPU_NUM_THREADS);
            clock_gettime(CLOCK_MONOTONIC, &end);
            rt = get_dt(&start, &end);
            rt_cp += rt;
            printf("Completed in %lu us\n\n", rt/1000);
            PRINT_MATRIX("C", N, N, (float*)C, false);
            #endif

        }

        // average data
        rt_mn/=NUM_RUNS; rt_mt/=NUM_RUNS; rt_tt/=NUM_RUNS; rt_cb/=NUM_RUNS; rt_cp/=NUM_RUNS;

        fprintf(f, "%u,%lu,%lu,%lu,%lu,%lu\n", N, rt_mn, rt_mt, rt_tt, rt_cb, rt_cp);

        free(A);
        free(B);
        free(C);
    }

    fclose(f);

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

    printf("\n");

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


uint64_t get_dt(struct timespec *start, struct timespec *end)
{
    return BILLION*(end->tv_sec - start->tv_sec) + (end->tv_nsec - start->tv_nsec);
}
