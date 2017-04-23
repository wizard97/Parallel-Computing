#include "matMul.h"
//#include <cublas.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define IDX2C(i,j,ld) (((j)*(ld))+(i))


void matMul_cublas(const uint32_t n, const float *A, const float *B, float *C)
{
    cublasHandle_t handle;

    float *dev_A, *dev_B, *dev_C;

    float *a = (float *)malloc(sizeof(float[n][n]));
    float *b = (float *)malloc(sizeof(float[n][n]));
    float *c = (float *)malloc(sizeof(float[n][n]));

    for (uint32_t j = 0; j < n; j++) {
        for (uint32_t i = 0; i < n; i++) {
            a[IDX2C(i,j,n)] = A[i*n + j];
            b[IDX2C(i,j,n)] = B[i*n + j];
        }
    }

    //cublasInit();

    cudaMalloc((void**)&dev_A, sizeof(float[n][n]));
    cudaMalloc((void**)&dev_B, sizeof(float[n][n]));
    cudaMalloc((void**)&dev_C, sizeof(float[n][n]));

    cublasCreate(&handle);

    cublasSetMatrix (n, n, sizeof(float), a, n, dev_A, n);
    cublasSetMatrix (n, n, sizeof(float), b, n, dev_B, n);

    float alpha = 1;
    float beta = 0;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, dev_A, n, dev_B, n, &beta, dev_C, n);

    cublasGetMatrix(n, n, sizeof(float), dev_C, n, c, n);


    // copy back answer into C
    for (uint32_t j = 0; j < n; j++) {
        for (uint32_t i = 0; i < n; i++) {
            C[n*i + j] = c[IDX2C(i,j,n)];
        }
    }

    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);

    cublasDestroy(handle);

    free(a);
    free(b);
    free(c);

    //cublasShutdown();

}
