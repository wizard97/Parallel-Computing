#include "matMul.h"

__global__ static void matrixMulCuda(const uint32_t n, const  float *dev_A, const float *dev_B,
            float *dev_C)
{
    float partial = 0.0;
    uint32_t i = blockIdx.y * blockDim.y + threadIdx.y; // Row i of C
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x; // Column j of C

    for (uint32_t k = 0; k < n; k++)
        partial += dev_A[n*i + k]*dev_B[n*k + j];

    dev_C[n*i + j] = partial;
}


void matMul_naive(const uint32_t n, const float *A, const float *B, float *C,
            const uint32_t block_dim)
{
    float *dev_A, *dev_B, *dev_C;
    // copy A
    cudaMalloc(&dev_A, sizeof(float[n][n]));
    cudaMemcpy(dev_A, A, sizeof(float[n][n]), cudaMemcpyHostToDevice);

    // Copy B
    cudaMalloc(&dev_B, sizeof(float[n][n]));
    cudaMemcpy(dev_B, (float*)B, sizeof(float[n][n]), cudaMemcpyHostToDevice);

    // Allocate space for C
    cudaMalloc(&dev_C, sizeof(float[n][n]));

    dim3 Block(block_dim, block_dim);
    dim3 Grid(n/Block.x, n/Block.y);

    matrixMulCuda<<< Grid, Block>>>(n, dev_A, dev_B, dev_C);

    cudaMemcpy(C, dev_C, sizeof(float[n][n]), cudaMemcpyDeviceToHost);

    // Clean up memory
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);

}
