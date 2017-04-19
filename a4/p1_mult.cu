#include "matrix_multiply.h"

__global__ void matrixMulCuda(uint32_t n, float *dev_A, float *dev_B, float *dev_C)
{
    float partial = 0.0;
    uint32_t i = blockIdx.y * blockDim.y + threadIdx.y; // Row i of C
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x; // Column j of C

    for (uint32_t k = 0; k < n; k++)
        partial += dev_A[n*i + k]*dev_B[n*k + j];

    dev_C[n*i + j] = partial;
}


void matrixMul(uint32_t n, float *dev_A, float *dev_B, float *dev_C, uint32_t block_dim)
{
    dim3 Block(block_dim, block_dim);
    dim3 Grid(n/Block.x, n/Block.y);
    matrixMulCuda<<< Grid, Block>>>(N, dev_A, dev_B, dev_C);
}
