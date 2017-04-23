#include "matMul.h"

__global__ static void matrixMulCuda(const uint32_t n, const float *dev_A, const float *dev_B,
            float *dev_C)
{
    extern __shared__ int s[]; //shared memory allocated during kernel launch

    // create A_tile and B_tile from shared memory s
    float *A_tile = (float *)s;
    float *B_tile = A_tile + blockDim.x*blockDim.y;

    float partial = 0.0;

    // blcok indxes
    uint16_t bx = blockIdx.x; uint16_t by = blockIdx.y;
    uint16_t bdx = blockDim.x; uint16_t bdy = blockDim.y;
    uint16_t tx = threadIdx.x; uint16_t ty = threadIdx.y;

    // tile
    uint16_t numTiles = n/bdx; // should also equal n/blockDim.y

    // which col and row each thread has of C
    uint16_t row = by*bdy + ty;
    uint16_t col = bx*bdx + tx;


    for (uint16_t m = 0; m < numTiles; m++) {
        // Loads are coallesced
        //A_tile[ty][tx] = dev_A[row][m*bdx + tx];
        A_tile[bdx*ty + tx] = dev_A[n*row + m*bdx + tx];


        //B_tile[ty][tx] = dev_B[m*bdy + ty][col];
        B_tile[bdy*ty + tx] = dev_B[n*(m*bdy + ty) + col];

        // wait for all threads to finish
        __syncthreads();

        // compute partial dot product
        for (uint16_t x = 0; x < bdx; x++)
            partial += A_tile[bdx*ty + x] * B_tile[bdy*x + tx];

        __syncthreads();
    }

    // update global memory
    dev_C[n*row + col] = partial;
}


void matMul_tiled(const uint32_t n, const float *A, const float *B, float *C, const uint32_t block_dim)
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

    matrixMulCuda<<< Grid, Block, 2*sizeof(float[Block.y][Block.x])>>>(n, dev_A, dev_B, dev_C);

    cudaMemcpy(C, dev_C, sizeof(float[n][n]), cudaMemcpyDeviceToHost);

    // Clean up memory
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);

}
