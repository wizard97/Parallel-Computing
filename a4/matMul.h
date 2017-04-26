#ifndef MATRIX_MULTIPLY_H
#define MATRIX_MULTIPLY_H

#include <stdlib.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
    // CPU implimentations
    void matMul_cpu(const uint32_t n, const float *A, const float *B,
                float *C, const uint16_t nthreads);

    // GPU implimentations
    void matMul_naive(const uint32_t n, const float *A, const float *B,
                float *C, uint32_t block_dim);

    void matMul_tiled(const uint32_t n, const float *A, const float *B,
                float *C, uint32_t block_dim);

    void matMul_tiled_transposed(const uint32_t n, const float *A, const float *B,
                float *C, uint32_t block_dim);

    void matMul_cublas(const uint32_t n, const float *A,
                const float *B, float *C);



#ifdef __cplusplus
}
#endif



#endif
