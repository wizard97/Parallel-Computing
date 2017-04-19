#ifndef MATRIX_MULTIPLY_H
#define MATRIX_MULTIPLY_H

#include <stdlib.h>
#include <stdint.h>

void matrixMul(uint32_t n, float *dev_A, float *dev_B, float *dev_C, uint32_t block_dim);

#endif
