#include "matMul.h"
#include <omp.h>

void matMul_cpu(const uint32_t n, const float *A, const float *B,
        float *C, const uint16_t nthreads)
{
    #pragma omp parallel for schedule(static) num_threads(nthreads)
    for (uint32_t i=0; i < n; i++)
    {
        for (uint32_t j=0; j < n; j++)
        {
            C[n*i + j]=0.;
            for (uint32_t k=0; k<n; k++)
            {
                C[n*i + j] += A[n*i + k]*B[n*k + j];
            }
        }
    }

}
