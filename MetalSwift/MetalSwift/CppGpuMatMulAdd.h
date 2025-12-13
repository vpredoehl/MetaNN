#ifndef CPP_GPU_MATMUL_ADD_H
#define CPP_GPU_MATMUL_ADD_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Computes C = A(rowsA x colsA) * B(rowsB x colsB) + D(rowsA x colsB)
// Arrays are row-major.
// Returns 0 on success.
int cpp_gpu_matmul_add(const float* A, size_t rowsA, size_t colsA,
                       const float* B, size_t rowsB, size_t colsB,
                       const float* D,
                       float* C);

#ifdef __cplusplus
}
#endif

#endif // CPP_GPU_MATMUL_ADD_H
