#import "CppGpuMatMulAdd.h"
#import <limits.h>
#include <stdint.h>

// Declare the Swift thunk symbol
extern "C" int32_t swift_gpu_matmul_add(const float* A, int32_t rowsA, int32_t colsA,
                                        const float* B, int32_t rowsB, int32_t colsB,
                                        const float* D,
                                        float* C);

int cpp_gpu_matmul_add(const float* A, size_t rowsA, size_t colsA,
                       const float* B, size_t rowsB, size_t colsB,
                       const float* D,
                       float* C) {
    if (!A || !B || !C || !D) return 9;
    if (colsA != rowsB) return 4;
    if (rowsA > INT32_MAX || colsA > INT32_MAX || rowsB > INT32_MAX || colsB > INT32_MAX) return 3;

    int32_t status = swift_gpu_matmul_add(A, (int32_t)rowsA, (int32_t)colsA,
                                          B, (int32_t)rowsB, (int32_t)colsB,
                                          D,
                                          C);
    return (int)status;
}

