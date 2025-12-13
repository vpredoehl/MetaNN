#include <vector>
#include <iostream>
#include <cstdio>
#include "CppGpuBridge.h"
#include "CppGpuMatMulAdd.h"

static void runVectorAddDemo() {
    std::vector<float> a = {1, 2, 3, 4, 5};
    std::vector<float> b = {10, 20, 30, 40, 50};
    std::vector<float> out(a.size(), 0.0f);

    int status = cpp_gpu_vector_add(a.data(), b.data(), out.data(), a.size());
    if (status != 0) {
        std::cerr << "cpp_gpu_vector_add failed with status: " << status << std::endl;
        return;
    }

    std::printf("VectorAdd: [");
    for (size_t i = 0; i < out.size(); ++i) {
        std::printf("%g%s", out[i], (i + 1 < out.size()) ? ", " : "]\n");
    }
}

static void runMatMulAddDemo() {
    const size_t rowsA = 2, colsA = 3;
    const size_t rowsB = 3, colsB = 2;
    const size_t rowsC = rowsA, colsC = colsB;

    std::vector<float> A = { 1, 2, 3, 4, 5, 6 };
    std::vector<float> B = { 7, 8, 9, 10, 11, 12 };
    std::vector<float> D = { 0.5f, 1.5f, 2.5f, 3.5f };
    std::vector<float> C(rowsC * colsC, 0.0f);

    int status = cpp_gpu_matmul_add(A.data(), rowsA, colsA,
                                    B.data(), rowsB, colsB,
                                    D.data(),
                                    C.data());
    if (status != 0) {
        std::cerr << "cpp_gpu_matmul_add failed with status: " << status << std::endl;
        return;
    }

    for (size_t r = 0; r < rowsC; ++r) {
        std::printf("MatMul+Add C[%zu]: [", r);
        for (size_t c = 0; c < colsC; ++c) {
            std::printf("%g%s", C[r * colsC + c], (c + 1 < colsC) ? ", " : "]\n");
        }
    }
}

extern "C" void cpp_run_gpu_demos() {
    runVectorAddDemo();
    runMatMulAddDemo();
}
