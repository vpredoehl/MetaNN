#include "CppGpuMatMulAdd.h"
#include <vector>
#include <cstdio>
#include <iostream>

extern "C" void cpp_run_matmul_add_demo();

void runCppMatMulAddDemo() {
    const size_t rowsA = 2, colsA = 3;
    const size_t rowsB = 3, colsB = 2;
    const size_t rowsC = rowsA, colsC = colsB;

    std::vector<float> A = {
        1, 2, 3,
        4, 5, 6
    };
    std::vector<float> B = {
        7, 8,
        9, 10,
        11, 12
    };
    std::vector<float> D = {
        0.5f, 1.5f,
        2.5f, 3.5f
    };

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
        std::printf("C[%zu]: [", r);
        for (size_t c = 0; c < colsC; ++c) {
            std::printf("%g%s", C[r * colsC + c], (c + 1 < colsC) ? ", " : "");
        }
        std::puts("]");
    }
}

// Provide a C ABI entry point that Swift can call
extern "C" void cpp_run_matmul_add_demo() {
    runCppMatMulAddDemo();
}
