//
//  main.cpp
//  MetalSwiftTest
//
//  Created by Vincent Predoehl on 3/7/26.
//
#include <iostream>
#include <MetaNN/meta_nn.h>
#include <MetaNN/metal/metal_add.h>

using namespace MetaNN;

int main()
{
    Matrix<float, DeviceTags::CPU> a_cpu(2, 3);
    Matrix<float, DeviceTags::CPU> b_cpu(2, 3);
    Matrix<float, DeviceTags::Metal> a_gpu(2, 3);
    Matrix<float, DeviceTags::Metal> b_gpu(2, 3);
    Matrix<float, DeviceTags::Metal> c_gpu(2, 3);
    Matrix<float, DeviceTags::CPU> c_cpu(2, 3);

    auto a_mem = LowerAccess(a_cpu);
    auto b_mem = LowerAccess(b_cpu);

    for (size_t i = 0; i < 6; ++i)
    {
        a_mem.MutableRawMemory()[i] = float(i + 1);        // 1 2 3 4 5 6
        b_mem.MutableRawMemory()[i] = float((i + 1) * 10); // 10 20 30 40 50 60
    }

    DataCopy(a_cpu, a_gpu);
    DataCopy(b_cpu, b_gpu);

    auto a_gpu_mem = LowerAccess(a_gpu);
    auto b_gpu_mem = LowerAccess(b_gpu);
    auto c_gpu_mem = LowerAccess(c_gpu);

    const auto a_gpu_shared = a_gpu_mem.SharedMemory();
    const auto b_gpu_shared = b_gpu_mem.SharedMemory();
    auto c_gpu_shared = c_gpu_mem.SharedMemory();

    NSMetalAdd::Add(a_gpu_shared,
                    b_gpu_shared,
                    c_gpu_shared,
                    6);

    DataCopy(c_gpu, c_cpu);

    auto c_mem = LowerAccess(c_cpu);
    for (size_t i = 0; i < 6; ++i)
    {
        std::cout << c_mem.RawMemory()[i] << " ";
    }
    std::cout << '\n';

    return 0;
}
