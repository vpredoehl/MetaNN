//
//  main.cpp
//  MetalSwiftTest
//
//  Created by Vincent Predoehl on 3/7/26.
//
#import <Metal/Metal.h>
#include <iostream>
#include <MetaNN/meta_nn.h>

using namespace MetaNN;

int main()
{
    const size_t rows = 2;
    const size_t cols = 3;

    Matrix<float, DeviceTags::Metal> src(rows, cols);
    Matrix<float, DeviceTags::Metal> dst(rows, cols);

    auto mem_src = LowerAccess(src);
    auto mem_dst = LowerAccess(dst);

    float* src_ptr = mem_src.MutableRawMemory();
    for (size_t i = 0; i < rows * cols; ++i)
    {
        src_ptr[i] = static_cast<float>(i + 1);
    }

    DataCopy(src, dst);

    const float* dst_ptr = mem_dst.RawMemory();

    std::cout << "Destination matrix:\n";
    for (size_t i = 0; i < rows * cols; ++i)
    {
        std::cout << dst_ptr[i] << " ";
        if ((i + 1) % cols == 0)
        {
            std::cout << '\n';
        }
    }

    return 0;
}
