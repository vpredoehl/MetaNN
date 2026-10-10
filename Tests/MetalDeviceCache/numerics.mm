#import <Foundation/Foundation.h>
#include <MetaNN/data/tensor.h>
#include <MetaNN/metal/metal_matmul.h>
#include <iostream>

using namespace MetaNN;

template<class T>
void CheckTensor()
{
    Tensor<T, DeviceTags::Metal, 2> matrix(3, 5);
    for (size_t r = 0; r < 3; ++r)
        for (size_t c = 0; c < 5; ++c) matrix.SetValue(r, c, T(r * 5 + c) + T(0.125));
    auto row = matrix[2];
    for (size_t c = 0; c < 5; ++c) assert(row(c) == T(10 + c) + T(0.125));
    Tensor<T, DeviceTags::Metal, 0> scalar(T(2.25));
    assert(scalar.Value() == T(2.25));
}

int main()
{
    @autoreleasepool
    {
        CheckTensor<float>();
        CheckTensor<double>();
        ContinuousMemory<float, DeviceTags::Metal> baseA(8), baseB(8), baseC(8);
        auto a = baseA.Shift(4), b = baseB.Shift(4), c = baseC.Shift(4);
        for (size_t i = 0; i < 4; ++i)
        {
            a.MutableRawMemory()[i] = float(i + 1);
            b.MutableRawMemory()[i] = float(i + 5);
            baseC.MutableRawMemory()[i] = -123.f;
        }
        NSMetalMatMul::MatMul(a, b, c, 2, 2, 2);
        const float expected[] = {19, 22, 43, 50};
        for (size_t i = 0; i < 4; ++i)
        {
            assert(c.RawMemory()[i] == expected[i]);
            assert(baseC.RawMemory()[i] == -123.f);
        }
        std::cout << "PASS float/double tensor values, slices, and offset-backed GPU matmul: 19,22,43,50\n";
    }
}
