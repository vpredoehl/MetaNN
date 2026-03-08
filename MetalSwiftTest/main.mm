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

#include <cassert>
#include <cmath>
#include <iostream>

#include "MetaNN/meta_nn.h"

using namespace MetaNN;

namespace
{
    template <typename T>
    bool NearlyEqual(T a, T b, T eps = static_cast<T>(1e-5))
    {
        return std::fabs(a - b) < eps;
    }

    void Fill2x3(Matrix<float, DeviceTags::Metal>& m,
                 float v00, float v01, float v02,
                 float v10, float v11, float v12)
    {
        auto mem = LowerAccess(m);
        auto* p = mem.MutableRawMemory();
        p[0] = v00; p[1] = v01; p[2] = v02;
        p[3] = v10; p[4] = v11; p[5] = v12;
    }

    void Check2x3(const Matrix<float, DeviceTags::Metal>& m,
                  float v00, float v01, float v02,
                  float v10, float v11, float v12)
    {
        auto mem = LowerAccess(m);
        const auto* p = mem.RawMemory();
        assert(NearlyEqual(p[0], v00));
        assert(NearlyEqual(p[1], v01));
        assert(NearlyEqual(p[2], v02));
        assert(NearlyEqual(p[3], v10));
        assert(NearlyEqual(p[4], v11));
        assert(NearlyEqual(p[5], v12));
    }

    void TestAddLazyEvaluationMetal()
    {
        std::cout << "Test add lazy evaluation (Metal)\t";

        Matrix<float, DeviceTags::Metal> a(2, 3);
        Matrix<float, DeviceTags::Metal> b(2, 3);

        // Initial values
        Fill2x3(a,
                1, 2, 3,
                4, 5, 6);

        Fill2x3(b,
                10, 20, 30,
                40, 50, 60);

        // Build expression only. This should be lazy.
        auto op = a + b;

        static_assert(IsMatrix<decltype(op)>);

        // If shape metadata is available before evaluation, this should work.
        assert(op.Shape()[0] == 2);
        assert(op.Shape()[1] == 3);

        // Mutate inputs AFTER expression creation.
        // If add is lazy, Evaluate(op) should see THESE values, not the old ones.
        Fill2x3(a,
                100, 200, 300,
                400, 500, 600);

        Fill2x3(b,
                1, 2, 3,
                4, 5, 6);

        auto res = Evaluate(op);

        static_assert(IsMatrix<decltype(res)>);
        assert(res.Shape()[0] == 2);
        assert(res.Shape()[1] == 3);

        Check2x3(res,
                 101, 202, 303,
                 404, 505, 606);

        std::cout << "done" << std::endl;
    }
}

int main()
{
  TestAddLazyEvaluationMetal();

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
        b_mem.MutableRawMemory()[i] = float((i + 1) * 20); // 10 20 30 40 50 60
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
