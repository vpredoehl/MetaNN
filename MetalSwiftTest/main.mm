//
//  main.cpp
//  MetalSwiftTest
//
//  Created by Vincent Predoehl on 3/7/26.
//
#include <iostream>
#include <cassert>
#include <cmath>
#include <MetaNN/meta_nn.h>
#include <MetaNN/metal/metal_add.h>

using namespace MetaNN;


using namespace MetaNN;

template <typename MatT>
void printMatrix(const char* name, const MatT& mat)
{
    std::cout << name << " (" << mat.Shape()[0] << " x " << mat.Shape()[1] << ")\n";
    for (size_t i = 0; i < mat.Shape()[0]; ++i)
    {
        for (size_t j = 0; j < mat.Shape()[1]; ++j)
        {
            std::cout << mat(i, j) << (j + 1 == mat.Shape()[1] ? '\n' : ' ');
        }
    }
    std::cout << std::endl;
}

namespace
{
    template <typename T>
    bool NearlyEqual(T a, T b, T eps = static_cast<T>(1e-5))
    {
        return std::fabs(a - b) < eps;
    }

    void Fill2x3(Matrix<float, DeviceTags::CPU>& m,
                 float v00, float v01, float v02,
                 float v10, float v11, float v12)
    {
        auto mem = LowerAccess(m);
        auto* p = mem.MutableRawMemory();

        p[0] = v00; p[1] = v01; p[2] = v02;
        p[3] = v10; p[4] = v11; p[5] = v12;
    }

    void Fill2x3(Matrix<float, DeviceTags::Metal>& m,
                 float v00, float v01, float v02,
                 float v10, float v11, float v12)
    {
        Matrix<float, DeviceTags::CPU> tmp(2,3);
        Fill2x3(tmp, v00,v01,v02,v10,v11,v12);
        DataCopy(tmp, m);
    }

    void Check2x3(const Matrix<float, DeviceTags::CPU>& m,
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
        Matrix<float, DeviceTags::CPU> a_cpu(2, 3);
        Matrix<float, DeviceTags::CPU> b_cpu(2, 3);

        // Initial values
        Fill2x3(a_cpu,
                1, 2, 3,
                4, 5, 6);

        Fill2x3(b_cpu,
                10, 20, 30,
                40, 50, 60);

        DataCopy(a_cpu, a);
        DataCopy(b_cpu, b);

        // Build expression only. This should be lazy.
        auto op = a + b;

        static_assert(IsMatrix<decltype(op)>);

        // If shape metadata is available before evaluation, this should work.
        assert(op.Shape()[0] == 2);
        assert(op.Shape()[1] == 3);

        // Mutate inputs AFTER expression creation.
        // If add is lazy, Evaluate(op) should see THESE values, not the old ones.
        Fill2x3(a_cpu,
                100, 200, 300,
                400, 500, 600);

        Fill2x3(b_cpu,
                1, 2, 3,
                4, 5, 6);

        DataCopy(a_cpu, a);
        DataCopy(b_cpu, b);

        auto res = Evaluate(op);
        Matrix<float, DeviceTags::CPU> res_cpu(2, 3);
        DataCopy(res, res_cpu);

        static_assert(IsMatrix<decltype(res)>);
        assert(res.Shape()[0] == 2);
        assert(res.Shape()[1] == 3);

        Check2x3(res_cpu,
                 101, 202, 303,
                 404, 505, 606);

        std::cout << "done" << std::endl;
    }

    void TestSubstractLazyEvaluationMetal()
    {
        std::cout << "Test substract lazy evaluation (Metal)\t";

        Matrix<float, DeviceTags::Metal> a(2, 3);
        Matrix<float, DeviceTags::Metal> b(2, 3);
        Matrix<float, DeviceTags::CPU> a_cpu(2, 3);
        Matrix<float, DeviceTags::CPU> b_cpu(2, 3);

        // Initial values
        Fill2x3(a_cpu,
                1, 2, 3,
                4, 5, 6);

        Fill2x3(b_cpu,
                10, 20, 30,
                40, 50, 60);

        DataCopy(a_cpu, a);
        DataCopy(b_cpu, b);

        // Build expression only. This should be lazy.
        auto op = a - b;

        static_assert(IsMatrix<decltype(op)>);

        // If shape metadata is available before evaluation, this should work.
        assert(op.Shape()[0] == 2);
        assert(op.Shape()[1] == 3);

        // Mutate inputs AFTER expression creation.
        // If substract is lazy, Evaluate(op) should see THESE values, not the old ones.
        Fill2x3(a_cpu,
                100, 200, 300,
                400, 500, 600);

        Fill2x3(b_cpu,
                1, 2, 3,
                4, 5, 6);

        DataCopy(a_cpu, a);
        DataCopy(b_cpu, b);

        auto res = Evaluate(op);
        Matrix<float, DeviceTags::CPU> res_cpu(2, 3);
        DataCopy(res, res_cpu);

        static_assert(IsMatrix<decltype(res)>);
        assert(res.Shape()[0] == 2);
        assert(res.Shape()[1] == 3);

        Check2x3(res_cpu,
                 99, 198, 297,
                 396, 495, 594);

        std::cout << "test metal subtract done" << std::endl;
    }
}

//
//  main.mm
//

using namespace MetaNN;

template <typename MatT>
void PrintMatrix(const char* name, const MatT& mat)
{
    std::cout << name << " (" << mat.Shape()[0] << " x " << mat.Shape()[1] << ")\n";
    for (size_t i = 0; i < mat.Shape()[0]; ++i)
    {
        for (size_t j = 0; j < mat.Shape()[1]; ++j)
        {
            std::cout << mat(i, j) << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

//////////////////////////////////////////////////////////////
// Expression-level tests
//////////////////////////////////////////////////////////////

void TestExprAddMetal()
{
    std::cout << "Expr Add (Metal)\t";

    Matrix<float, DeviceTags::Metal> a(2,3);
    Matrix<float, DeviceTags::Metal> b(2,3);

    Fill2x3(a, 1,2,3,4,5,6);
    Fill2x3(b,10,20,30,40,50,60);

    auto expr = a + b;
    auto r = Evaluate(expr);

    Matrix<float, DeviceTags::CPU> cpu(2,3);
    DataCopy(r, cpu);

    Check2x3(cpu,
             11,22,33,
             44,55,66);

    std::cout << "ok\n";
}

void TestExprSubMetal()
{
    std::cout << "Expr Sub (Metal)\t";

    Matrix<float, DeviceTags::Metal> a(2,3);
    Matrix<float, DeviceTags::Metal> b(2,3);

    Fill2x3(a, 100,200,300,400,500,600);
    Fill2x3(b, 1,2,3,4,5,6);

    auto expr = a - b;
    auto r = Evaluate(expr);

    Matrix<float, DeviceTags::CPU> cpu(2,3);
    DataCopy(r, cpu);

    Check2x3(cpu,
             99,198,297,
             396,495,594);

    std::cout << "ok\n";
}

void TestExprHadamardMetal()
{
    std::cout << "Expr Hadamard (Metal)\t";

    Matrix<float, DeviceTags::Metal> a(2,3);
    Matrix<float, DeviceTags::Metal> b(2,3);

    Fill2x3(a,1,2,3,4,5,6);
    Fill2x3(b,10,20,30,40,50,60);

    auto expr = a * b;
    auto r = Evaluate(expr);

    Matrix<float, DeviceTags::CPU> cpu(2,3);
    DataCopy(r, cpu);

    Check2x3(cpu,
             10,40,90,
             160,250,360);

    std::cout << "ok\n";
}

void TestExprUnaryMetal()
{
    std::cout << "Expr Unary Ops (Metal)\n";

    Matrix<float, DeviceTags::Metal> a(2,3);
    Matrix<float, DeviceTags::CPU> cpu(2,3);

    Fill2x3(a,-1,0,1,2,3,4);

    DataCopy(Evaluate(-a), cpu);
    PrintMatrix("neg", cpu);

    DataCopy(Evaluate(Tanh(a)), cpu);
    PrintMatrix("tanh", cpu);

    DataCopy(Evaluate(Sigmoid(a)), cpu);
    PrintMatrix("sigmoid", cpu);

    DataCopy(Evaluate(Exp(a)), cpu);
    PrintMatrix("exp", cpu);
}

void TestExprActivationMetal()
{
    std::cout << "Expr Activations (Metal)\n";

    Matrix<float, DeviceTags::Metal> a(2,3);
    Matrix<float, DeviceTags::CPU> cpu(2,3);

    Fill2x3(a,0,1,2,3,4,5);

    DataCopy(Evaluate(ReLU(a)), cpu);
    PrintMatrix("relu", cpu);

    DataCopy(Evaluate(Softmax(a)), cpu);
    PrintMatrix("softmax", cpu);
}

//////////////////////////////////////////////////////////////
// Backend-level tests
//////////////////////////////////////////////////////////////

void TestBackendAddMetal()
{
    std::cout << "Backend Add (Metal)\t";

    Matrix<float, DeviceTags::CPU> a_cpu(2,3);
    Matrix<float, DeviceTags::CPU> b_cpu(2,3);
    Matrix<float, DeviceTags::CPU> c_cpu(2,3);

    Matrix<float, DeviceTags::Metal> a_gpu(2,3);
    Matrix<float, DeviceTags::Metal> b_gpu(2,3);
    Matrix<float, DeviceTags::Metal> c_gpu(2,3);

    Fill2x3(a_cpu,1,2,3,4,5,6);
    Fill2x3(b_cpu,10,20,30,40,50,60);

    DataCopy(a_cpu,a_gpu);
    DataCopy(b_cpu,b_gpu);

    auto a_mem = LowerAccess(a_gpu);
    auto b_mem = LowerAccess(b_gpu);
    auto c_mem = LowerAccess(c_gpu);

    // Capture SharedMemory() into lvalues to avoid binding temporaries to non-const lvalue references
    auto a_shmem = a_mem.SharedMemory();
    auto b_shmem = b_mem.SharedMemory();
    auto c_shmem = c_mem.SharedMemory();

    NSMetalAdd::Add(
        a_shmem,
        b_shmem,
        c_shmem,
        6);

    DataCopy(c_gpu, c_cpu);

    Check2x3(c_cpu,
             11,22,33,
             44,55,66);

    std::cout << "ok\n";
}

void TestBackendSubMetal()
{
    std::cout << "Backend Sub (Metal)\t";

    Matrix<float, DeviceTags::CPU> a_cpu(2,3);
    Matrix<float, DeviceTags::CPU> b_cpu(2,3);
    Matrix<float, DeviceTags::CPU> c_cpu(2,3);

    Matrix<float, DeviceTags::Metal> a_gpu(2,3);
    Matrix<float, DeviceTags::Metal> b_gpu(2,3);
    Matrix<float, DeviceTags::Metal> c_gpu(2,3);

    Fill2x3(a_cpu,100,200,300,400,500,600);
    Fill2x3(b_cpu,1,2,3,4,5,6);

    DataCopy(a_cpu, a_gpu);
    DataCopy(b_cpu, b_gpu);

    auto a_mem = LowerAccess(a_gpu);
    auto b_mem = LowerAccess(b_gpu);
    auto c_mem = LowerAccess(c_gpu);

    auto a_shmem = a_mem.SharedMemory();
    auto b_shmem = b_mem.SharedMemory();
    auto c_shmem = c_mem.SharedMemory();

    NSMetalAdd::Sub(
        a_shmem,
        b_shmem,
        c_shmem,
        6);

    DataCopy(c_gpu, c_cpu);

    Check2x3(c_cpu,
             99,198,297,
             396,495,594);

    std::cout << "ok\n";
}

void TestBackendNegMetal()
{
    std::cout << "Backend Neg (Metal)\t";

    Matrix<float, DeviceTags::CPU> a_cpu(2,3);
    Matrix<float, DeviceTags::CPU> c_cpu(2,3);

    Matrix<float, DeviceTags::Metal> a_gpu(2,3);
    Matrix<float, DeviceTags::Metal> c_gpu(2,3);

    Fill2x3(a_cpu,-1,0,1,2,3,4);

    DataCopy(a_cpu, a_gpu);

    auto a_mem = LowerAccess(a_gpu);
    auto c_mem = LowerAccess(c_gpu);

    auto a_shmem = a_mem.SharedMemory();
    auto c_shmem = c_mem.SharedMemory();

    NSMetalAdd::Neg(
        a_shmem,
        c_shmem,
        6);

    DataCopy(c_gpu, c_cpu);

    Check2x3(c_cpu,
             1,0,-1,
             -2,-3,-4);

    std::cout << "ok\n";
}

void TestBackendTanhMetal()
{
    std::cout << "Backend Tanh (Metal)\t";

    Matrix<float, DeviceTags::CPU> a_cpu(2,3);
    Matrix<float, DeviceTags::CPU> c_cpu(2,3);

    Matrix<float, DeviceTags::Metal> a_gpu(2,3);
    Matrix<float, DeviceTags::Metal> c_gpu(2,3);

    Fill2x3(a_cpu,-1,0,1,2,3,4);

    DataCopy(a_cpu, a_gpu);

    auto a_mem = LowerAccess(a_gpu);
    auto c_mem = LowerAccess(c_gpu);

    auto a_shmem = a_mem.SharedMemory();
    auto c_shmem = c_mem.SharedMemory();

    NSMetalAdd::Tanh(
        a_shmem,
        c_shmem,
        6);

    DataCopy(c_gpu, c_cpu);

    Check2x3(c_cpu,
             std::tanh(-1.0f), std::tanh(0.0f), std::tanh(1.0f),
             std::tanh(2.0f), std::tanh(3.0f), std::tanh(4.0f));

    std::cout << "ok\n";
}

void TestBackendSigmoidMetal()
{
    std::cout << "Backend Sigmoid (Metal)\t";

    Matrix<float, DeviceTags::CPU> a_cpu(2,3);
    Matrix<float, DeviceTags::CPU> c_cpu(2,3);

    Matrix<float, DeviceTags::Metal> a_gpu(2,3);
    Matrix<float, DeviceTags::Metal> c_gpu(2,3);

    Fill2x3(a_cpu,-1,0,1,2,3,4);

    DataCopy(a_cpu, a_gpu);

    auto a_mem = LowerAccess(a_gpu);
    auto c_mem = LowerAccess(c_gpu);

    auto a_shmem = a_mem.SharedMemory();
    auto c_shmem = c_mem.SharedMemory();

    NSMetalAdd::Sigmoid(
        a_shmem,
        c_shmem,
        6);

    DataCopy(c_gpu, c_cpu);

    Check2x3(c_cpu,
             1.0f / (1.0f + std::exp(1.0f)),
             1.0f / (1.0f + std::exp(0.0f)),
             1.0f / (1.0f + std::exp(-1.0f)),
             1.0f / (1.0f + std::exp(-2.0f)),
             1.0f / (1.0f + std::exp(-3.0f)),
             1.0f / (1.0f + std::exp(-4.0f)));

    std::cout << "ok\n";
}

void TestBackendExpMetal()
{
    std::cout << "Backend Exp (Metal)\t";

    Matrix<float, DeviceTags::CPU> a_cpu(2,3);
    Matrix<float, DeviceTags::CPU> c_cpu(2,3);

    Matrix<float, DeviceTags::Metal> a_gpu(2,3);
    Matrix<float, DeviceTags::Metal> c_gpu(2,3);

    Fill2x3(a_cpu,-1,0,1,2,3,4);

    DataCopy(a_cpu, a_gpu);

    auto a_mem = LowerAccess(a_gpu);
    auto c_mem = LowerAccess(c_gpu);

    auto a_shmem = a_mem.SharedMemory();
    auto c_shmem = c_mem.SharedMemory();

    NSMetalAdd::Exp(
        a_shmem,
        c_shmem,
        6);

    DataCopy(c_gpu, c_cpu);

    Check2x3(c_cpu,
             std::exp(-1.0f), std::exp(0.0f), std::exp(1.0f),
             std::exp(2.0f), std::exp(3.0f), std::exp(4.0f));

    std::cout << "ok\n";
}

//////////////////////////////////////////////////////////////

int main()
{
    std::cout << "\n=== MetaNN Metal Expression Tests ===\n";

    TestExprAddMetal();
    TestExprSubMetal();
    TestExprHadamardMetal();
    TestExprUnaryMetal();
    TestExprActivationMetal();

    std::cout << "\n=== Metal Backend Tests ===\n";

    TestBackendAddMetal();
    TestBackendSubMetal();
    TestBackendNegMetal();
    TestBackendTanhMetal();
    TestBackendSigmoidMetal();
    TestBackendExpMetal();

    std::cout << "\nAll tests complete\n";

    return 0;
}
