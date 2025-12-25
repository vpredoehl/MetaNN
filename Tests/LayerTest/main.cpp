#include <MetaNN/meta_nn.h>
#include <cassert>
#include <iostream>

using namespace MetaNN;
using std::cout;
using std::endl;

// Helper to print a 2D Tensor/Matrix after evaluation
template <class Mat>
void printMatrix(const char* name, const Mat& mat)
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

//namespace Test::Layer
//{
//    void test_composite();
//    void test_principal();
//    void test_recurrent();
//}
extern "C" void cpp_run_matmul_add_demo();
extern "C" void cpp_run_gpu_demos();


int main()
{
//    Test::Layer::test_composite();
//    Test::Layer::test_principal();
//    Test::Layer::test_recurrent();
  
  
  std::cout << "-- Running C++ GPU MatMul+Add demo --" << std::endl;
  cpp_run_matmul_add_demo();

  std::cout << "-- Running additional GPU demos --" << std::endl;
  cpp_run_gpu_demos();


    using Elem = float;     // Typically float
    using Dev  = DeviceTags::CPU;      // Typically CPU device

    // Dimensions
    const size_t M = 4;  // rows of A and C
    const size_t K = 4;  // cols of A and rows of B
    const size_t N = 4;  // cols of B and C

    // A: M x K
    Matrix<Elem, Dev> A(M, K);
    // B: K x N
    Matrix<Elem, Dev> B(K, N);
    // D: M x N (addend)
    Matrix<Elem, Dev> D(M, N);

    // Fill A, B, D with some values
    int c = 0;
    for (size_t i = 0; i < M; ++i)
    {
        for (size_t j = 0; j < K; ++j)
        {
            A.SetValue(i, j, static_cast<Elem>(c++));
        }
    }

    c = 1;
    for (size_t i = 0; i < K; ++i)
    {
        for (size_t j = 0; j < N; ++j)
        {
            B.SetValue(i, j, static_cast<Elem>(c++));
        }
    }

    c = 10;
    for (size_t i = 0; i < M; ++i)
    {
        for (size_t j = 0; j < N; ++j)
        {
            D.SetValue(i, j, static_cast<Elem>(c++));
        }
    }
  
    cout << "A shape: " << A.Shape()[0] << " x " << A.Shape()[1] << endl;
    cout << "B shape: " << B.Shape()[0] << " x " << B.Shape()[1] << endl;

    // Print input matrices
    printMatrix("A", A);
    printMatrix("B", B);
    printMatrix("D", D);

    // Compute C = A * B
  auto Cmul = Dot(A,B); // MetaNN overloads operator* for matrix multiplication
    // Then C = (A * B) + D
    auto C = Cmul + D; // Element-wise addition with the addend matrix D
  auto E = 2 * C;
  auto t = Tanh(C);

    // Evaluate (A*B)
    auto abHandle = Cmul.EvalRegister();
  auto sumHandle = C.EvalRegister();
  auto h2 = E.EvalRegister();
  auto th = t.EvalRegister();
  
  EvalPlan::Inst().Eval();
  
  const auto& AB = abHandle.Data();
  const auto& ABD = sumHandle.Data();
//  const auto& E2 = h2.Data();
  const auto& tnh = th.Data();
  printMatrix("(A*B)+D", ABD);
  printMatrix("A*B", AB);
  printMatrix("Tanh", tnh);

    // Sanity-check dimensions
    assert(AB.Shape()[0] == M);
    assert(AB.Shape()[1] == N);
    assert(ABD.Shape()[0] == M);
    assert(ABD.Shape()[1] == N);

    cout << "done" << endl;
    return 0;
}
