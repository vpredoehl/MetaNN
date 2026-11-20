#include <MetaNN/data/tensor.h>
#include <MetaNN/metal/metal_matrix_checks.h>
#include <MetaNN/metal/metal_view_binding.h>
#include <cstdint>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace
{
using namespace MetaNN;
using namespace MetaNN::MemoryDetail;
using namespace MetaNN::NSMetalMatMul::Detail;
constexpr auto maxSize = std::numeric_limits<std::size_t>::max();
constexpr auto maxShader = std::numeric_limits<std::uint32_t>::max();
unsigned checks = 0;

void Check(bool valid, const char* message)
{
    ++checks;
    if (!valid)
        throw std::runtime_error(message);
}

template <typename Error, typename Run>
void Reject(Run run, const char* message)
{
    try
    {
        run();
    }
    catch (const Error&)
    {
        ++checks;
        return;
    }
    throw std::runtime_error(message);
}

struct MemoryRange
{
    std::size_t offset;
    std::size_t length;
    std::size_t Offset() const { return offset; }
    std::size_t Size() const { return length; }
};

void Allocation()
{
    Check(CpuAllocationBytes<float>(0) == 0, "CPU zero allocation");
    Check(CpuAllocationBytes<float>(256) == 1024, "exact CPU bucket");
    Check(CpuAllocationBytes<float>(257) == 2048, "round up CPU bucket");
    Check(CpuAllocationBytes<double>(129) == 2048, "double CPU bucket");
    Check(ByteCount<double>(maxSize / sizeof(double)) == maxSize - 7,
          "last representable double byte count");
    Reject<std::overflow_error>([] { ByteCount<double>(maxSize / sizeof(double) + 1); },
                                "Metal double byte wrap accepted");
    // The original byte product wrapped to four; the original bucket round-up
    // wrapped to zero. Both inputs are rejected before the allocator is entered.
    Reject<std::overflow_error>([] {
        ContinuousMemory<float, DeviceTags::CPU> bad(maxSize / sizeof(float) + 2);
    }, "CPU allocation byte wrap accepted");
    Reject<std::overflow_error>([] {
        ContinuousMemory<std::uint8_t, DeviceTags::CPU> bad(maxSize);
    }, "CPU allocation padding wrap accepted");
    Reject<std::overflow_error>([] {
        Matrix<float, DeviceTags::CPU> bad(maxSize / 2 + 2, 2);
    }, "matrix allocation element wrap accepted");
    Reject<std::overflow_error>([] {
        Matrix<float, DeviceTags::CPU> bad(maxSize / sizeof(float) + 2, 1);
    }, "matrix allocation byte wrap accepted");
    Reject<std::overflow_error>([] { Shape(2, maxSize, 2).Count(); },
                                "batched dimension wrap accepted");
    Check(Shape(maxSize, 2, 0).Count() == 0, "empty shape product");
    Reject<std::invalid_argument>([] { Shape(-1, 1); }, "negative dimension accepted");
    Reject<std::invalid_argument>([] { Shape(1.5, 1); }, "fractional dimension accepted");
    Reject<std::invalid_argument>([] { Shape(std::numeric_limits<double>::infinity(), 1); },
                                  "infinite dimension accepted");
    Check(Shape(2.0, 3).Count() == 6, "integral floating dimension");
    std::array<std::size_t, 2> index{0, maxSize - 1};
    Shape(2, maxSize).ShiftIndex(index, 2);
    Check(index[0] == 1 && index[1] == 1, "shape coordinate carry overflow");
    Reject<std::out_of_range>([] { Shape(0, 1).OffsetToIndex(0); },
                              "empty shape accepted offset");
}

void Views()
{
    ContinuousMemory<double, DeviceTags::CPU> allocation(8);
    auto* base = allocation.MutableRawMemory();
    for (std::size_t i = 0; i < allocation.Size(); ++i)
        base[i] = -123;
    const auto same = allocation.Shift(0);
    const auto first = same.Shift(2);
    auto last = first.Shift(5);
    Check(same.RawMemory() == base && same.Size() == 8, "zero shift changes view");
    Check(first.RawMemory() == base + 2 && first.Size() == 6, "first suffix");
    Check(last.RawMemory() == base + 7 && last.Size() == 1, "final element suffix");
    Check(allocation.IsShared() && first.IsShared(), "shared ownership lost");
    last.MutableRawMemory()[0] = 42;
    for (std::size_t i = 0; i < 7; ++i)
        Check(base[i] == -123, "shifted write touched guard");
    Check(first.RawMemory()[5] == 42, "shifted write missed final element");
    const auto retained = [] {
        ContinuousMemory<double, DeviceTags::CPU> owner(2);
        owner.MutableRawMemory()[1] = 77;
        return owner.Shift(1);
    }();
    Check(retained.RawMemory()[0] == 77, "suffix did not retain allocation");
    Reject<std::out_of_range>([&] { allocation.Shift(8); }, "one-past shift accepted");
    Reject<std::out_of_range>([&] { first.Shift(maxSize); }, "huge shift accepted");
    Reject<std::out_of_range>([] {
        ContinuousMemory<float, DeviceTags::CPU>(0).Shift(0);
    }, "empty shift accepted");
    Reject<std::out_of_range>([&] {
        Matrix<double, DeviceTags::CPU> bad(first, Shape(2, 4));
    }, "shape exceeding existing memory view accepted");
    Matrix<double, DeviceTags::CPU> matrix(allocation, Shape(2, 4));
    Check(matrix[1][3].Value() == 42, "tensor nested slice");
    Reject<std::out_of_range>([&] { (void)matrix[maxSize / 4 + 1]; },
                              "tensor index multiplication wrapped to first row");
    Check(ShiftedOffset(0, 8, 2) == 2 && ShiftedOffset(2, 6, 5) == 7,
          "Metal nested element offset");
    Reject<std::overflow_error>([] { ShiftedOffset(maxSize - 1, 3, 2); },
                                "Metal offset accumulation wrapped");
    Check(CheckedViewOffset<double>(MemoryRange{7, 1}, 64, 1) == 56,
          "native double final-element offset");
    Check(CheckedViewOffset<float>(MemoryRange{8, 0}, 32, 0) == 32,
          "non-accessing empty native range at end");
    Reject<std::out_of_range>([] { CheckedViewOffset<float>(MemoryRange{7, 1}, 32, 2); },
                              "native logical overrun accepted");
    Reject<std::out_of_range>([] {
        CheckedViewOffset<std::uint8_t>(MemoryRange{maxSize - 1, 2}, maxSize, 2);
    }, "native offset plus length wrap accepted");
    Reject<std::overflow_error>([] {
        CheckedViewOffset<float>(MemoryRange{maxSize / 4 + 1, 1}, maxSize, 1);
    }, "element offset byte wrap accepted");
}

void Matrices()
{
    const auto bytes = CheckedMatrixSize(2, 3);
    Check(bytes.elements == 6 && bytes.rowBytes == 12 && bytes.bytes == 24,
          "contiguous float32 matrix layout");
    Reject<std::overflow_error>([] { CheckedMatrixSize(maxSize / 2 + 1, 2); },
                                "matrix element product wrapped");
    Reject<std::overflow_error>([] { CheckedMatrixSize(1, maxSize / 4 + 1); },
                                "matrix row stride wrapped");
    Reject<std::overflow_error>([] { CheckedMatrixSize(maxSize / 4 + 1, 1); },
                                "matrix descriptor bytes wrapped");
    Check(CheckedBiasDimensions(2, (std::size_t{1} + maxShader) / 2).rows == 2,
          "last representable bias index");
    Reject<std::overflow_error>([] { CheckedBiasDimensions(std::size_t{1} + maxShader, 1); },
                                "bias dimension narrowed");
    Reject<std::overflow_error>([] { CheckedBiasDimensions(3, maxShader); },
                                "bias shader index wrapped");
    const auto gate = CheckedGateDimensions(2, 8);
    Check(gate.batch == 2 && gate.hidden == 8 && gate.stateElements == 16 &&
          gate.gateElements == 64, "fused gate layout");
    Check(CheckedGateDimensions(2, (std::size_t{1} + maxShader) / 8).gateElements ==
              std::size_t{1} + maxShader, "last representable fused gate index");
    Reject<std::overflow_error>([] { CheckedGateDimensions(1, maxShader / 4 + 1); },
                                "fused gate row stride wrapped");
    Reject<std::overflow_error>([] { CheckedGateDimensions(maxShader, 2); },
                                "fused gate thread count wrapped");
    Reject<std::overflow_error>([] { CheckedGateDimensions(6, maxShader / 8); },
                                "fused gate packed index wrapped");
}

struct HostFixture
{
    const float* base;
    ContinuousMemory<float, DeviceTags::CPU> view;
    const float* RawMemory() const { return view.RawMemory(); }
    const float* AllocationBase() const { return base; }
};

void GateDiagnostic()
{
    ContinuousMemory<float, DeviceTags::CPU> allocation(6);
    auto* base = allocation.MutableRawMemory();
    base[0] = base[1] = -123;
    base[2] = 6;
    base[3] = 8;
    base[4] = 10;
    base[5] = 12;
    std::ostringstream out;
    WriteGateHostProbe(out, HostFixture{base, allocation.Shift(1).Shift(1)},
                       CheckedGateDimensions(1, 1).gateElements);
    Check(out.str() == "DIAG_GATESTATE_GATES_HOST,g0=6,g1=8,g2=10,g3=12\n",
          "gate diagnostic observed the allocation base instead of the shifted view");
    Check(base[0] == -123 && base[1] == -123, "gate diagnostic modified guards");
}
}

int main(int argc, char** argv)
{
    try
    {
        if (argc == 2 && std::string_view(argv[1]) == "gate")
            GateDiagnostic();
        else
        {
            Allocation();
            Views();
            Matrices();
            GateDiagnostic();
        }
        std::cout << "MEMORY_SAFETY_PASS checks=" << checks << '\n';
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
