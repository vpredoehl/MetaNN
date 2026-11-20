#pragma once

#include <MetaNN/data/facilities/checked_memory.h>
#include <algorithm>
#include <cstdint>
#include <ostream>

namespace MetaNN::NSMetalMatMul::Detail
{
    struct MatrixSize
    {
        std::size_t elements;
        std::size_t rowBytes;
        std::size_t bytes;
    };

    inline MatrixSize CheckedMatrixSize(std::size_t rows, std::size_t columns)
    {
        const auto elements = MemoryDetail::CheckedProduct(rows, columns);
        return {elements, MemoryDetail::ByteCount<float>(columns),
                MemoryDetail::ByteCount<float>(elements)};
    }

    inline std::uint32_t ShaderUInt(std::size_t value)
    {
        if (value > std::numeric_limits<std::uint32_t>::max())
            throw std::overflow_error("Metal matrix shader index exceeds uint32");
        return static_cast<std::uint32_t>(value);
    }

    struct BiasDimensions
    {
        std::uint32_t rows;
        std::uint32_t columns;
    };

    inline BiasDimensions CheckedBiasDimensions(std::size_t rows, std::size_t columns)
    {
        const BiasDimensions dims{ShaderUInt(rows), ShaderUInt(columns)};
        const auto elements = MemoryDetail::CheckedProduct(rows, columns);
        if (elements != 0)
            ShaderUInt(elements - 1); // The shader stores the final element index in uint.
        return dims;
    }

    struct GateDimensions
    {
        std::uint32_t batch;
        std::uint32_t hidden;
        std::uint32_t stateElements;
        std::size_t gateElements;
    };

    inline GateDimensions CheckedGateDimensions(std::size_t batch, std::size_t hidden)
    {
        const auto batchCount = ShaderUInt(batch);
        const auto hiddenCount = ShaderUInt(hidden);
        const auto stateElements = ShaderUInt(MemoryDetail::CheckedProduct(batch, hidden));
        const auto stride = ShaderUInt(MemoryDetail::CheckedProduct(4, hidden));
        const auto gateElements = MemoryDetail::CheckedProduct(batch, stride);
        if (gateElements != 0)
            ShaderUInt(gateElements - 1);
        return {batchCount, hiddenCount, stateElements, gateElements};
    }

}
