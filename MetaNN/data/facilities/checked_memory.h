#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace MetaNN::MemoryDetail
{
    inline std::size_t CheckedProduct(std::size_t a, std::size_t b)
    {
        if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b)
            throw std::overflow_error("MetaNN size product overflow");
        return a * b;
    }

    inline std::size_t CheckedSum(std::size_t a, std::size_t b)
    {
        if (b > std::numeric_limits<std::size_t>::max() - a)
            throw std::overflow_error("MetaNN size sum overflow");
        return a + b;
    }

    template <typename Element>
    std::size_t ByteCount(std::size_t elements)
    {
        return CheckedProduct(elements, sizeof(Element));
    }

    template <typename Element>
    std::size_t CpuAllocationBytes(std::size_t elements)
    {
        constexpr std::size_t bucket = 1024;
        const auto bytes = ByteCount<Element>(elements);
        const auto remainder = bytes % bucket;
        return remainder == 0 ? bytes : CheckedSum(bytes, bucket - remainder);
    }

    inline void ValidateShift(std::size_t size, std::size_t pos)
    {
        // Shift creates a nonempty suffix, as required by the original API.
        if (pos >= size)
            throw std::out_of_range("MetaNN shift exceeds memory view");
    }

    inline std::size_t ShiftedOffset(std::size_t offset, std::size_t size, std::size_t pos)
    {
        ValidateShift(size, pos);
        return CheckedSum(offset, pos);
    }

    template <typename Element, typename View>
    std::size_t CheckedViewOffset(const View& view, std::size_t bufferBytes, std::size_t count)
    {
        const auto offsetBytes = ByteCount<Element>(view.Offset());
        const auto countBytes = ByteCount<Element>(count);
        if (count > view.Size() || offsetBytes > bufferBytes ||
            countBytes > bufferBytes - offsetBytes)
            throw std::out_of_range("MetaNN operand exceeds memory view or buffer");

        return offsetBytes;
    }
}
