#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace MetaNN::NSMetalAdd::Detail
{
    // The native handle is the allocation base. Offset() and Size() are in
    // elements. Only the encoder applies the offset, in bytes.
    template <typename Element, typename View, typename Bind>
    void BindView(const View& view, std::size_t bufferBytes, std::size_t count,
                  std::size_t slot, Bind&& bind)
    {
        constexpr auto maxElements = std::numeric_limits<std::size_t>::max() / sizeof(Element);
        const std::size_t start = view.Offset();
        if (start > maxElements || count > maxElements ||
            count > std::numeric_limits<std::uint32_t>::max())
            throw std::overflow_error("MetalAdd operand offset or count overflow");

        const std::size_t offsetBytes = start * sizeof(Element);
        const std::size_t countBytes = count * sizeof(Element);
        if (count > view.Size() || offsetBytes > bufferBytes ||
            countBytes > bufferBytes - offsetBytes)
            throw std::out_of_range("MetalAdd operand exceeds view or buffer");

        bind(offsetBytes, slot);
    }
}
