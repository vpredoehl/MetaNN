#pragma once

#include <MetaNN/data/facilities/checked_memory.h>
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
        if (count > std::numeric_limits<std::uint32_t>::max())
            throw std::overflow_error("MetalAdd shader count overflow");

        const auto offsetBytes = MemoryDetail::CheckedViewOffset<Element>(view, bufferBytes, count);
        bind(offsetBytes, slot);
    }
}
