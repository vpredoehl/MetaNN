//
//  continuous_memory_metal.mm
//  MetaNNXC
//
//  Created by Vincent Predoehl on 3/7/26.
//
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "continuous_memory.h"
#include <stdexcept>
#include <utility>

namespace
{
    inline id<MTLDevice> GetMetalDevice()
    {
        static id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device)
        {
            throw std::runtime_error("Metal device not available.");
        }
        return device;
    }

    inline void GpuCopyBuffer(id<MTLBuffer> srcBuffer,
                              size_t srcOffsetElems,
                              id<MTLBuffer> dstBuffer,
                              size_t dstOffsetElems,
                              size_t elemCount,
                              size_t elemSize)
    {
        if (elemCount == 0)
        {
            return;
        }

        id<MTLDevice> device = GetMetalDevice();
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue)
        {
            throw std::runtime_error("Failed to create Metal command queue.");
        }

        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        if (!cmd)
        {
            throw std::runtime_error("Failed to create Metal command buffer.");
        }

        id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
        if (!blit)
        {
            throw std::runtime_error("Failed to create Metal blit command encoder.");
        }

        const size_t srcOffsetBytes = srcOffsetElems * elemSize;
        const size_t dstOffsetBytes = dstOffsetElems * elemSize;
        const size_t copyBytes = elemCount * elemSize;

        [blit copyFromBuffer:srcBuffer
                sourceOffset:srcOffsetBytes
                    toBuffer:dstBuffer
           destinationOffset:dstOffsetBytes
                        size:copyBytes];

        [blit endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        if (cmd.status == MTLCommandBufferStatusError)
        {
            throw std::runtime_error("Metal buffer copy failed.");
        }
    }
}

namespace MetaNN
{
template<typename TElem>
ContinuousMemory<TElem, DeviceTags::Metal>
ContinuousMemory<TElem, DeviceTags::Metal>::Shift(size_t pos) const
{
    assert(pos < m_size);
    return ContinuousMemory(m_impl, m_offset + pos, m_size - pos);
}

template<typename TElem>
bool ContinuousMemory<TElem, DeviceTags::Metal>::IsShared() const
{
    return m_impl.use_count() > 1;
}

template<typename TElem>
size_t ContinuousMemory<TElem, DeviceTags::Metal>::Size() const
{
    return m_size;
}

template<typename TElem>
struct ContinuousMemory<TElem, DeviceTags::Metal>::Impl
{
    id<MTLBuffer> buffer = nil;
};

template<typename TElem>
ContinuousMemory<TElem, DeviceTags::Metal>::ContinuousMemory(size_t p_size)
    : m_impl(std::make_shared<Impl>())
    , m_offset(0)
    , m_size(p_size)
{
    if (p_size == 0)
    {
        throw std::runtime_error("Metal ContinuousMemory allocated with size 0");
    }

    id<MTLDevice> device = GetMetalDevice();
    m_impl->buffer = [device newBufferWithLength:sizeof(TElem) * p_size
                                         options:MTLResourceStorageModeShared];

    if (!m_impl->buffer)
    {
        throw std::runtime_error("Failed to allocate Metal buffer.");
    }
}

template<typename TElem>
ContinuousMemory<TElem, DeviceTags::Metal>::
ContinuousMemory(const ContinuousMemory& rhs)
    : ContinuousMemory(rhs.m_size)
{
    GpuCopyBuffer(rhs.m_impl->buffer,
                  rhs.m_offset,
                  m_impl->buffer,
                  0,
                  rhs.m_size,
                  sizeof(TElem));
}

template<typename TElem>
ContinuousMemory<TElem, DeviceTags::Metal>&
ContinuousMemory<TElem, DeviceTags::Metal>::operator=(const ContinuousMemory& rhs)
{
    if (this == &rhs)
    {
        return *this;
    }

    ContinuousMemory tmp(rhs);
    *this = std::move(tmp);
    return *this;
}
template<typename TElem>
const TElem* ContinuousMemory<TElem, DeviceTags::Metal>::RawMemory() const
{
    return reinterpret_cast<const TElem*>(
        static_cast<char*>([m_impl->buffer contents])) + m_offset;
}

template<typename TElem>
TElem* ContinuousMemory<TElem, DeviceTags::Metal>::MutableRawMemory()
{
    return reinterpret_cast<TElem*>(
        static_cast<char*>([m_impl->buffer contents])) + m_offset;
}

template<typename TElem>
void* ContinuousMemory<TElem, DeviceTags::Metal>::NativeHandle() const
{
    return (__bridge void*)m_impl->buffer;
}

// explicit instantiations
template class ContinuousMemory<float, DeviceTags::Metal>;
template class ContinuousMemory<double, DeviceTags::Metal>;
}

