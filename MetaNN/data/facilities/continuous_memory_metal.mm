//
//  continuous_memory_metal.mm
//  MetaNNXC
//
//  Created by Vincent Predoehl on 3/7/26.
//
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "continuous_memory.h"

namespace MetaNN
{
template<typename TElem>
ContinuousMemory<TElem, DeviceTags::Metal>
ContinuousMemory<TElem, DeviceTags::Metal>::Shift(size_t pos) const
{
    assert(pos < m_size);
    ContinuousMemory ret(*this);
    ret.m_offset += pos;
    ret.m_size -= pos;
    return ret;
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
    if (p_size == 0)  throw std::runtime_error("Metal ContinuousMemory allocated with size 0");

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device)  throw std::runtime_error("Metal device not available.");

    m_impl->buffer = [device newBufferWithLength:sizeof(TElem) * p_size
                            options:MTLResourceStorageModeShared];

    if (!m_impl->buffer)  throw std::runtime_error("Failed to allocate Metal buffer.");
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

template<typename TElem>
size_t ContinuousMemory<TElem, DeviceTags::Metal>::Offset() const
{
    return m_offset;
}

// explicit instantiations
template class ContinuousMemory<float, DeviceTags::Metal>;
template class ContinuousMemory<double, DeviceTags::Metal>;
}
