//
//  continuous_memory_metal.mm
//  MetaNNXC
//
//  Created by Vincent Predoehl on 3/7/26.
//
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "continuous_memory.h"
#include <atomic>
#include <iostream>
#include <type_traits>
#include <array>
#include <mutex>
#include <vector>

namespace
{
    std::atomic<size_t> g_metal_buffer_alloc_count{0};
    std::atomic<size_t> g_metal_buffer_free_count{0};
    std::atomic<size_t> g_metal_buffer_live_count{0};
    std::atomic<size_t> g_metal_buffer_live_bytes{0};
    std::atomic<size_t> g_metal_buffer_reuse_count{0};
    std::atomic<size_t> g_metal_buffer_pool_return_count{0};
    std::atomic<size_t> g_metal_buffer_pool_drop_count{0};

    constexpr size_t kMetalSmallBufferPoolMaxBytes = 4096;
    constexpr size_t kMetalSmallBufferPoolMaxPerSize = 256;
    constexpr size_t kMetalSmallBufferPoolBucketCount = kMetalSmallBufferPoolMaxBytes / sizeof(float) + 1;

    std::mutex g_metal_small_buffer_pool_mutex;
    std::array<std::vector<id<MTLBuffer>>, kMetalSmallBufferPoolBucketCount> g_metal_small_buffer_pool;

    inline const char* MetalElemNameFloatDouble(size_t elemBytes)
    {
        if (elemBytes == sizeof(float)) return "float_or_same_size";
        if (elemBytes == sizeof(double)) return "double_or_same_size";
        return "other";
    }

    inline bool ShouldPrintMetalBufferDiag(size_t count)
    {
        return count <= 16 || (count % 1024) == 0;
    }

    inline bool IsMetalSmallPoolEligible(size_t bytes)
    {
        return bytes != 0 && bytes <= kMetalSmallBufferPoolMaxBytes && (bytes % sizeof(float)) == 0;
    }

    inline size_t MetalSmallPoolBucketIndex(size_t bytes)
    {
        return bytes / sizeof(float);
    }

    inline id<MTLBuffer> TryTakeSmallMetalBufferFromPool(size_t bytes)
    {
        if (!IsMetalSmallPoolEligible(bytes))
        {
            return nil;
        }

        std::lock_guard<std::mutex> lock(g_metal_small_buffer_pool_mutex);
        auto& bucket = g_metal_small_buffer_pool[MetalSmallPoolBucketIndex(bytes)];
        if (bucket.empty())
        {
            return nil;
        }

        id<MTLBuffer> buffer = bucket.back();
        bucket.pop_back();
        return buffer;
    }

    inline bool ReturnSmallMetalBufferToPool(id<MTLBuffer> buffer, size_t bytes)
    {
        if (!buffer || !IsMetalSmallPoolEligible(bytes))
        {
            return false;
        }

        std::lock_guard<std::mutex> lock(g_metal_small_buffer_pool_mutex);
        auto& bucket = g_metal_small_buffer_pool[MetalSmallPoolBucketIndex(bytes)];
        if (bucket.size() >= kMetalSmallBufferPoolMaxPerSize)
        {
            return false;
        }

        bucket.push_back(buffer);
        return true;
    }
}

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
    size_t bytes = 0;
    size_t elems = 0;
    bool pooled = false;

    ~Impl()
    {
        if (buffer != nil && bytes != 0)
        {
            const size_t freeCount = g_metal_buffer_free_count.fetch_add(1, std::memory_order_relaxed) + 1;
            const size_t liveCount = g_metal_buffer_live_count.fetch_sub(1, std::memory_order_relaxed) - 1;
            const size_t liveBytes = g_metal_buffer_live_bytes.fetch_sub(bytes, std::memory_order_relaxed) - bytes;

            const bool returnedToPool = ReturnSmallMetalBufferToPool(buffer, bytes);
            if (returnedToPool)
            {
                pooled = true;
                g_metal_buffer_pool_return_count.fetch_add(1, std::memory_order_relaxed);
            }
            else if (IsMetalSmallPoolEligible(bytes))
            {
                g_metal_buffer_pool_drop_count.fetch_add(1, std::memory_order_relaxed);
            }

            if (ShouldPrintMetalBufferDiag(freeCount))
            {
                std::cout << "DIAG_METAL_BUFFER_FREE"
                          << ",count=" << freeCount
                          << ",elems=" << elems
                          << ",bytes=" << bytes
                          << ",live_count=" << liveCount
                          << ",live_bytes=" << liveBytes
                          << ",pooled=" << (returnedToPool ? 1 : 0)
                          << "\n";
            }
        }
    }
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

    const size_t allocBytes = sizeof(TElem) * p_size;
    bool reusedBuffer = false;
    m_impl->buffer = TryTakeSmallMetalBufferFromPool(allocBytes);
    if (m_impl->buffer)
    {
        reusedBuffer = true;
        g_metal_buffer_reuse_count.fetch_add(1, std::memory_order_relaxed);
    }
    else
    {
        m_impl->buffer = [device newBufferWithLength:allocBytes
                                options:MTLResourceStorageModeShared];
    }

    if (!m_impl->buffer)  throw std::runtime_error("Failed to allocate Metal buffer.");

    m_impl->bytes = allocBytes;
    m_impl->elems = p_size;

    const size_t allocCount = g_metal_buffer_alloc_count.fetch_add(1, std::memory_order_relaxed) + 1;
    const size_t liveCount = g_metal_buffer_live_count.fetch_add(1, std::memory_order_relaxed) + 1;
    const size_t liveBytes = g_metal_buffer_live_bytes.fetch_add(allocBytes, std::memory_order_relaxed) + allocBytes;

    if (ShouldPrintMetalBufferDiag(allocCount))
    {
        std::cout << "DIAG_METAL_BUFFER_ALLOC"
                  << ",count=" << allocCount
                  << ",elem=" << MetalElemNameFloatDouble(sizeof(TElem))
                  << ",elems=" << p_size
                  << ",bytes=" << allocBytes
                  << ",live_count=" << liveCount
                  << ",live_bytes=" << liveBytes
                  << ",reused=" << (reusedBuffer ? 1 : 0)
                  << "\n";
    }
    if (reusedBuffer)
    {
        const size_t reuseCount = g_metal_buffer_reuse_count.load(std::memory_order_relaxed);
        if (ShouldPrintMetalBufferDiag(reuseCount))
        {
            std::cout << "DIAG_METAL_BUFFER_REUSE"
                      << ",count=" << reuseCount
                      << ",elems=" << p_size
                      << ",bytes=" << allocBytes
                      << ",live_count=" << liveCount
                      << ",live_bytes=" << liveBytes
                      << "\n";
        }
    }
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
