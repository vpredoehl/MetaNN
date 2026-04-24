#pragma once

#include <MetaNN/data/facilities/allocators.h>
#include <MetaNN/facilities/traits.h>
#include <cassert>

#include <cstddef>
#include <memory>
#include <type_traits>
#include <stdexcept>


namespace MetaNN
{
template <typename TElem, typename TDevice>
class ContinuousMemory
{
    static_assert(std::is_same<RemConstRef<TElem>, TElem>::value);
    using ElementType = TElem;
    
public:
    explicit ContinuousMemory(size_t p_size)
        : m_mem(Allocator<TDevice>::template Allocate<ElementType>(p_size))
        , m_size(p_size)
    {}

    ContinuousMemory Shift(size_t pos) const
    {
        assert(pos < m_size);
        return ContinuousMemory(std::shared_ptr<ElementType>(m_mem, m_mem.get() + pos),
                                m_size - pos);
    }
    
    auto RawMemory() const  { return m_mem.get(); }
    TElem* MutableRawMemory() { return m_mem.get();  }

    bool IsShared() const
    {
        return m_mem.use_count() > 1;
    }
    
    size_t Size() const
    {
        return m_size;
    }
    
    bool operator== (const ContinuousMemory& val) const noexcept
    {
        return (m_mem == val.m_mem) && (m_size == val.m_size);
    }

    bool operator!= (const ContinuousMemory& val) const noexcept
    {
        return !(operator==(val));
    }

private:
    ContinuousMemory(std::shared_ptr<ElementType> ptr, size_t p_size)
        : m_mem(std::move(ptr))
        , m_size(p_size)
    {}
    
private:
    std::shared_ptr<ElementType> m_mem;
    size_t m_size;
};


template<typename TElem>
class ContinuousMemory<TElem, DeviceTags::Metal>
{
public:
    explicit ContinuousMemory(size_t p_size);
    ContinuousMemory Shift(size_t pos) const;

    const TElem* RawMemory() const;
    TElem* MutableRawMemory();
    bool IsShared() const;
    size_t Size() const;
    void* NativeHandle() const;
    size_t Offset() const;
private:
    struct Impl;
    std::shared_ptr<Impl> m_impl;
    size_t m_offset = 0;
    size_t m_size = 0;
};
}

