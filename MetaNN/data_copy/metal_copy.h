//
//  metal_copy.h
//  MetaNNXC
//
//  Created by Vincent Predoehl on 3/7/26.
//

#pragma once
#include <cstddef>

namespace MetaNN
{
    template<typename TElem>
    class ContinuousMemory<TElem, DeviceTags::Metal>;

    namespace NSMetalCopy
    {
        template<typename TElem>
        void CopyBuffer(const ContinuousMemory<TElem, DeviceTags::Metal>& src,
                        ContinuousMemory<TElem, DeviceTags::Metal>& dst,
                        size_t count);
    }
}

