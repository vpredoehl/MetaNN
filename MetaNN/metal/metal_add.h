//
//  metal_add.h
//  MetaNNXC
//
//  Created by Vincent Predoehl on 3/8/26.
//

#pragma once
#include <cstddef>
#include <MetaNN/data/facilities/continuous_memory.h>

namespace MetaNN::NSMetalAdd
{
    void Add(const ContinuousMemory<float, DeviceTags::Metal>& a,
             const ContinuousMemory<float, DeviceTags::Metal>& b,
             ContinuousMemory<float, DeviceTags::Metal>& c,
             size_t count);
}

