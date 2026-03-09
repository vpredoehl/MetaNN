//
//  metal_add.h
//  MetaNNXC
//
//  Created by Vincent Predoehl on 3/8/26.
//

#pragma once

#include "continuous_memory.h"
#include <cstddef>

namespace MetaNN::NSMetalAdd
{
    void Add(const ContinuousMemory<float, DeviceTags::Metal>& a,
             const ContinuousMemory<float, DeviceTags::Metal>& b,
             ContinuousMemory<float, DeviceTags::Metal>& c,
             size_t count);

    void Sub(const ContinuousMemory<float, DeviceTags::Metal>& a,
             const ContinuousMemory<float, DeviceTags::Metal>& b,
             ContinuousMemory<float, DeviceTags::Metal>& c,
             size_t count);

    void Neg(const ContinuousMemory<float, DeviceTags::Metal>& a,
             ContinuousMemory<float, DeviceTags::Metal>& c,
             size_t count);

    void Tanh(const ContinuousMemory<float, DeviceTags::Metal>& a,
              ContinuousMemory<float, DeviceTags::Metal>& c,
              size_t count);

    void Sigmoid(const ContinuousMemory<float, DeviceTags::Metal>& a,
                 ContinuousMemory<float, DeviceTags::Metal>& c,
                 size_t count);

    void Exp(const ContinuousMemory<float, DeviceTags::Metal>& a,
             ContinuousMemory<float, DeviceTags::Metal>& c,
             size_t count);
}
