//
//  metal_matmull.h
//  MetaNNXC
//
//  Created by Vincent Predoehl on 3/8/26.
//

#pragma once

#include "continuous_memory.h"
#include <cstddef>

namespace MetaNN::NSMetalMatMul
{
    void MatMul(const ContinuousMemory<float, DeviceTags::Metal>& a,
                const ContinuousMemory<float, DeviceTags::Metal>& b,
                ContinuousMemory<float, DeviceTags::Metal>& c,
                size_t m, size_t k, size_t n);
    void MatMulBias(const ContinuousMemory<float, DeviceTags::Metal>& a,
                    const ContinuousMemory<float, DeviceTags::Metal>& b,
                    const ContinuousMemory<float, DeviceTags::Metal>& bias,
                    ContinuousMemory<float, DeviceTags::Metal>& c,
                    size_t m, size_t k, size_t n);
}

