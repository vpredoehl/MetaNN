//
//  metal_matmull.h
//  MetaNNXC
//
//  Created by Vincent Predoehl on 3/8/26.
//

#pragma once

#include "continuous_memory.h"

namespace MetaNN::NSMetalMatMul
{
    void WaitForAll();
    void MatMul(const ContinuousMemory<float, DeviceTags::Metal>& a,
                const ContinuousMemory<float, DeviceTags::Metal>& b,
                ContinuousMemory<float, DeviceTags::Metal>& c,
                size_t m, size_t k, size_t n);

    void MatMulBias(const ContinuousMemory<float, DeviceTags::Metal>& a,
                    const ContinuousMemory<float, DeviceTags::Metal>& b,
                    const ContinuousMemory<float, DeviceTags::Metal>& bias,
                    ContinuousMemory<float, DeviceTags::Metal>& c,
                    size_t m, size_t k, size_t n);

    void GateStateFused(const ContinuousMemory<float, DeviceTags::Metal>& gates,
                        const ContinuousMemory<float, DeviceTags::Metal>& prevCell,
                        ContinuousMemory<float, DeviceTags::Metal>& gateI,
                        ContinuousMemory<float, DeviceTags::Metal>& gateF,
                        ContinuousMemory<float, DeviceTags::Metal>& gateG,
                        ContinuousMemory<float, DeviceTags::Metal>& gateO,
                        ContinuousMemory<float, DeviceTags::Metal>& cellOut,
                        ContinuousMemory<float, DeviceTags::Metal>& hiddenOut,
                        size_t batchSize,
                        size_t hiddenSize);
}
