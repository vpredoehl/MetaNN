//
//  matmul.metal
//  MetaNNXC
//
//  Created by Vincent Predoehl on 3/8/26.
//

#include <metal_stdlib>
using namespace metal;

kernel void matrix_mul_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint row = gid.y;
    const uint col = gid.x;

    if (row >= m || col >= n) return;

    float sum = 0.0f;
    const uint aRowBase = row * k;
    for (uint t = 0; t < k; ++t)
    {
        sum += a[aRowBase + t] * b[t * n + col];
    }
    c[row * n + col] = sum;
}

kernel void add_row_bias_f32(
    device float* c [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    constant uint& m [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint col = gid.x;
    uint row = gid.y;
    if (row >= m || col >= n) return;
    c[row * n + col] += bias[col];
}

inline float sigmoid_f32(float x)
{
    return 1.0f / (1.0f + exp(-x));
}

kernel void gate_state_fused_f32(const device float* gates      [[buffer(0)]],
                                 const device float* prevCell   [[buffer(1)]],
                                 device float* gateI            [[buffer(2)]],
                                 device float* gateF            [[buffer(3)]],
                                 device float* gateG            [[buffer(4)]],
                                 device float* gateO            [[buffer(5)]],
                                 device float* cellOut          [[buffer(6)]],
                                 device float* hiddenOut        [[buffer(7)]],
                                 constant uint& batchSize       [[buffer(8)]],
                                 constant uint& hiddenSize      [[buffer(9)]],
                                 uint gid                       [[thread_position_in_grid]])
{
    const uint total = batchSize * hiddenSize;
    if (gid >= total)
    {
        return;
    }

    const uint b = gid / hiddenSize;
    const uint h = gid - b * hiddenSize;

    const uint gateBase = b * (hiddenSize * 4u);
    const uint rowBase = b * hiddenSize + h;

    const float iPre = gates[gateBase + h + 0u * hiddenSize];
    const float fPre = gates[gateBase + h + 1u * hiddenSize];
    const float gPre = gates[gateBase + h + 2u * hiddenSize];
    const float oPre = gates[gateBase + h + 3u * hiddenSize];
    const float prevC = prevCell[rowBase];

    const float iVal = sigmoid_f32(iPre);
    const float fVal = sigmoid_f32(fPre);
    const float gVal = tanh(gPre);
    const float oVal = sigmoid_f32(oPre);
    const float cVal = fVal * prevC + iVal * gVal;
    const float hVal = oVal * tanh(cVal);

    gateI[rowBase] = iVal;
    gateF[rowBase] = fVal;
    gateG[rowBase] = gVal;
    gateO[rowBase] = oVal;
    cellOut[rowBase] = cVal;
    hiddenOut[rowBase] = hVal;
}
