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

kernel void gate_state_fused_f32(
    device const float* gates      [[buffer(0)]],
    device const float* prevCell   [[buffer(1)]],
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
    const uint h = gid % hiddenSize;

    const uint gateRowBase = b * (4u * hiddenSize);
    const uint stateIndex = b * hiddenSize + h;

    const float iLogits = gates[gateRowBase + h];
    const float fLogits = gates[gateRowBase + hiddenSize + h];
    const float gLogits = gates[gateRowBase + 2u * hiddenSize + h];
    const float oLogits = gates[gateRowBase + 3u * hiddenSize + h];

    const float i = 1.0f / (1.0f + exp(-iLogits));
    const float f = 1.0f / (1.0f + exp(-fLogits));
    const float g = tanh(gLogits);
    const float o = 1.0f / (1.0f + exp(-oLogits));

    const float prevC = prevCell[stateIndex];
    const float c = f * prevC + i * g;
    const float hOut = o * tanh(c);

    gateI[stateIndex] = i;
    gateF[stateIndex] = f;
    gateG[stateIndex] = g;
    gateO[stateIndex] = o;
    cellOut[stateIndex] = c;
    hiddenOut[stateIndex] = hOut;
}
