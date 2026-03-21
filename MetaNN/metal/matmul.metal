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

