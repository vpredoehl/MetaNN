//
//  add.metal
//  MetaNNXC
//
//  Created by Vincent Predoehl on 3/8/26.
//

#include <metal_stdlib>
using namespace metal;

kernel void vector_add_f32(device const float* a [[buffer(0)]],
                           device const float* b [[buffer(1)]],
                           device float* c [[buffer(2)]],
                           constant uint& count [[buffer(3)]],
                           uint gid [[thread_position_in_grid]])
{
    if (gid < count)
    {
        c[gid] = a[gid] + b[gid];
    }
}

