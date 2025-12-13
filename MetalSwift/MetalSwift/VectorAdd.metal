// VectorAdd.metal
#include <metal_stdlib>
using namespace metal;

kernel void vector_add(
    device const float *inA [[ buffer(0) ]],
    device const float *inB [[ buffer(1) ]],
    device float *outC [[ buffer(2) ]],
    constant uint &count [[ buffer(3) ]],
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid < count) {
        outC[tid] = inA[tid] + inB[tid];
    }
}
