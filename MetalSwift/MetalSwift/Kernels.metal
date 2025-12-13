#include <metal_stdlib>
using namespace metal;

struct VecParams {
    uint count;
};

kernel void vec_add(const device float* a   [[buffer(0)]],
                    const device float* b   [[buffer(1)]],
                    device float* out       [[buffer(2)]],
                    constant VecParams& p   [[buffer(3)]],
                    uint gid [[thread_position_in_grid]]) {
    if (gid >= p.count) return;
    out[gid] = a[gid] + b[gid];
}
