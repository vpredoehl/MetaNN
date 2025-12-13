#ifndef CPP_GPU_BRIDGE_H
#define CPP_GPU_BRIDGE_H

#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

// Runs vector add on the GPU using Metal. Returns 0 on success, non-zero on failure.
// - a, b: input arrays of length `count`
// - out: output array of length `count`
// - count: number of elements
int cpp_gpu_vector_add(const float *a, const float *b, float *out, size_t count);

#ifdef __cplusplus
}
#endif

#endif // CPP_GPU_BRIDGE_H
