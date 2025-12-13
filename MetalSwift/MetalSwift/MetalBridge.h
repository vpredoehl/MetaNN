// Deprecated: Objective-C/C bridge disabled. Use Swift-only MetalVectorAdder instead.
#if 0
#ifndef MetalBridge_h
#define MetalBridge_h

#import <Foundation/Foundation.h>

#ifdef __cplusplus
extern "C" {
#endif

bool metal_vector_add(const float* a, const float* b, float* out, unsigned int count, char* errorMessage, unsigned int errorMessageCapacity);

#ifdef __cplusplus
}
#endif

#endif /* MetalBridge_h */
#endif // disabled
