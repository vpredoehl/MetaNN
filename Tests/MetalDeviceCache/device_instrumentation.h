#pragma once

#import <Metal/Metal.h>
#include <atomic>

// Test-only interception: import Metal first so its declaration is untouched.
// Timing builds omit this header and call the real API without instrumentation.
inline std::atomic<size_t> deviceResolutionCount{0};
inline bool simulateUnavailableDevice = false;
inline id<MTLDevice> (*resolveTestDevice)() = nullptr;

inline id<MTLDevice> ResolveCountedDevice()
{
    deviceResolutionCount.fetch_add(1, std::memory_order_relaxed);
    if (simulateUnavailableDevice) return nil;
    if (resolveTestDevice) return resolveTestDevice();
    return MTLCreateSystemDefaultDevice();
}

#define MTLCreateSystemDefaultDevice ResolveCountedDevice
