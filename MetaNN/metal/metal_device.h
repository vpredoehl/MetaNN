#pragma once

#if !__has_feature(objc_arc)
#error "ARC must be enabled for Metal .mm files"
#endif

#import <Metal/Metal.h>

namespace MetaNN::NSMetal
{
    // An inline function's local static is shared across translation units and
    // element types. C++ initializes it once, safely even on concurrent first use;
    // ARC's strong static keeps the device alive independently of buffers/pools.
    // Return the owning slot by reference so allocation callers can borrow it
    // without an additional retain/autorelease on every pool hit.
    inline id<MTLDevice> const& DefaultDevice()
    {
        static id<MTLDevice> const device = MTLCreateSystemDefaultDevice();
        return device;
    }
}
