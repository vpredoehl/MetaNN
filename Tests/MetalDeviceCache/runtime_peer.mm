#include "device_instrumentation.h"
// Exercise the actual private runtime accessors in another translation unit.
#include "../../MetaNN/metal/metal_add.mm"
#include "../../MetaNN/metal/metal_matmul.mm"

void CheckRuntimeDevices(id<MTLDevice> expected)
{
    assert(MetaNN::NSMetalAdd::GetDevice() == expected);
    assert(MetaNN::NSMetalMatMul::GetDevice() == expected);
    assert(MetaNN::NSMetal::DefaultDevice() == expected);
}
