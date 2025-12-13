// Deprecated: Objective-C++ bridge disabled. Use Swift-only MetalVectorAdder instead.
#if 0
#import <Metal/Metal.h>
#import "MetalBridge.h"

bool metal_vec_add(const float *inA, const float *inB, float *out, size_t count, char *errorMessage, size_t errorMessageCapacity)
{
    if (count == 0) {
        return true;
    }

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        const char *err = "Failed to create system default Metal device.";
        if (errorMessage && errorMessageCapacity > 0) {
            snprintf(errorMessage, errorMessageCapacity, "%s", err);
        }
        return false;
    }

    id<MTLCommandQueue> commandQueue = [device newCommandQueue];
    if (!commandQueue) {
        const char *err = "Failed to create Metal command queue.";
        if (errorMessage && errorMessageCapacity > 0) {
            snprintf(errorMessage, errorMessageCapacity, "%s", err);
        }
        return false;
    }

    NSError *libraryError = nil;
    id<MTLLibrary> defaultLibrary = [device newDefaultLibraryWithBundle:[NSBundle mainBundle] error:&libraryError];
    if (!defaultLibrary) {
        const char *errPrefix = "Failed to load default Metal library: ";
        if (errorMessage && errorMessageCapacity > 0) {
            snprintf(errorMessage, errorMessageCapacity, "%s%.*s", errPrefix, (int)[libraryError.localizedDescription lengthOfBytesUsingEncoding:NSUTF8StringEncoding], [libraryError.localizedDescription UTF8String]);
        }
        return false;
    }

    NSError *pipelineError = nil;
    id<MTLFunction> function = [defaultLibrary newFunctionWithName:@"vec_add"];
    if (!function) {
        const char *err = "Failed to find function 'vec_add' in Metal library.";
        if (errorMessage && errorMessageCapacity > 0) {
            snprintf(errorMessage, errorMessageCapacity, "%s", err);
        }
        return false;
    }

    id<MTLComputePipelineState> pipelineState = [device newComputePipelineStateWithFunction:function error:&pipelineError];
    if (!pipelineState) {
        const char *errPrefix = "Failed to create compute pipeline state: ";
        if (errorMessage && errorMessageCapacity > 0) {
            snprintf(errorMessage, errorMessageCapacity, "%s%.*s", errPrefix, (int)[pipelineError.localizedDescription lengthOfBytesUsingEncoding:NSUTF8StringEncoding], [pipelineError.localizedDescription UTF8String]);
        }
        return false;
    }

    id<MTLBuffer> bufferA = [device newBufferWithLength:sizeof(float)*count options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufferB = [device newBufferWithLength:sizeof(float)*count options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufferOut = [device newBufferWithLength:sizeof(float)*count options:MTLResourceStorageModeShared];
    if (!bufferA || !bufferB || !bufferOut) {
        const char *err = "Failed to allocate Metal buffers.";
        if (errorMessage && errorMessageCapacity > 0) {
            snprintf(errorMessage, errorMessageCapacity, "%s", err);
        }
        return false;
    }

    memcpy(bufferA.contents, inA, sizeof(float)*count);
    memcpy(bufferB.contents, inB, sizeof(float)*count);

    uint32_t count32 = (uint32_t)count;
    id<MTLBuffer> paramsBuffer = [device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
    if (!paramsBuffer) {
        const char *err = "Failed to allocate Metal params buffer.";
        if (errorMessage && errorMessageCapacity > 0) {
            snprintf(errorMessage, errorMessageCapacity, "%s", err);
        }
        return false;
    }
    memcpy(paramsBuffer.contents, &count32, sizeof(uint32_t));

    id<MTLCommandBuffer> cmdBuffer = [commandQueue commandBuffer];
    if (!cmdBuffer) {
        const char *err = "Failed to create Metal command buffer.";
        if (errorMessage && errorMessageCapacity > 0) {
            snprintf(errorMessage, errorMessageCapacity, "%s", err);
        }
        return false;
    }

    id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];
    if (!encoder) {
        const char *err = "Failed to create Metal compute command encoder.";
        if (errorMessage && errorMessageCapacity > 0) {
            snprintf(errorMessage, errorMessageCapacity, "%s", err);
        }
        return false;
    }

    [encoder setComputePipelineState:pipelineState];
    [encoder setBuffer:bufferA offset:0 atIndex:0];
    [encoder setBuffer:bufferB offset:0 atIndex:1];
    [encoder setBuffer:bufferOut offset:0 atIndex:2];
    [encoder setBuffer:paramsBuffer offset:0 atIndex:3];

    NSUInteger threadExecutionWidth = pipelineState.threadExecutionWidth;
    NSUInteger maxThreadsPerThreadgroup = pipelineState.maxTotalThreadsPerThreadgroup;
    NSUInteger threadsPerThreadgroupCount = MAX(threadExecutionWidth, 1);
    if (threadsPerThreadgroupCount > maxThreadsPerThreadgroup) {
        threadsPerThreadgroupCount = maxThreadsPerThreadgroup;
    }
    MTLSize threadsPerThreadgroup = MTLSizeMake(threadsPerThreadgroupCount, 1, 1);

    NSUInteger threadgroupsCount = (count + threadsPerThreadgroupCount - 1) / threadsPerThreadgroupCount;
    MTLSize threadgroups = MTLSizeMake(threadgroupsCount, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerThreadgroup];

    [encoder endEncoding];

    [cmdBuffer commit];
    [cmdBuffer waitUntilCompleted];

    if (cmdBuffer.error) {
        const char *errPrefix = "Metal command buffer error: ";
        NSString *errDesc = cmdBuffer.error.localizedDescription ?: @"Unknown error";
        if (errorMessage && errorMessageCapacity > 0) {
            snprintf(errorMessage, errorMessageCapacity, "%s%.*s", errPrefix, (int)[errDesc lengthOfBytesUsingEncoding:NSUTF8StringEncoding], [errDesc UTF8String]);
        }
        return false;
    }

    memcpy(out, bufferOut.contents, sizeof(float)*count);

    return true;
}
#endif // disabled
