#import <Metal/Metal.h>
#import "CppGpuBridge.h"

int cpp_gpu_vector_add(const float *a, const float *b, float *out, size_t count) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) return 1;
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) return 2;

        NSError *error = nil;
        id<MTLLibrary> library = [device newDefaultLibrary];
        if (!library) return 3;
        id<MTLFunction> fn = [library newFunctionWithName:@"vector_add"];
        if (!fn) return 4;

        id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&error];
        if (!pso) return 5;

        // Create buffers
        id<MTLBuffer> aBuf = [device newBufferWithBytes:a length:sizeof(float) * count options:MTLResourceStorageModeShared];
        id<MTLBuffer> bBuf = [device newBufferWithBytes:b length:sizeof(float) * count options:MTLResourceStorageModeShared];
        id<MTLBuffer> oBuf = [device newBufferWithLength:sizeof(float) * count options:MTLResourceStorageModeShared];
        if (!aBuf || !bBuf || !oBuf) return 6;

        uint32_t uCount = (uint32_t)count;

        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        if (!cmd) return 7;
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!enc) return 8;

        [enc setComputePipelineState:pso];
        [enc setBuffer:aBuf offset:0 atIndex:0];
        [enc setBuffer:bBuf offset:0 atIndex:1];
        [enc setBuffer:oBuf offset:0 atIndex:2];
        [enc setBytes:&uCount length:sizeof(uint32_t) atIndex:3];

        MTLSize tgSize = MTLSizeMake(pso.threadExecutionWidth, 1, 1);
        MTLSize grid = MTLSizeMake((NSUInteger)count, 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tgSize];
        [enc endEncoding];

        [cmd commit];
        [cmd waitUntilCompleted];

        // Copy back results
        memcpy(out, [oBuf contents], sizeof(float) * count);
        return 0;
    }
}
