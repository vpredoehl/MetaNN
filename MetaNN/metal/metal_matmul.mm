//
//  metal_matmul.mm
//  MetaNNXC
//
//  Created by Vincent Predoehl on 3/8/26.
//

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdexcept>
#include "metal_matmul.h"

namespace MetaNN::NSMetalMatMul
{
namespace
{
    id<MTLDevice> GetDevice()
    {
        static id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        return device;
    }

    id<MTLCommandQueue> GetQueue()
    {
        static id<MTLCommandQueue> queue = [GetDevice() newCommandQueue];
        return queue;
    }

    id<MTLLibrary> GetLibrary()
    {
        static id<MTLLibrary> lib = [GetDevice() newDefaultLibrary];
        return lib;
    }

    id<MTLComputePipelineState> GetPipeline()
    {
        static id<MTLComputePipelineState> pso = nil;
        static dispatch_once_t onceToken;
        dispatch_once(&onceToken, ^{
            NSError* error = nil;
            id<MTLFunction> fn = [GetLibrary() newFunctionWithName:@"matrix_mul_f32"];
            pso = [GetDevice() newComputePipelineStateWithFunction:fn error:&error];
            if (!pso)
            {
                @throw [NSException exceptionWithName:@"MetalMatMul"
                                               reason:error.localizedDescription
                                             userInfo:nil];
            }
        });
        return pso;
    }

    template <typename TMem>
    id<MTLBuffer> BufferOf(const TMem& mem)
    {
        return (__bridge id<MTLBuffer>)mem.NativeHandle();
    }
}

void MatMul(const ContinuousMemory<float, DeviceTags::Metal>& a,
            const ContinuousMemory<float, DeviceTags::Metal>& b,
            ContinuousMemory<float, DeviceTags::Metal>& c,
            size_t m, size_t k, size_t n)
{
    @autoreleasepool
    {
        if (m == 0 || k == 0 || n == 0)
        {
            return;
        }

        id<MTLCommandBuffer> cmd = [GetQueue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        id<MTLComputePipelineState> pso = GetPipeline();

        id<MTLBuffer> bufA = BufferOf(a);
        id<MTLBuffer> bufB = BufferOf(b);
        id<MTLBuffer> bufC = BufferOf(c);

        uint mm = (uint)m;
        uint kk = (uint)k;
        uint nn = (uint)n;

        [enc setComputePipelineState:pso];
        [enc setBuffer:bufA offset:0 atIndex:0];
        [enc setBuffer:bufB offset:0 atIndex:1];
        [enc setBuffer:bufC offset:0 atIndex:2];
        [enc setBytes:&mm length:sizeof(mm) atIndex:3];
        [enc setBytes:&kk length:sizeof(kk) atIndex:4];
        [enc setBytes:&nn length:sizeof(nn) atIndex:5];

        MTLSize grid = MTLSizeMake(n, m, 1);

        NSUInteger tw = pso.threadExecutionWidth;
        NSUInteger th = pso.maxTotalThreadsPerThreadgroup / tw;
        if (th == 0) th = 1;

        MTLSize group = MTLSizeMake(tw, th, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:group];
        [enc endEncoding];

        [cmd commit];
        [cmd waitUntilCompleted];
    }
}
}
