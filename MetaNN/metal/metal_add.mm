//
//  metal_add.mm
//  MetaNNXC
//
//  Created by Vincent Predoehl on 3/8/26.
//

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdexcept>
#include "metal_add.h"

namespace MetaNN::NSMetalAdd
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
            static id<MTLLibrary> lib =
                [GetDevice() newDefaultLibrary];
            return lib;
        }

        id<MTLComputePipelineState> GetPipeline()
        {
            static id<MTLComputePipelineState> pso = nil;
            static dispatch_once_t onceToken;
            dispatch_once(&onceToken, ^
            {
                NSError* error = nil;
                id<MTLFunction> fn = [GetLibrary() newFunctionWithName:@"vector_add_f32"];
                pso = [GetDevice() newComputePipelineStateWithFunction:fn error:&error];
                if (!pso)
                {
                    @throw [NSException exceptionWithName:@"MetalAdd"
                                                   reason:error.localizedDescription
                                                 userInfo:nil];
                }
            });
            return pso;
        }

        template<typename TElem>
        id<MTLBuffer> BufferOf(const ContinuousMemory<TElem, DeviceTags::Metal>& mem)
        {
            return (__bridge id<MTLBuffer>)mem.NativeHandle();
        }
    }

    void Add(const ContinuousMemory<float, DeviceTags::Metal>& a,
             const ContinuousMemory<float, DeviceTags::Metal>& b,
             ContinuousMemory<float, DeviceTags::Metal>& c,
             size_t count)
    {
        @autoreleasepool
        {
            id<MTLCommandBuffer> cmd = [GetQueue() commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            id<MTLComputePipelineState> pso = GetPipeline();

            id<MTLBuffer> bufA = BufferOf(a);
            id<MTLBuffer> bufB = BufferOf(b);
            id<MTLBuffer> bufC = BufferOf(c);

            uint n = (uint)count;

            [enc setComputePipelineState:pso];
            [enc setBuffer:bufA offset:0 atIndex:0];
            [enc setBuffer:bufB offset:0 atIndex:1];
            [enc setBuffer:bufC offset:0 atIndex:2];
            [enc setBytes:&n length:sizeof(n) atIndex:3];

            MTLSize grid = MTLSizeMake(count, 1, 1);
            NSUInteger w = pso.maxTotalThreadsPerThreadgroup;
            if (w > count) w = count;
            MTLSize group = MTLSizeMake(w, 1, 1);

            [enc dispatchThreads:grid
          threadsPerThreadgroup:group];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
    }
}
