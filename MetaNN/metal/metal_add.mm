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
            static id<MTLLibrary> lib = nil;
            static dispatch_once_t onceToken;
            dispatch_once(&onceToken, ^
            {
                NSBundle *bundle = [NSBundle mainBundle];
                NSURL *url = [bundle URLForResource:@"MetaNN" withExtension:@"metallib"];
                if (!url)  @throw [NSException exceptionWithName:@"MetalAdd"
                                                   reason:@"MetaNN.metallib not found in main bundle"
                                                 userInfo:nil];
    
                NSError *error = nil;
                if (@available(macOS 13.0, iOS 16.0, *))  lib = [GetDevice() newLibraryWithURL:url error:&error];
                else  lib = [GetDevice() newLibraryWithURL:url error:&error];

                if (!lib)
                {
                    NSString *reason =
                        [NSString stringWithFormat:@"Failed to load MetaNN.metallib: %@",
                                                   error.localizedDescription];
                    @throw [NSException exceptionWithName:@"MetalAdd"
                                                   reason:reason
                                                 userInfo:nil];
                }
            });
            return lib;
        }
    
        id<MTLComputePipelineState> GetPipeline(NSString* fnName)
        {
            static NSMutableDictionary<NSString*, id<MTLComputePipelineState>>* cache = nil;
            static dispatch_once_t onceToken;
            dispatch_once(&onceToken, ^
            {
                cache = [[NSMutableDictionary alloc] init];
            });

            id<MTLComputePipelineState> pso = cache[fnName];
            if (pso)
            {
                return pso;
            }

            NSError* error = nil;
            id<MTLFunction> fn = [GetLibrary() newFunctionWithName:fnName];
            if (!fn)
            {
                @throw [NSException exceptionWithName:@"MetalAdd"
                                               reason:[NSString stringWithFormat:@"Missing Metal function %@", fnName]
                                             userInfo:nil];
            }

            pso = [GetDevice() newComputePipelineStateWithFunction:fn error:&error];
            if (!pso)
            {
                @throw [NSException exceptionWithName:@"MetalAdd"
                                               reason:error.localizedDescription
                                             userInfo:nil];
            }

            cache[fnName] = pso;
            return pso;
        }

        template<typename TElem>
        id<MTLBuffer> BufferOf(const ContinuousMemory<TElem, DeviceTags::Metal>& mem)
        {
            return (__bridge id<MTLBuffer>)mem.NativeHandle();
        }

        void DispatchBinary(NSString* fnName,
                            const ContinuousMemory<float, DeviceTags::Metal>& a,
                            const ContinuousMemory<float, DeviceTags::Metal>& b,
                            ContinuousMemory<float, DeviceTags::Metal>& c,
                            size_t count)
        {
            @autoreleasepool
            {
                id<MTLCommandBuffer> cmd = [GetQueue() commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                id<MTLComputePipelineState> pso = GetPipeline(fnName);

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
                if (w == 0) w = 1;
                MTLSize group = MTLSizeMake(w, 1, 1);

                [enc dispatchThreads:grid threadsPerThreadgroup:group];
                [enc endEncoding];
                [cmd commit];
                [cmd waitUntilCompleted];
            }
        }

        void DispatchUnary(NSString* fnName,
                           const ContinuousMemory<float, DeviceTags::Metal>& a,
                           ContinuousMemory<float, DeviceTags::Metal>& c,
                           size_t count)
        {
            @autoreleasepool
            {
                id<MTLCommandBuffer> cmd = [GetQueue() commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                id<MTLComputePipelineState> pso = GetPipeline(fnName);

                id<MTLBuffer> bufA = BufferOf(a);
                id<MTLBuffer> bufC = BufferOf(c);
                uint n = (uint)count;

                [enc setComputePipelineState:pso];
                [enc setBuffer:bufA offset:0 atIndex:0];
                [enc setBuffer:bufC offset:0 atIndex:1];
                [enc setBytes:&n length:sizeof(n) atIndex:2];

                MTLSize grid = MTLSizeMake(count, 1, 1);
                NSUInteger w = pso.maxTotalThreadsPerThreadgroup;
                if (w > count) w = count;
                if (w == 0) w = 1;
                MTLSize group = MTLSizeMake(w, 1, 1);

                [enc dispatchThreads:grid threadsPerThreadgroup:group];
                [enc endEncoding];
                [cmd commit];
                [cmd waitUntilCompleted];
            }
        }
    }

    void Add(const ContinuousMemory<float, DeviceTags::Metal>& a,
             const ContinuousMemory<float, DeviceTags::Metal>& b,
             ContinuousMemory<float, DeviceTags::Metal>& c,
             size_t count)
    {
        DispatchBinary(@"vector_add_f32", a, b, c, count);
    }

    void Sub(const ContinuousMemory<float, DeviceTags::Metal>& a,
             const ContinuousMemory<float, DeviceTags::Metal>& b,
             ContinuousMemory<float, DeviceTags::Metal>& c,
             size_t count)
    {
        DispatchBinary(@"vector_sub_f32", a, b, c, count);
    }

    void Neg(const ContinuousMemory<float, DeviceTags::Metal>& a,
             ContinuousMemory<float, DeviceTags::Metal>& c,
             size_t count)
    {
        DispatchUnary(@"vector_neg_f32", a, c, count);
    }

    void Tanh(const ContinuousMemory<float, DeviceTags::Metal>& a,
              ContinuousMemory<float, DeviceTags::Metal>& c,
              size_t count)
    {
        DispatchUnary(@"vector_tanh_f32", a, c, count);
    }

    void Sigmoid(const ContinuousMemory<float, DeviceTags::Metal>& a,
                 ContinuousMemory<float, DeviceTags::Metal>& c,
                 size_t count)
    {
        DispatchUnary(@"vector_sigmoid_f32", a, c, count);
    }

    void Exp(const ContinuousMemory<float, DeviceTags::Metal>& a,
             ContinuousMemory<float, DeviceTags::Metal>& c,
             size_t count)
    {
        DispatchUnary(@"vector_exp_f32", a, c, count);
    }
}

