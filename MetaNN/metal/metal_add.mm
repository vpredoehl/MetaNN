//
//  metal_add.mm
//  MetaNNXC
//
//  Created by Vincent Predoehl on 3/8/26.
//
#if !__has_feature(objc_arc)
#error "ARC must be enabled for Metal .mm files"
#endif

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdexcept>
#include "metal_add.h"
#include "metal_view_binding.h"
#if defined(METANN_MINIMAL_DIAGNOSTIC)
#include <iostream>
#endif

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

        void BindOperand(id<MTLComputeCommandEncoder> encoder,
                         const ContinuousMemory<float, DeviceTags::Metal>& view,
                         size_t count, NSUInteger slot)
        {
            id<MTLBuffer> buffer = BufferOf(view);
            if (!buffer)
                throw std::invalid_argument("MetalAdd operand has no buffer");

            Detail::BindView<float>(view, buffer.length, count, slot,
                [&](size_t offsetBytes, size_t index)
                {
#if defined(METANN_MINIMAL_DIAGNOSTIC)
                    std::cout << "ADD_BIND slot=" << index
                              << " element_offset=" << view.Offset()
                              << " byte_offset=" << offsetBytes
                              << " view_elements=" << view.Size()
                              << " buffer_bytes=" << buffer.length
                              << " count=" << count << std::endl;
#endif
                    [encoder setBuffer:buffer offset:offsetBytes atIndex:index];
                });
        }

#if defined(METANN_MINIMAL_DIAGNOSTIC)
        void ObserveCompletion(id<MTLCommandBuffer> command)
        {
            NSError* error = command.error;
            std::cout << "ADD_COMMAND status=" << command.status
                      << " completed=" << (command.status == MTLCommandBufferStatusCompleted)
                      << " error_domain=" << (error.domain.UTF8String ?: "none")
                      << " error_code=" << error.code << std::endl;
            if (error)
                std::cerr << "ADD_METAL_ERROR "
                          << (error.localizedDescription.UTF8String ?: "unavailable") << std::endl;
            if (command.status != MTLCommandBufferStatusCompleted || error)
                throw std::runtime_error("MetalAdd command did not complete successfully");
        }
#endif

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

                uint n = (uint)count;

                [enc setComputePipelineState:pso];
                BindOperand(enc, a, count, 0);
                BindOperand(enc, b, count, 1);
                BindOperand(enc, c, count, 2);
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
#if defined(METANN_MINIMAL_DIAGNOSTIC)
                ObserveCompletion(cmd);
#endif
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

                uint n = (uint)count;

                [enc setComputePipelineState:pso];
                BindOperand(enc, a, count, 0);
                BindOperand(enc, c, count, 1);
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
#if defined(METANN_MINIMAL_DIAGNOSTIC)
                ObserveCompletion(cmd);
#endif
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

