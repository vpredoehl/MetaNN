//
//  metal_matmul.mm
//  MetaNNXC
//

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <stdexcept>
#include <unordered_map>
#include <mutex>
#include <tuple>

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
    id<MTLLibrary> GetLibrary()
    {
        static id<MTLLibrary> lib = [GetDevice() newDefaultLibrary];
        return lib;
    }
    id<MTLCommandQueue> GetQueue()
    {
        static id<MTLCommandQueue> queue = [GetDevice() newCommandQueue];
        return queue;
    }

    template <typename TMem>
    id<MTLBuffer> BufferOf(const TMem& mem)
    {
        return (__bridge id<MTLBuffer>)mem.NativeHandle();
    }

    struct KernelKey
    {
        size_t m;
        size_t k;
        size_t n;

        bool operator==(const KernelKey& other) const noexcept
        {
            return m == other.m && k == other.k && n == other.n;
        }
    };

    struct KernelKeyHash
    {
        size_t operator()(const KernelKey& x) const noexcept
        {
            size_t h = std::hash<size_t>{}(x.m);
            h ^= std::hash<size_t>{}(x.k) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            h ^= std::hash<size_t>{}(x.n) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            return h;
        }
    };

    MPSMatrixMultiplication* GetKernel(size_t m, size_t k, size_t n)
    {
        static std::unordered_map<KernelKey, MPSMatrixMultiplication*, KernelKeyHash> cache;
        static std::mutex cacheMutex;

        const KernelKey key{m, k, n};

        {
            std::lock_guard<std::mutex> lock(cacheMutex);
            auto it = cache.find(key);
            if (it != cache.end())
            {
                return it->second;
            }
        }

        id<MTLDevice> device = GetDevice();
        if (!device)
        {
            throw std::runtime_error("MetalMatMul: failed to create MTLDevice");
        }

        MPSMatrixMultiplication* kernel =
            [[MPSMatrixMultiplication alloc] initWithDevice:device
                                             transposeLeft:NO
                                            transposeRight:NO
                                                resultRows:m
                                             resultColumns:n
                                           interiorColumns:k
                                                     alpha:1.0
                                                      beta:0.0];

        if (!kernel)
        {
            throw std::runtime_error("MetalMatMul: failed to create MPSMatrixMultiplication");
        }

        {
            std::lock_guard<std::mutex> lock(cacheMutex);
            auto [it, inserted] = cache.emplace(key, kernel);
            if (!inserted)
            {
                kernel = it->second;
            }
        }

        return kernel;
    }

    MPSMatrix* MakeMatrix(id<MTLBuffer> buffer,
                          size_t rows,
                          size_t cols,
                          size_t rowBytes)
    {
        MPSMatrixDescriptor* desc =
            [MPSMatrixDescriptor matrixDescriptorWithRows:rows
                                                 columns:cols
                                                rowBytes:rowBytes
                                                dataType:MPSDataTypeFloat32];

        if (!desc)
        {
            throw std::runtime_error("MetalMatMul: failed to create MPSMatrixDescriptor");
        }

        MPSMatrix* mat = [[MPSMatrix alloc] initWithBuffer:buffer descriptor:desc];
        if (!mat)
        {
            throw std::runtime_error("MetalMatMul: failed to create MPSMatrix");
        }

        return mat;
    }
}

id<MTLComputePipelineState> GetBiasPipeline()
{
    static id<MTLComputePipelineState> pso = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        NSError* error = nil;
        id<MTLFunction> fn = [GetLibrary() newFunctionWithName:@"add_row_bias_f32"];
        pso = [GetDevice() newComputePipelineStateWithFunction:fn error:&error];
        if (!pso)
        {
            @throw [NSException exceptionWithName:@"MetalMatMulBias"
                                           reason:error.localizedDescription
                                         userInfo:nil];
        }
    });
    return pso;
}

id<MTLComputePipelineState> GetGateStateFusedPipeline()
{
    static id<MTLComputePipelineState> pipeline = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        id<MTLDevice> device = GetDevice();
        if (device == nil)
        {
            @throw [NSException exceptionWithName:@"MetalAdd"
                                           reason:@"GetDevice returned nil in GateStateFused"
                                         userInfo:nil];
        }

        NSError* error = nil;
        id<MTLLibrary> library = GetLibrary();
        if (library == nil)
        {
            @throw [NSException exceptionWithName:@"MetalAdd"
                                           reason:[NSString stringWithFormat:@"Failed to load default Metal library in GateStateFused: %@", error]
                                         userInfo:nil];
        }

        id<MTLFunction> fn = [library newFunctionWithName:@"gate_state_fused_f32"];
        if (fn == nil)
        {
            @throw [NSException exceptionWithName:@"MetalAdd"
                                           reason:@"Missing Metal function gate_state_fused_f32"
                                         userInfo:nil];
        }

        pipeline = [device newComputePipelineStateWithFunction:fn error:&error];
        if (pipeline == nil)
        {
            @throw [NSException exceptionWithName:@"MetalAdd"
                                           reason:[NSString stringWithFormat:@"Failed to create GateStateFused pipeline: %@", error]
                                         userInfo:nil];
        }
    });
    return pipeline;
}

void GateStateFused(const ContinuousMemory<float, DeviceTags::Metal>& gates,
                    const ContinuousMemory<float, DeviceTags::Metal>& prevCell,
                    ContinuousMemory<float, DeviceTags::Metal>& gateI,
                    ContinuousMemory<float, DeviceTags::Metal>& gateF,
                    ContinuousMemory<float, DeviceTags::Metal>& gateG,
                    ContinuousMemory<float, DeviceTags::Metal>& gateO,
                    ContinuousMemory<float, DeviceTags::Metal>& cellOut,
                    ContinuousMemory<float, DeviceTags::Metal>& hiddenOut,
                    size_t batchSize,
                    size_t hiddenSize)
{
    @autoreleasepool
    {
        if (batchSize == 0 || hiddenSize == 0)
        {
            return;
        }

        id<MTLDevice> device = GetDevice();
        if (device == nil)
        {
            @throw [NSException exceptionWithName:@"MetalAdd"
                                           reason:@"GetDevice returned nil in GateStateFused"
                                         userInfo:nil];
        }

        id<MTLCommandQueue> queue = GetQueue();
        if (queue == nil)
        {
            @throw [NSException exceptionWithName:@"MetalAdd"
                                           reason:@"Failed to create Metal command queue in GateStateFused"
                                         userInfo:nil];
        }

        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        if (commandBuffer == nil)
        {
            @throw [NSException exceptionWithName:@"MetalAdd"
                                           reason:@"Failed to create Metal command buffer in GateStateFused"
                                         userInfo:nil];
        }

        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        if (encoder == nil)
        {
            @throw [NSException exceptionWithName:@"MetalAdd"
                                           reason:@"Failed to create Metal compute encoder in GateStateFused"
                                         userInfo:nil];
        }

        id<MTLComputePipelineState> pipeline = GetGateStateFusedPipeline();
        [encoder setComputePipelineState:pipeline];

        // gates is a row-major [batchSize, 4 * hiddenSize] tensor in CPU order:
        // I = [0:H), F = [H:2H), G = [2H:3H), O = [3H:4H)
        id<MTLBuffer> bufGates = BufferOf(gates);
        id<MTLBuffer> bufPrevCell = BufferOf(prevCell);
        id<MTLBuffer> bufGateI = BufferOf(gateI);
        id<MTLBuffer> bufGateF = BufferOf(gateF);
        id<MTLBuffer> bufGateG = BufferOf(gateG);
        id<MTLBuffer> bufGateO = BufferOf(gateO);
        id<MTLBuffer> bufCellOut = BufferOf(cellOut);
        id<MTLBuffer> bufHiddenOut = BufferOf(hiddenOut);

        if (!bufGates || !bufPrevCell || !bufGateI || !bufGateF || !bufGateG || !bufGateO || !bufCellOut || !bufHiddenOut)
        {
            @throw [NSException exceptionWithName:@"MetalAdd"
                                           reason:@"One or more Metal buffers are null in GateStateFused"
                                         userInfo:nil];
        }

        [encoder setBuffer:bufGates offset:0 atIndex:0];
        [encoder setBuffer:bufPrevCell offset:0 atIndex:1];
        [encoder setBuffer:bufGateI offset:0 atIndex:2];
        [encoder setBuffer:bufGateF offset:0 atIndex:3];
        [encoder setBuffer:bufGateG offset:0 atIndex:4];
        [encoder setBuffer:bufGateO offset:0 atIndex:5];
        [encoder setBuffer:bufCellOut offset:0 atIndex:6];
        [encoder setBuffer:bufHiddenOut offset:0 atIndex:7];

        uint32_t batchCount = static_cast<uint32_t>(batchSize);
        uint32_t hiddenCount = static_cast<uint32_t>(hiddenSize);
        [encoder setBytes:&batchCount length:sizeof(batchCount) atIndex:8];
        [encoder setBytes:&hiddenCount length:sizeof(hiddenCount) atIndex:9];

        const NSUInteger total = static_cast<NSUInteger>(batchSize * hiddenSize);
        const NSUInteger w = pipeline.threadExecutionWidth;
        const NSUInteger tgSize = (w == 0) ? 64 : w;

        [encoder dispatchThreads:MTLSizeMake(total, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
        [encoder endEncoding];

        // ✅ ASYNC: do NOT wait here
        [commandBuffer commit];
    }
}

void MatMulBias(const ContinuousMemory<float, DeviceTags::Metal>& a,
                const ContinuousMemory<float, DeviceTags::Metal>& b,
                const ContinuousMemory<float, DeviceTags::Metal>& bias,
                ContinuousMemory<float, DeviceTags::Metal>& c,
                size_t m, size_t k, size_t n)
{
    @autoreleasepool
    {
        if (m == 0 || k == 0 || n == 0)
        {
            return;
        }

        // First do the MPS GEMM into c
        MatMul(a, b, c, m, k, n);

        // Then add bias row-wise to c without materializing (m x n)
        id<MTLCommandBuffer> cmd = [GetQueue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        id<MTLComputePipelineState> pso = GetBiasPipeline();

        id<MTLBuffer> bufC = BufferOf(c);
        id<MTLBuffer> bufBias = BufferOf(bias);

        uint mm = (uint)m;
        uint nn = (uint)n;

        [enc setComputePipelineState:pso];
        [enc setBuffer:bufC offset:0 atIndex:0];
        [enc setBuffer:bufBias offset:0 atIndex:1];
        [enc setBytes:&mm length:sizeof(mm) atIndex:2];
        [enc setBytes:&nn length:sizeof(nn) atIndex:3];

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

        id<MTLDevice> device = GetDevice();
        if (!device)
        {
            throw std::runtime_error("MetalMatMul: no Metal device");
        }

        if (!MPSSupportsMTLDevice(device))
        {
            throw std::runtime_error("MetalMatMul: MPS does not support this Metal device");
        }

        id<MTLCommandQueue> queue = GetQueue();
        if (!queue)
        {
            throw std::runtime_error("MetalMatMul: failed to create command queue");
        }

        id<MTLBuffer> bufA = BufferOf(a);
        id<MTLBuffer> bufB = BufferOf(b);
        id<MTLBuffer> bufC = BufferOf(c);

        if (!bufA || !bufB || !bufC)
        {
            throw std::runtime_error("MetalMatMul: one or more MTLBuffer handles are null");
        }

        // Current MetaNN buffers are logically contiguous row-major matrices.
        // If you later adopt padded row strides, pass those actual rowBytes values here.
        const size_t rowBytesA = k * sizeof(float);
        const size_t rowBytesB = n * sizeof(float);
        const size_t rowBytesC = n * sizeof(float);

        MPSMatrix* matA = MakeMatrix(bufA, m, k, rowBytesA);
        MPSMatrix* matB = MakeMatrix(bufB, k, n, rowBytesB);
        MPSMatrix* matC = MakeMatrix(bufC, m, n, rowBytesC);

        MPSMatrixMultiplication* kernel = GetKernel(m, k, n);

        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        if (!cmd)
        {
            throw std::runtime_error("MetalMatMul: failed to create command buffer");
        }

        [kernel encodeToCommandBuffer:cmd
                           leftMatrix:matA
                          rightMatrix:matB
                         resultMatrix:matC];

        [cmd commit];
        [cmd waitUntilCompleted];

        if (cmd.status == MTLCommandBufferStatusError)
        {
            NSString* errStr = cmd.error ? cmd.error.localizedDescription : @"unknown command buffer error";
            throw std::runtime_error(std::string("MetalMatMul: ") + [errStr UTF8String]);
        }
    }
}

    void WaitForAll()
    {
        @autoreleasepool
        {
            id<MTLCommandQueue> queue = GetQueue();
            if (!queue)
            {
                throw std::runtime_error("MetalMatMul: failed to create command queue");
            }

            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            if (!cmd)
            {
                throw std::runtime_error("MetalMatMul: failed to create command buffer");
            }

            [cmd commit];
            [cmd waitUntilCompleted];
        }
    }
} // namespace MetaNN::NSMetalMatMul
