//
//  metal_matmul.mm
//  MetaNNXC
//

#if !__has_feature(objc_arc)
#error "ARC must be enabled for Metal .mm files"
#endif

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <stdexcept>
#include <unordered_map>
#include <mutex>
#include <tuple>
#include <iostream>
#include <cstdlib>
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

    template <typename TMem>
    NSUInteger BufferOffsetBytes(const TMem& mem)
    {
        return static_cast<NSUInteger>(mem.Offset() * sizeof(float));
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
                          NSUInteger offsetBytes,
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

        MPSMatrix* mat = [[MPSMatrix alloc] initWithBuffer:buffer
                                                    offset:offsetBytes
                                                descriptor:desc];
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
        const NSUInteger offGates = BufferOffsetBytes(gates);
        const NSUInteger offPrevCell = BufferOffsetBytes(prevCell);
        const NSUInteger offGateI = BufferOffsetBytes(gateI);
        const NSUInteger offGateF = BufferOffsetBytes(gateF);
        const NSUInteger offGateG = BufferOffsetBytes(gateG);
        const NSUInteger offGateO = BufferOffsetBytes(gateO);
        const NSUInteger offCellOut = BufferOffsetBytes(cellOut);
        const NSUInteger offHiddenOut = BufferOffsetBytes(hiddenOut);
        
        static bool s_printedGateStateBufferDiag = false;
        if (!s_printedGateStateBufferDiag)
        {
            s_printedGateStateBufferDiag = true;
            std::cout
                << "DIAG_GATESTATE_BUFFERS"
                << ",gates=" << (__bridge const void*)bufGates
                << ",prevCell=" << (__bridge const void*)bufPrevCell
                << ",gateI=" << (__bridge const void*)bufGateI
                << ",gateF=" << (__bridge const void*)bufGateF
                << ",gateG=" << (__bridge const void*)bufGateG
                << ",gateO=" << (__bridge const void*)bufGateO
                << ",cellOut=" << (__bridge const void*)bufCellOut
                << ",hiddenOut=" << (__bridge const void*)bufHiddenOut
                << ",batchSize=" << batchSize
                << ",hiddenSize=" << hiddenSize
                << "\n";
        }
        static bool s_printedGateStateHostProbe = false;
        if (!s_printedGateStateHostProbe)
        {
            s_printedGateStateHostProbe = true;
            const float* gatesHost = static_cast<const float*>([bufGates contents]);
            std::cout << "DIAG_GATESTATE_GATES_HOST";
            if (gatesHost)
            {
                const size_t probeCount = std::min<size_t>(32, batchSize * 4 * hiddenSize);
                for (size_t i = 0; i < probeCount; ++i)
                    std::cout << ",g" << i << "=" << gatesHost[i];
            }
            else
            {
                std::cout << ",contents=null";
            }
            std::cout << "\n";
        }
        if (!bufGates || !bufPrevCell || !bufGateI || !bufGateF || !bufGateG || !bufGateO || !bufCellOut || !bufHiddenOut)
        {
            @throw [NSException exceptionWithName:@"MetalAdd"
                                           reason:@"One or more Metal buffers are null in GateStateFused"
                                         userInfo:nil];
        }
        
        auto abortAlias = [&](const char* tag, id<MTLBuffer> a, id<MTLBuffer> b)
        {
            if (a == b)
            {
                std::cout
                    << "DIAG_ABORT_GATESTATE_BUFFER_ALIAS"
                    << ",tag=" << tag
                    << ",a=" << (__bridge const void*)a
                    << ",b=" << (__bridge const void*)b
                    << "\n";
                std::abort();
            }
        };

        abortAlias("gateI_vs_gates", bufGateI, bufGates);
        abortAlias("gateF_vs_gates", bufGateF, bufGates);
        abortAlias("gateG_vs_gates", bufGateG, bufGates);
        abortAlias("gateO_vs_gates", bufGateO, bufGates);
        abortAlias("cellOut_vs_gates", bufCellOut, bufGates);
        abortAlias("hiddenOut_vs_gates", bufHiddenOut, bufGates);

        abortAlias("gateI_vs_prevCell", bufGateI, bufPrevCell);
        abortAlias("gateF_vs_prevCell", bufGateF, bufPrevCell);
        abortAlias("gateG_vs_prevCell", bufGateG, bufPrevCell);
        abortAlias("gateO_vs_prevCell", bufGateO, bufPrevCell);
        abortAlias("cellOut_vs_prevCell", bufCellOut, bufPrevCell);
        abortAlias("hiddenOut_vs_prevCell", bufHiddenOut, bufPrevCell);

        abortAlias("gateI_vs_gateF", bufGateI, bufGateF);
        abortAlias("gateI_vs_gateG", bufGateI, bufGateG);
        abortAlias("gateI_vs_gateO", bufGateI, bufGateO);
        abortAlias("gateI_vs_cellOut", bufGateI, bufCellOut);
        abortAlias("gateI_vs_hiddenOut", bufGateI, bufHiddenOut);
        abortAlias("gateF_vs_gateG", bufGateF, bufGateG);
        abortAlias("gateF_vs_gateO", bufGateF, bufGateO);
        abortAlias("gateF_vs_cellOut", bufGateF, bufCellOut);
        abortAlias("gateF_vs_hiddenOut", bufGateF, bufHiddenOut);
        abortAlias("gateG_vs_gateO", bufGateG, bufGateO);
        abortAlias("gateG_vs_cellOut", bufGateG, bufCellOut);
        abortAlias("gateG_vs_hiddenOut", bufGateG, bufHiddenOut);
        abortAlias("gateO_vs_cellOut", bufGateO, bufCellOut);
        abortAlias("gateO_vs_hiddenOut", bufGateO, bufHiddenOut);
        abortAlias("cellOut_vs_hiddenOut", bufCellOut, bufHiddenOut);

        const NSUInteger gatesBytesNeeded = static_cast<NSUInteger>(batchSize * 4 * hiddenSize * sizeof(float));
        const NSUInteger stateBytesNeeded = static_cast<NSUInteger>(batchSize * hiddenSize * sizeof(float));

        auto abortIfTooSmall = [&](const char* tag, id<MTLBuffer> buf, NSUInteger need)
        {
            const NSUInteger have = [buf length];
            if (have < need)
            {
                std::cout
                    << "DIAG_ABORT_GATESTATE_BUFFER_TOO_SMALL"
                    << ",tag=" << tag
                    << ",have=" << static_cast<unsigned long long>(have)
                    << ",need=" << static_cast<unsigned long long>(need)
                    << ",batchSize=" << batchSize
                    << ",hiddenSize=" << hiddenSize
                    << "\n";
                std::abort();
            }
        };

        abortIfTooSmall("gates", bufGates, gatesBytesNeeded);
        abortIfTooSmall("prevCell", bufPrevCell, stateBytesNeeded);
        abortIfTooSmall("gateI", bufGateI, stateBytesNeeded);
        abortIfTooSmall("gateF", bufGateF, stateBytesNeeded);
        abortIfTooSmall("gateG", bufGateG, stateBytesNeeded);
        abortIfTooSmall("gateO", bufGateO, stateBytesNeeded);
        abortIfTooSmall("cellOut", bufCellOut, stateBytesNeeded);
        abortIfTooSmall("hiddenOut", bufHiddenOut, stateBytesNeeded);

        [encoder setBuffer:bufGates offset:offGates atIndex:0];
        [encoder setBuffer:bufPrevCell offset:offPrevCell atIndex:1];
        [encoder setBuffer:bufGateI offset:offGateI atIndex:2];
        [encoder setBuffer:bufGateF offset:offGateF atIndex:3];
        [encoder setBuffer:bufGateG offset:offGateG atIndex:4];
        [encoder setBuffer:bufGateO offset:offGateO atIndex:5];
        [encoder setBuffer:bufCellOut offset:offCellOut atIndex:6];
        [encoder setBuffer:bufHiddenOut offset:offHiddenOut atIndex:7];

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

        // This helper is synchronous for now so callers can safely read outputs immediately.
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
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
        const NSUInteger offC = BufferOffsetBytes(c);
        const NSUInteger offBias = BufferOffsetBytes(bias);

        uint mm = (uint)m;
        uint nn = (uint)n;

        [enc setComputePipelineState:pso];
        [enc setBuffer:bufC offset:offC atIndex:0];
        [enc setBuffer:bufBias offset:offBias atIndex:1];
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
        const NSUInteger offA = BufferOffsetBytes(a);
        const NSUInteger offB = BufferOffsetBytes(b);
        const NSUInteger offC = BufferOffsetBytes(c);

        if (!bufA || !bufB || !bufC)
        {
            throw std::runtime_error("MetalMatMul: one or more MTLBuffer handles are null");
        }

        // Current MetaNN buffers are logically contiguous row-major matrices.
        // If you later adopt padded row strides, pass those actual rowBytes values here.
        const size_t rowBytesA = k * sizeof(float);
        const size_t rowBytesB = n * sizeof(float);
        const size_t rowBytesC = n * sizeof(float);

        MPSMatrix* matA = MakeMatrix(bufA, offA, m, k, rowBytesA);
        MPSMatrix* matB = MakeMatrix(bufB, offB, k, n, rowBytesB);
        MPSMatrix* matC = MakeMatrix(bufC, offC, m, n, rowBytesC);

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
