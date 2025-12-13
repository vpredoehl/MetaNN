import Foundation
import Metal

/// Errors that can occur during Metal vector addition.
enum MetalVectorAddError: Error, LocalizedError {
    case deviceUnavailable
    case commandQueueCreationFailed
    case defaultLibraryUnavailable(String)
    case functionNotFound(String)
    case pipelineCreationFailed(String)
    case bufferAllocationFailed
    case commandBufferCreationFailed
    case encoderCreationFailed
    case commandBufferError(String)
    case mismatchedInputLengths

    var errorDescription: String? {
        switch self {
        case .deviceUnavailable:
            return "Failed to create system default Metal device."
        case .commandQueueCreationFailed:
            return "Failed to create Metal command queue."
        case .defaultLibraryUnavailable(let reason):
            return "Failed to load default Metal library: \(reason)"
        case .functionNotFound(let name):
            return "Failed to find function '\(name)' in Metal library."
        case .pipelineCreationFailed(let reason):
            return "Failed to create compute pipeline state: \(reason)"
        case .bufferAllocationFailed:
            return "Failed to allocate Metal buffers."
        case .commandBufferCreationFailed:
            return "Failed to create Metal command buffer."
        case .encoderCreationFailed:
            return "Failed to create Metal compute command encoder."
        case .commandBufferError(let reason):
            return "Metal command buffer error: \(reason)"
        case .mismatchedInputLengths:
            return "Input arrays must have the same length."
        }
    }
}

/// A Swift-only Metal helper to perform vector addition on the GPU.
struct MetalVectorAdder {
    /// Adds two arrays of `Float` element-wise using Metal and returns the result.
    /// - Parameters:
    ///   - a: First input array.
    ///   - b: Second input array. Must be the same length as `a`.
    /// - Returns: A new array where `result[i] = a[i] + b[i]`.
    /// - Throws: `MetalVectorAddError` if Metal setup or execution fails.
    static func add(a: [Float], b: [Float]) throws -> [Float] {
        guard a.count == b.count else { throw MetalVectorAddError.mismatchedInputLengths }
        let count = a.count
        if count == 0 { return [] }

        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalVectorAddError.deviceUnavailable
        }
        guard let commandQueue = device.makeCommandQueue() else {
            throw MetalVectorAddError.commandQueueCreationFailed
        }

        // Metal shader source code for vector addition
        let shaderSource = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void vec_add(
            device const float* inA [[ buffer(0) ]],
            device const float* inB [[ buffer(1) ]],
            device float* out [[ buffer(2) ]],
            constant uint& count [[ buffer(3) ]],
            uint id [[ thread_position_in_grid ]]) {
            if (id >= count) return;
            out[id] = inA[id] + inB[id];
        }
        """

        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: shaderSource, options: nil)
        } catch {
            throw MetalVectorAddError.defaultLibraryUnavailable(String(describing: error))
        }

        guard let function = library.makeFunction(name: "vec_add") else {
            throw MetalVectorAddError.functionNotFound("vec_add")
        }

        let pipelineState: MTLComputePipelineState
        do {
            pipelineState = try device.makeComputePipelineState(function: function)
        } catch {
            throw MetalVectorAddError.pipelineCreationFailed(String(describing: error))
        }

        // Create buffers
        let byteCount = MemoryLayout<Float>.stride * count
        guard let bufferA = device.makeBuffer(length: byteCount, options: .storageModeShared),
              let bufferB = device.makeBuffer(length: byteCount, options: .storageModeShared),
              let bufferOut = device.makeBuffer(length: byteCount, options: .storageModeShared),
              let paramsBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared) else {
            throw MetalVectorAddError.bufferAllocationFailed
        }

        // Copy data
        bufferA.contents().copyMemory(from: a, byteCount: byteCount)
        bufferB.contents().copyMemory(from: b, byteCount: byteCount)
        var count32 = UInt32(count)
        paramsBuffer.contents().copyMemory(from: &count32, byteCount: MemoryLayout<UInt32>.stride)

        guard let cmdBuffer = commandQueue.makeCommandBuffer() else {
            throw MetalVectorAddError.commandBufferCreationFailed
        }
        guard let encoder = cmdBuffer.makeComputeCommandEncoder() else {
            throw MetalVectorAddError.encoderCreationFailed
        }

        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(bufferA, offset: 0, index: 0)
        encoder.setBuffer(bufferB, offset: 0, index: 1)
        encoder.setBuffer(bufferOut, offset: 0, index: 2)
        encoder.setBuffer(paramsBuffer, offset: 0, index: 3)

        let threadExecutionWidth = pipelineState.threadExecutionWidth
        let maxTotalThreadsPerThreadgroup = pipelineState.maxTotalThreadsPerThreadgroup
        var threadsPerThreadgroupCount = threadExecutionWidth
        if threadsPerThreadgroupCount > maxTotalThreadsPerThreadgroup {
            threadsPerThreadgroupCount = maxTotalThreadsPerThreadgroup
        }
        let threadsPerThreadgroup = MTLSize(width: threadsPerThreadgroupCount, height: 1, depth: 1)

        let threadgroupsCount = (count + threadsPerThreadgroupCount - 1) / threadsPerThreadgroupCount
        let threadgroups = MTLSize(width: threadgroupsCount, height: 1, depth: 1)

        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()

        cmdBuffer.commit()
        cmdBuffer.waitUntilCompleted()

        if let error = cmdBuffer.error {
            throw MetalVectorAddError.commandBufferError(error.localizedDescription)
        }

        // Read result
        var result = Array<Float>(repeating: 0, count: count)
        bufferOut.contents().copyMemory(to: &result, byteCount: byteCount)
        return result
    }
}

private extension UnsafeMutableRawPointer {
    func copyMemory<T>(from array: [T], byteCount: Int) {
        array.withUnsafeBytes { src in
            memcpy(self, src.baseAddress!, byteCount)
        }
    }

    func copyMemory<T>(to array: inout [T], byteCount: Int) {
        array.withUnsafeMutableBytes { dst in
            memcpy(dst.baseAddress!, self, byteCount)
        }
    }
}
