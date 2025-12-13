import Foundation
import Metal

enum MetalMatMulError: Error, LocalizedError {
    case deviceUnavailable
    case commandQueueCreationFailed
    case libraryBuildFailed(String)
    case functionNotFound(String)
    case pipelineCreationFailed(String)
    case bufferAllocationFailed
    case commandBufferCreationFailed
    case encoderCreationFailed
    case shapeMismatch
    case commandBufferError(String)

    var errorDescription: String? {
        switch self {
        case .deviceUnavailable: return "Failed to create system default Metal device."
        case .commandQueueCreationFailed: return "Failed to create Metal command queue."
        case .libraryBuildFailed(let reason): return "Failed to build Metal library: \(reason)"
        case .functionNotFound(let name): return "Failed to find function '\(name)' in Metal library."
        case .pipelineCreationFailed(let reason): return "Failed to create compute pipeline state: \(reason)"
        case .bufferAllocationFailed: return "Failed to allocate Metal buffers."
        case .commandBufferCreationFailed: return "Failed to create Metal command buffer."
        case .encoderCreationFailed: return "Failed to create Metal compute command encoder."
        case .shapeMismatch: return "Matrix shape mismatch: colsA must equal rowsB."
        case .commandBufferError(let reason): return "Metal command buffer error: \(reason)"
        }
    }
}

struct MetalMatrixMultiplier {
    /// Multiplies matrices A (rowsA x colsA) and B (rowsB x colsB) producing C (rowsA x colsB).
    /// - Parameters:
    ///   - a: Flattened input matrix A.
    ///   - rowsA: Number of rows in matrix A.
    ///   - colsA: Number of columns in matrix A.
    ///   - b: Flattened input matrix B.
    ///   - rowsB: Number of rows in matrix B.
    ///   - colsB: Number of columns in matrix B.
    ///   - tileSize: Tile size to use in the GPU kernel (default 16).
    /// - Returns: Flattened output matrix C.
    /// - Throws: MetalMatMulError on failure.
    static func matmul(a: [Float], rowsA: Int, colsA: Int,
                       b: [Float], rowsB: Int, colsB: Int,
                       tileSize: Int = 16) throws -> [Float] {
        precondition(rowsA >= 0 && colsA >= 0 && rowsB >= 0 && colsB >= 0, "Matrix dimensions must be non-negative")
        guard colsA == rowsB else { throw MetalMatMulError.shapeMismatch }
        guard a.count == rowsA * colsA, b.count == rowsB * colsB else {
            throw MetalMatMulError.shapeMismatch
        }
        let rowsC = rowsA
        let colsC = colsB
        let countC = rowsC * colsC
        if rowsC == 0 || colsC == 0 { return [] }

        guard let device = MTLCreateSystemDefaultDevice() else { throw MetalMatMulError.deviceUnavailable }
        guard let commandQueue = device.makeCommandQueue() else { throw MetalMatMulError.commandQueueCreationFailed }

        let source = """
        #include <metal_stdlib>
        using namespace metal;

        // Specialization constant for tile size (default 16)
        constant uint TILE_SIZE [[function_constant(0)]];

        struct MatDims {
            uint rowsA;
            uint colsA;
            uint rowsB;
            uint colsB;
        };

        // Fallback if the driver doesn't provide TILE_SIZE
        inline uint tileSize() {
            return TILE_SIZE == 0 ? 16 : TILE_SIZE;
        }

        kernel void matmul_tiled_f32(
            device const float* A [[ buffer(0) ]],
            device const float* B [[ buffer(1) ]],
            device float* C [[ buffer(2) ]],
            constant MatDims& dims [[ buffer(3) ]],
            uint2 tgp_id [[ threadgroup_position_in_grid ]],
            uint2 tid [[ thread_position_in_threadgroup ]]) {
            const uint TS = tileSize();

            const uint row = tgp_id.y * TS + tid.y;
            const uint col = tgp_id.x * TS + tid.x;

            threadgroup float As[32][32];
            threadgroup float Bs[32][32];

            float sum = 0.0f;

            const uint numTiles = (dims.colsA + TS - 1) / TS;

            for (uint t = 0; t < numTiles; ++t) {
                const uint aCol = t * TS + tid.x;
                const uint bRow = t * TS + tid.y;

                if (row < dims.rowsA && aCol < dims.colsA) {
                    As[tid.y][tid.x] = A[row * dims.colsA + aCol];
                } else {
                    As[tid.y][tid.x] = 0.0f;
                }

                if (bRow < dims.rowsB && col < dims.colsB) {
                    Bs[tid.y][tid.x] = B[bRow * dims.colsB + col];
                } else {
                    Bs[tid.y][tid.x] = 0.0f;
                }

                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (uint k = 0; k < TS; ++k) {
                    sum += As[tid.y][k] * Bs[k][tid.x];
                }

                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (row < dims.rowsA && col < dims.colsB) {
                C[row * dims.colsB + col] = sum;
            }
        }

        kernel void matmul_tiled_f16(
            device const half* A [[ buffer(0) ]],
            device const half* B [[ buffer(1) ]],
            device half* C [[ buffer(2) ]],
            constant MatDims& dims [[ buffer(3) ]],
            uint2 tgp_id [[ threadgroup_position_in_grid ]],
            uint2 tid [[ thread_position_in_threadgroup ]]) {
            const uint TS = tileSize();

            const uint row = tgp_id.y * TS + tid.y;
            const uint col = tgp_id.x * TS + tid.x;

            threadgroup half As[32][32];
            threadgroup half Bs[32][32];

            half sum = (half)0.0h;

            const uint numTiles = (dims.colsA + TS - 1) / TS;

            for (uint t = 0; t < numTiles; ++t) {
                const uint aCol = t * TS + tid.x;
                const uint bRow = t * TS + tid.y;

                if (row < dims.rowsA && aCol < dims.colsA) {
                    As[tid.y][tid.x] = A[row * dims.colsA + aCol];
                } else {
                    As[tid.y][tid.x] = (half)0.0h;
                }

                if (bRow < dims.rowsB && col < dims.colsB) {
                    Bs[tid.y][tid.x] = B[bRow * dims.colsB + col];
                } else {
                    Bs[tid.y][tid.x] = (half)0.0h;
                }

                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (uint k = 0; k < TS; ++k) {
                    sum += As[tid.y][k] * Bs[k][tid.x];
                }

                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (row < dims.rowsA && col < dims.colsB) {
                C[row * dims.colsB + col] = sum;
            }
        }
        """

        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: source, options: nil)
        } catch {
            throw MetalMatMulError.libraryBuildFailed(String(describing: error))
        }

        guard (library.makeFunction(name: "matmul_tiled_f32")) != nil else {
            throw MetalMatMulError.functionNotFound("matmul_tiled_f32")
        }
        let constantValues = MTLFunctionConstantValues()
        var tileSizeConst: UInt32 = UInt32(tileSize)
        constantValues.setConstantValue(&tileSizeConst, type: .uint, index: 0)
        let specializedF32 = try library.makeFunction(name: "matmul_tiled_f32", constantValues: constantValues)
        let pipeline = try device.makeComputePipelineState(function: specializedF32)

        // Buffers
        let bytesA = MemoryLayout<Float>.stride * a.count
        let bytesB = MemoryLayout<Float>.stride * b.count
        let bytesC = MemoryLayout<Float>.stride * countC
        guard let bufA = device.makeBuffer(length: bytesA, options: .storageModeShared),
              let bufB = device.makeBuffer(length: bytesB, options: .storageModeShared),
              let bufC = device.makeBuffer(length: bytesC, options: .storageModeShared),
              let dimsBuf = device.makeBuffer(length: MemoryLayout<UInt32>.stride * 4, options: .storageModeShared) else {
            throw MetalMatMulError.bufferAllocationFailed
        }

        bufA.contents().copyMemory(from: a, byteCount: bytesA)
        bufB.contents().copyMemory(from: b, byteCount: bytesB)
        let dims = [UInt32(rowsA), UInt32(colsA), UInt32(rowsB), UInt32(colsB)]
        dimsBuf.contents().copyMemory(from: dims, byteCount: MemoryLayout<UInt32>.stride * 4)

        guard let cmd = commandQueue.makeCommandBuffer() else { throw MetalMatMulError.commandBufferCreationFailed }
        guard let enc = cmd.makeComputeCommandEncoder() else { throw MetalMatMulError.encoderCreationFailed }

        enc.setComputePipelineState(pipeline)
        enc.setBuffer(bufA, offset: 0, index: 0)
        enc.setBuffer(bufB, offset: 0, index: 1)
        enc.setBuffer(bufC, offset: 0, index: 2)
        enc.setBuffer(dimsBuf, offset: 0, index: 3)

        let tile = tileSize
        let threadsPerThreadgroup = MTLSize(width: tile, height: tile, depth: 1)
        let tgWidth = (colsC + tile - 1) / tile
        let tgHeight = (rowsC + tile - 1) / tile
        let threadsPerGrid = MTLSize(width: tgWidth * tile, height: tgHeight * tile, depth: 1)

        enc.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        enc.endEncoding()

        cmd.commit()
        cmd.waitUntilCompleted()

        if let e = cmd.error { throw MetalMatMulError.commandBufferError(e.localizedDescription) }

        var c = Array<Float>(repeating: 0, count: countC)
        bufC.contents().copyMemory(to: &c, byteCount: bytesC)
        return c
    }

    static func matmulHalf(a: [UInt16], rowsA: Int, colsA: Int,
                           b: [UInt16], rowsB: Int, colsB: Int,
                           tileSize: Int = 16) throws -> [UInt16] {
        precondition(rowsA >= 0 && colsA >= 0 && rowsB >= 0 && colsB >= 0, "Matrix dimensions must be non-negative")
        guard colsA == rowsB else { throw MetalMatMulError.shapeMismatch }
        guard a.count == rowsA * colsA, b.count == rowsB * colsB else {
            throw MetalMatMulError.shapeMismatch
        }
        let rowsC = rowsA
        let colsC = colsB
        let countC = rowsC * colsC
        if rowsC == 0 || colsC == 0 { return [] }

        guard let device = MTLCreateSystemDefaultDevice() else { throw MetalMatMulError.deviceUnavailable }
        guard let commandQueue = device.makeCommandQueue() else { throw MetalMatMulError.commandQueueCreationFailed }

        let source = """
        #include <metal_stdlib>
        using namespace metal;
        constant uint TILE_SIZE [[function_constant(0)]];
        struct MatDims { uint rowsA; uint colsA; uint rowsB; uint colsB; };
        inline uint tileSize() { return TILE_SIZE == 0 ? 16 : TILE_SIZE; }
        kernel void matmul_tiled_f16(
            device const half* A [[ buffer(0) ]],
            device const half* B [[ buffer(1) ]],
            device half* C [[ buffer(2) ]],
            constant MatDims& dims [[ buffer(3) ]],
            uint2 tgp_id [[ threadgroup_position_in_grid ]],
            uint2 tid [[ thread_position_in_threadgroup ]]) {
            const uint TS = tileSize();
            const uint row = tgp_id.y * TS + tid.y;
            const uint col = tgp_id.x * TS + tid.x;
            threadgroup half As[32][32];
            threadgroup half Bs[32][32];
            half sum = (half)0.0h;
            const uint numTiles = (dims.colsA + TS - 1) / TS;
            for (uint t = 0; t < numTiles; ++t) {
                const uint aCol = t * TS + tid.x;
                const uint bRow = t * TS + tid.y;
                As[tid.y][tid.x] = (row < dims.rowsA && aCol < dims.colsA) ? A[row * dims.colsA + aCol] : (half)0.0h;
                Bs[tid.y][tid.x] = (bRow < dims.rowsB && col < dims.colsB) ? B[bRow * dims.colsB + col] : (half)0.0h;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint k = 0; k < TS; ++k) { sum += As[tid.y][k] * Bs[k][tid.x]; }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (row < dims.rowsA && col < dims.colsB) { C[row * dims.colsB + col] = sum; }
        }
        """

        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: source, options: nil)
        } catch {
            throw MetalMatMulError.libraryBuildFailed(String(describing: error))
        }

        let constantValues = MTLFunctionConstantValues()
        var tileSizeConst: UInt32 = UInt32(tileSize)
        constantValues.setConstantValue(&tileSizeConst, type: .uint, index: 0)

        guard (library.makeFunction(name: "matmul_tiled_f16")) != nil else {
            throw MetalMatMulError.functionNotFound("matmul_tiled_f16")
        }
        let specialized = try library.makeFunction(name: "matmul_tiled_f16", constantValues: constantValues)
        let pipeline = try device.makeComputePipelineState(function: specialized)

        let bytesA = MemoryLayout<UInt16>.stride * a.count
        let bytesB = MemoryLayout<UInt16>.stride * b.count
        let bytesC = MemoryLayout<UInt16>.stride * countC
        guard let bufA = device.makeBuffer(length: bytesA, options: .storageModeShared),
              let bufB = device.makeBuffer(length: bytesB, options: .storageModeShared),
              let bufC = device.makeBuffer(length: bytesC, options: .storageModeShared),
              let dimsBuf = device.makeBuffer(length: MemoryLayout<UInt32>.stride * 4, options: .storageModeShared) else {
            throw MetalMatMulError.bufferAllocationFailed
        }

        bufA.contents().copyMemory(from: a, byteCount: bytesA)
        bufB.contents().copyMemory(from: b, byteCount: bytesB)
        let dims = [UInt32(rowsA), UInt32(colsA), UInt32(rowsB), UInt32(colsB)]
        dimsBuf.contents().copyMemory(from: dims, byteCount: MemoryLayout<UInt32>.stride * 4)

        guard let cmd = commandQueue.makeCommandBuffer() else { throw MetalMatMulError.commandBufferCreationFailed }
        guard let enc = cmd.makeComputeCommandEncoder() else { throw MetalMatMulError.encoderCreationFailed }

        enc.setComputePipelineState(pipeline)
        enc.setBuffer(bufA, offset: 0, index: 0)
        enc.setBuffer(bufB, offset: 0, index: 1)
        enc.setBuffer(bufC, offset: 0, index: 2)
        enc.setBuffer(dimsBuf, offset: 0, index: 3)

        let tile = tileSize
        let threadsPerThreadgroup = MTLSize(width: tile, height: tile, depth: 1)
        let tgWidth = (colsC + tile - 1) / tile
        let tgHeight = (rowsC + tile - 1) / tile
        let threadsPerGrid = MTLSize(width: tgWidth * tile, height: tgHeight * tile, depth: 1)

        enc.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        enc.endEncoding()

        cmd.commit()
        cmd.waitUntilCompleted()

        if let e = cmd.error { throw MetalMatMulError.commandBufferError(e.localizedDescription) }

        var c = Array<UInt16>(repeating: 0, count: countC)
        bufC.contents().copyMemory(to: &c, byteCount: bytesC)
        return c
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

// MARK: - Float16 conversion helpers
extension MetalMatrixMultiplier {
    // Convert Float32 array to IEEE 754 binary16 bit patterns (UInt16)
    static func float32ToFloat16Bits(_ src: [Float]) -> [UInt16] {
        return src.map { f in
            var value = f
            return withUnsafeBytes(of: &value) { bytes -> UInt16 in
                let bits = bytes.load(as: UInt32.self)
                let sign = UInt16((bits >> 16) & 0x8000)
                let exp = Int((bits >> 23) & 0xFF) - 127 + 15
                var mant = UInt32(bits & 0x7FFFFF)
                if exp <= 0 {
                    if exp < -10 { return sign } // underflow to zero
                    // subnormal
                    mant |= 0x800000
                    let shift = UInt32(14 - exp)
                    let halfMant = UInt16((mant >> (shift + 13)) & 0x3FF)
                    return sign | halfMant
                } else if exp >= 31 {
                    // overflow to inf
                    return sign | 0x7C00
                } else {
                    let halfExp = UInt16(exp & 0x1F)
                    let halfMant = UInt16((mant >> 13) & 0x3FF)
                    return sign | (halfExp << 10) | halfMant
                }
            }
        }
    }

    // Convert IEEE 754 binary16 bit patterns (UInt16) to Float32 array
    static func float16BitsToFloat32(_ src: [UInt16]) -> [Float] {
        return src.map { h in
            let sign = UInt32(h & 0x8000) << 16
            let exp = UInt32(h & 0x7C00) >> 10
            let mant = UInt32(h & 0x03FF)
            var bits: UInt32
            if exp == 0 {
                if mant == 0 {
                    bits = sign
                } else {
                    // subnormal
                    var e = -14
                    var m = mant
                    while (m & 0x0400) == 0 { m <<= 1; e -= 1 }
                    m &= 0x03FF
                    bits = sign | UInt32(e + 127) << 23 | (m << 13)
                }
            } else if exp == 0x1F {
                // inf/NaN
                bits = sign | 0x7F800000 | (mant << 13)
            } else {
                let e = Int(exp) - 15 + 127
                bits = sign | UInt32(e) << 23 | (mant << 13)
            }
            var f = Float.zero
            withUnsafeMutableBytes(of: &f) { ptr in
                ptr.storeBytes(of: bits, as: UInt32.self)
            }
            return f
        }
    }
}

// MARK: - Benchmarking helper
extension MetalMatrixMultiplier {
    /// Benchmarks several tile sizes and returns the best performing size and average time (ms).
    /// - Parameters:
    ///   - a, b: Input matrices (Float32)
    ///   - rowsA, colsA, rowsB, colsB: Dimensions
    ///   - candidates: Tile sizes to test (default [8, 16, 32])
    ///   - iterations: Number of runs to average per candidate
    /// - Returns: (bestTileSize, avgMilliseconds)
    static func benchmarkTileSizes(a: [Float], rowsA: Int, colsA: Int,
                                   b: [Float], rowsB: Int, colsB: Int,
                                   candidates: [Int] = [8, 16, 32],
                                   iterations: Int = 3) throws -> (Int, Double) {
        precondition(iterations > 0)
        var bestSize = candidates.first ?? 16
        var bestTime = Double.infinity
        for sz in candidates {
            var total: Double = 0
            for _ in 0..<iterations {
                let start = CFAbsoluteTimeGetCurrent()
                _ = try matmul(a: a, rowsA: rowsA, colsA: colsA,
                                b: b, rowsB: rowsB, colsB: colsB,
                                tileSize: sz)
                let end = CFAbsoluteTimeGetCurrent()
                total += (end - start) * 1000.0
            }
            let avg = total / Double(iterations)
            if avg < bestTime {
                bestTime = avg
                bestSize = sz
            }
        }
        return (bestSize, bestTime)
    }
}

// MARK: - GPU Vector Add and MatMul+Add
extension MetalMatrixMultiplier {
    /// Adds D into C on the GPU in-place. C and D must have the same count.
    static func addInPlace(c: inout [Float], d: [Float]) throws {
        enum AddErr: Error { case deviceUnavailable, commandQueueCreationFailed, libraryBuildFailed(String), functionNotFound, pipelineCreationFailed(String), bufferAllocationFailed, commandBufferCreationFailed, encoderCreationFailed, commandBufferError(String), sizeMismatch }
        guard c.count == d.count else { throw AddErr.sizeMismatch }
        guard let device = MTLCreateSystemDefaultDevice() else { throw AddErr.deviceUnavailable }
        guard let commandQueue = device.makeCommandQueue() else { throw AddErr.commandQueueCreationFailed }

        let source = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void vadd(device float* C [[ buffer(0) ]],
                         device const float* D [[ buffer(1) ]],
                         uint gid [[ thread_position_in_grid ]]) {
            C[gid] += D[gid];
        }
        """
        let library: MTLLibrary
        do { library = try device.makeLibrary(source: source, options: nil) } catch { throw AddErr.libraryBuildFailed(String(describing: error)) }
        guard let fn = library.makeFunction(name: "vadd") else { throw AddErr.functionNotFound }
        let pipeline: MTLComputePipelineState
        do { pipeline = try device.makeComputePipelineState(function: fn) } catch { throw AddErr.pipelineCreationFailed(String(describing: error)) }

        let count = c.count
        let bytes = MemoryLayout<Float>.stride * count
        guard let bufC = device.makeBuffer(bytes: c, length: bytes, options: .storageModeShared),
              let bufD = device.makeBuffer(bytes: d, length: bytes, options: .storageModeShared) else {
            throw AddErr.bufferAllocationFailed
        }

        guard let cmd = commandQueue.makeCommandBuffer() else { throw AddErr.commandBufferCreationFailed }
        guard let enc = cmd.makeComputeCommandEncoder() else { throw AddErr.encoderCreationFailed }
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(bufC, offset: 0, index: 0)
        enc.setBuffer(bufD, offset: 0, index: 1)
        let w = pipeline.threadExecutionWidth
        let threadsPerThreadgroup = MTLSize(width: w, height: 1, depth: 1)
        let threadsPerGrid = MTLSize(width: count, height: 1, depth: 1)
        enc.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        if let e = cmd.error { throw AddErr.commandBufferError(e.localizedDescription) }
        bufC.contents().copyMemory(to: &c, byteCount: bytes)
    }

    /// Computes C = A(m×k) × B(k×n) + D(m×n) fully on the GPU.
    static func matmulThenAdd(a: [Float], rowsA: Int, colsA: Int,
                              b: [Float], rowsB: Int, colsB: Int,
                              d: [Float],
                              tileSize: Int = 16) throws -> [Float] {
        precondition(d.count == rowsA * colsB, "D must be m×n")
        var c = try matmul(a: a, rowsA: rowsA, colsA: colsA,
                           b: b, rowsB: rowsB, colsB: colsB,
                           tileSize: tileSize)
        try addInPlace(c: &c, d: d)
        return c
    }
}
