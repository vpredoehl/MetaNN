import Foundation

@_cdecl("swift_gpu_matmul_add")
public func swift_gpu_matmul_add(_ A: UnsafePointer<Float>, _ rowsA: Int32, _ colsA: Int32,
                                 _ B: UnsafePointer<Float>, _ rowsB: Int32, _ colsB: Int32,
                                 _ D: UnsafePointer<Float>,
                                 _ C: UnsafeMutablePointer<Float>) -> Int32 {
    let m = Int(rowsA), k = Int(colsA), k2 = Int(rowsB), n = Int(colsB)
    // Validate shapes
    if k != k2 { return 1 }
    if m < 0 || k < 0 || k2 < 0 || n < 0 { return 1 }

    // Convert raw buffers into Swift arrays
    let a = Array(UnsafeBufferPointer(start: A, count: m * k))
    let b = Array(UnsafeBufferPointer(start: B, count: k * n))
    let d = Array(UnsafeBufferPointer(start: D, count: m * n))

    do {
        let out = try MetalMatrixMultiplier.matmulThenAdd(a: a, rowsA: m, colsA: k,
                                                          b: b, rowsB: k, colsB: n,
                                                          d: d, tileSize: 16)
        out.withUnsafeBufferPointer { src in
            C.update(from: src.baseAddress!, count: m * n)
        }
        return 0
    } catch {
        return 2
    }
}
