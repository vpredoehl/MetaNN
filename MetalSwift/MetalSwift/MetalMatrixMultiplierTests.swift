import Foundation
import Testing

@testable import MetalSwift

@Suite("MetalMatrixMultiplier tests")
struct MetalMatrixMultiplierTests {

    // Simple CPU reference matmul for Float32
    func cpuMatmul(_ a: [Float], _ rowsA: Int, _ colsA: Int,
                   _ b: [Float], _ rowsB: Int, _ colsB: Int) -> [Float] {
        precondition(colsA == rowsB)
        var c = Array<Float>(repeating: 0, count: rowsA * colsB)
        for i in 0..<rowsA {
            for k in 0..<colsA {
                let aik = a[i * colsA + k]
                for j in 0..<colsB {
                    c[i * colsB + j] += aik * b[k * colsB + j]
                }
            }
        }
        return c
    }

    func testF32Correctness() async throws {
        let m = 50, k = 10, n = 10
        var a = [Float]()
        var b = [Float]()
        for i in 0..<(m*k) { a.append(Float((i * 7) % 13) / 3.0) }
        for i in 0..<(k*n) { b.append(Float((i * 11) % 17) / 5.0) }

        let ref = cpuMatmul(a, m, k, b, k, n)
        let gpu = try MetalMatrixMultiplier.matmul(a: a, rowsA: m, colsA: k,
                                                   b: b, rowsB: k, colsB: n,
                                                   tileSize: 16)
        // Allow tiny FP error
        for i in 0..<(m*n) {
            #expect(abs(ref[i] - gpu[i]) < 1e-3, "Mismatch at index \(i): ref=\(ref[i]) gpu=\(gpu[i])")
        }
    }

    func testF16RoundTripAndMatmul() async throws {
        let m = 50, k = 10, n = 10
        let a32 = (0..<(m*k)).map { i in Float((i % 7) - 3) / 2.0 }
        let b32 = (0..<(k*n)).map { i in Float((i % 11) - 5) / 3.0 }
        let a16 = MetalMatrixMultiplier.float32ToFloat16Bits(a32)
        let b16 = MetalMatrixMultiplier.float32ToFloat16Bits(b32)

        let c16 = try MetalMatrixMultiplier.matmulHalf(a: a16, rowsA: m, colsA: k,
                                                       b: b16, rowsB: k, colsB: n,
                                                       tileSize: 16)
        let c32 = MetalMatrixMultiplier.float16BitsToFloat32(c16)
        // Just check finite and reasonable magnitude
        for v in c32 {
            #expect(v.isFinite)
            #expect(abs(v) < 1e6)
        }
    }

    func testBenchmarkHelper() async throws {
        let m = 50, k = 10, n = 10
        let a = (0..<(m*k)).map { _ in Float.random(in: -1...1) }
        let b = (0..<(k*n)).map { _ in Float.random(in: -1...1) }
        let (best, avgMs) = try MetalMatrixMultiplier.benchmarkTileSizes(
            a: a, rowsA: m, colsA: k,
            b: b, rowsB: k, colsB: n,
            candidates: [8, 16],
            iterations: 2
        )
        #expect([8,16].contains(best))
        #expect(avgMs > 0)
    }

    @Test("Matmul then Add (GPU+GPU) matches CPU reference")
    func testMatmulThenAddGPU() async throws {
        let m = 50, k = 10, n = 10
        let a = (0..<(m*k)).map { Float(($0 * 7) % 13) / 3.0 }
        let b = (0..<(k*n)).map { Float(($0 * 11) % 17) / 5.0 }
        let d = (0..<(m*n)).map { _ in Float(0.25) }

        // CPU reference: C = A×B + D
        let refMatmul = cpuMatmul(a, m, k, b, k, n)
        let ref = zip(refMatmul, d).map(+)

        let gpu = try MetalMatrixMultiplier.matmulThenAdd(a: a, rowsA: m, colsA: k,
                                                          b: b, rowsB: k, colsB: n,
                                                          d: d,
                                                          tileSize: 16)
        for i in 0..<(m*n) {
            #expect(abs(ref[i] - gpu[i]) < 1e-3, "Mismatch at index \(i): ref=\(ref[i]) gpu=\(gpu[i])")
        }
    }
}
