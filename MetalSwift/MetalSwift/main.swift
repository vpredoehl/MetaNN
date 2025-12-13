import Foundation

let a: [Float] = [1, 2, 3, 4, 5]
let b: [Float] = [10, 20, 30, 40, 50]

do {
    let out = try MetalVectorAdder.add(a: a, b: b)
    print("GPU result:", out)
} catch {
    print("Metal error:", error.localizedDescription)
}

// Matrix multiplication demo: (2x3) * (3x2) -> (2x2)
let A: [Float] = [
    1, 2, 3,
    4, 5, 6
]
let B: [Float] = [
    7, 8,
    9, 10,
    11, 12
]

let rowsA = 2, colsA = 3
let rowsB = 3, colsB = 2

do {
    let C = try MetalMatrixMultiplier.matmul(a: A, rowsA: rowsA, colsA: colsA,
                                             b: B, rowsB: rowsB, colsB: colsB)
    // Pretty-print the 2x2 result
    for r in 0..<rowsA {
        let row = (0..<colsB).map { String(format: "%g", C[r * colsB + $0]) }.joined(separator: ", ")
        print("MatMul C[\(r)]: [\(row)]")
    }
} catch {
    print("MatMul error:", error.localizedDescription)
}

runSwiftDemo()
runCppDemo()

