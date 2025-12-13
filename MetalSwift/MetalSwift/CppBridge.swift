import Foundation

@_silgen_name("cpp_run_matmul_add_demo")
func cpp_run_matmul_add_demo()

/// Runs a simple C++-backed vector addition via Objective-C++ bridge and prints the result.
func runCppDemo() {
    // Call into the C++ demo that performs GPU MatMul+Add and prints the result
    cpp_run_matmul_add_demo()
}

import Metal
import Foundation

/// Runs a simple vector addition on the GPU using a C++/Objective-C++ Metal bridge and prints the result.
@_silgen_name("cpp_gpu_vector_add")
func cpp_gpu_vector_add(_ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>, _ out: UnsafeMutablePointer<Float>, _ count: Int) -> Int32

func runGpuDemo() {
    let a: [Float] = [1, 2, 3, 4, 5]
    let b: [Float] = [10, 20, 30, 40, 50]
    var out = Array<Float>(repeating: 0, count: a.count)

    let status = a.withUnsafeBufferPointer { aPtr in
        b.withUnsafeBufferPointer { bPtr in
            out.withUnsafeMutableBufferPointer { oPtr in
                cpp_gpu_vector_add(aPtr.baseAddress!, bPtr.baseAddress!, oPtr.baseAddress!, a.count)
            }
        }
    }

    if status == 0 {
        let formatted = out.map { String(format: "%g", $0) }.joined(separator: ", ")
        print("GPU (Metal via C++ bridge) result: [\(formatted)]")
    } else {
        print("GPU compute failed with status: \(status)")
    }
}

