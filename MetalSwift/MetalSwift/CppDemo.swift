import Foundation

/// Swift-only demo that mirrors the behavior of the old C++ demo.
/// It uses `MetalVectorAdder` to add two arrays on the GPU and prints the result.
func runSwiftDemo() {
    let a: [Float] = [1, 2, 3, 4, 5]
    let b: [Float] = [10, 20, 30, 40, 50]
    do {
        let out = try MetalVectorAdder.add(a: a, b: b)
        let formatted = out.map { String(format: "%g", $0) }.joined(separator: ", ")
        print("Swift GPU result: [\(formatted)]")
    } catch {
        print("Swift Metal error: \(error.localizedDescription)")
    }
}

/// Call this from your app entry point to test the Objective-C++ bridge.
/// Example:
///   runCppDemo()
/// This prints the result of a C++ vector addition.
// runCppDemo()  // Uncomment to run the C++ demo
