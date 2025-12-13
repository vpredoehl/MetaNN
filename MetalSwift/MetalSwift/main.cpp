#include <iostream>

extern "C" void cpp_run_matmul_add_demo();
extern "C" void cpp_run_gpu_demos();

// This project previously used Swift as the entry point. To use this C++ main instead:
// 1) Remove or exclude any Swift files that declare an entry (e.g., a type annotated with `@main`
//    or a SwiftUI `App` conformer) from the build target.
// 2) Ensure your target compiles and links this file (main.cpp) as part of the target's Sources.
// 3) If you still need to call into Swift code, expose C-compatible functions via `@_cdecl` in Swift
//    and declare them here with `extern "C"`.
// 4) For app bundles (iOS/macOS app), make sure your Info.plist and app lifecycle align with a C/C++
//    entry point. For command-line tools, this file alone is sufficient.

// Example of calling a Swift function (if provided):
// extern "C" void SwiftEntryPoint();

int main(int argc, char* argv[]) {
    std::cout << "C++ main running. argc=" << argc << std::endl;

    std::cout << "-- Running C++ GPU MatMul+Add demo --" << std::endl;
    cpp_run_matmul_add_demo();

    std::cout << "-- Running additional GPU demos --" << std::endl;
    cpp_run_gpu_demos();

    return 0;
}
