#pragma once
#ifdef __cplusplus
extern "C" {
#endif

// C entry point implemented in MatMulAddDemo.cpp
// Call this from C++ or expose it to Swift via @_silgen_name/extern "C" if needed.
void cpp_run_matmul_add_demo(void);

#ifdef __cplusplus
}
#endif
