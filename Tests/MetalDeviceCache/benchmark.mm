#import <Foundation/Foundation.h>
#ifdef COUNT_DEVICE_RESOLUTIONS
#include "device_instrumentation.h"
#endif

// Include the allocator to observe existing private diagnostics without adding
// production APIs or changing the counters in the allocation hot path.
#ifndef METANN_ALLOCATOR_SOURCE
#define METANN_ALLOCATOR_SOURCE "../../MetaNN/data/facilities/continuous_memory_metal.mm"
#endif
#include METANN_ALLOCATOR_SOURCE
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <iomanip>
#include <thread>

class StartGate
{
    std::mutex mutex;
    std::condition_variable cv;
    size_t ready = 0;
    bool start = false;
public:
    void Arrive()
    {
        std::unique_lock<std::mutex> lock(mutex);
        ++ready;
        cv.notify_all();
        cv.wait(lock, [&] { return start; });
    }
    void Wait(size_t count)
    {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&] { return ready == count; });
    }
    void Release()
    {
        std::lock_guard<std::mutex> lock(mutex);
        start = true;
        cv.notify_all();
    }
};

template<class T>
void Allocate(size_t bytes, size_t iterations)
{
    @autoreleasepool
    {
        for (size_t i = 0; i < iterations; ++i)
        {
            MetaNN::ContinuousMemory<T, MetaNN::DeviceTags::Metal> memory(bytes / sizeof(T));
            // Touch both ends to keep storage use and numerical checks observable.
            auto* data = memory.MutableRawMemory();
            data[0] = T(1.25);
            data[memory.Size() - 1] = T(2.5);
            if (data[0] != T(1.25) || data[memory.Size() - 1] != T(2.5)) std::abort();
        }
    }
}

int main(int argc, char** argv)
{
    if (argc != 5) return 2;
    const size_t bytes = std::stoull(argv[1]);
    const size_t threads = std::stoull(argv[2]);
    const size_t iterations = std::stoull(argv[3]);
    const bool useDouble = std::string(argv[4]) == "double";
    if (bytes < 16 || bytes % 8 || threads == 0 || iterations == 0) return 2;
    @autoreleasepool
    {
        // Warm all workers and populate the pool; initialization is excluded.
        StartGate gate;
        std::vector<std::thread> workers;
        for (size_t t = 0; t < threads; ++t)
        {
            workers.emplace_back([&] {
                if (useDouble) Allocate<double>(bytes, 100);
                else Allocate<float>(bytes, 100);
                gate.Arrive();
                if (useDouble) Allocate<double>(bytes, iterations);
                else Allocate<float>(bytes, iterations);
            });
        }
        gate.Wait(threads);
        const auto allocations = g_metal_buffer_alloc_count.load();
        const auto reuses = g_metal_buffer_reuse_count.load();
#ifdef COUNT_DEVICE_RESOLUTIONS
        const auto resolutions = deviceResolutionCount.load();
#endif
        const auto begin = std::chrono::steady_clock::now();
        gate.Release();
        for (auto& worker : workers) worker.join();
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        const auto allocated = g_metal_buffer_alloc_count.load() - allocations;
        const auto reused = g_metal_buffer_reuse_count.load() - reuses;
        std::cout << std::setprecision(12)
                  << "{\"bytes\":" << bytes << ",\"threads\":" << threads
                  << ",\"type\":\"" << (useDouble ? "double" : "float")
                  << "\",\"allocations\":" << allocated << ",\"reuses\":" << reused
                  << ",\"new_buffers\":" << allocated - reused
                  << ",\"live_after\":" << g_metal_buffer_live_count.load()
                  << ",\"live_bytes_after\":" << g_metal_buffer_live_bytes.load()
#ifdef COUNT_DEVICE_RESOLUTIONS
                  << ",\"device_resolutions\":" << deviceResolutionCount.load() - resolutions
                  << ",\"device_resolutions_process\":" << deviceResolutionCount.load()
#endif
                  << ",\"seconds\":" << seconds
                  << ",\"allocations_per_second\":" << allocated / seconds << "}\n";
        if (allocated != threads * iterations || g_metal_buffer_live_count.load() != 0) return 1;
    }
}
