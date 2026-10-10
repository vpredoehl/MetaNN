#include "device_instrumentation.h"
#include "../../MetaNN/data/facilities/continuous_memory_metal.mm"
#include <MetaNN/data/tensor.h>
#include <MetaNN/metal/metal_matmul.h>
#include <condition_variable>
#include <set>
#include <sstream>
#include <thread>

using namespace MetaNN;
template<class T> using Memory = ContinuousMemory<T, DeviceTags::Metal>;

// Standalone tests have no application runtime to provide this optional hook.
static std::atomic<bool> diagnosticLogging{true};
extern "C" bool LstmRuntimeDiagnosticLoggingEnabled() { return diagnosticLogging.load(); }

// Buffers belong to the underlying real device, so they cannot keep this proxy
// alive. Its weak reference specifically tests the cache's owning reference.
@interface LifetimeDevice : NSObject
@property(nonatomic, strong) id<MTLDevice> underlying;
@property(nonatomic) bool failBufferAllocation;
- (id<MTLBuffer>)newBufferWithLength:(NSUInteger)length options:(MTLResourceOptions)options;
@end
@implementation LifetimeDevice
- (id<MTLBuffer>)newBufferWithLength:(NSUInteger)length options:(MTLResourceOptions)options
{
    return self.failBufferAllocation ? nil : [self.underlying newBufferWithLength:length options:options];
}
@end

static __weak LifetimeDevice* weakDevice;
static id<MTLDevice> MakeLifetimeDevice()
{
    LifetimeDevice* proxy = [LifetimeDevice new];
    // The test interceptor macro is deliberately bypassed for the real GPU.
#undef MTLCreateSystemDefaultDevice
    proxy.underlying = MTLCreateSystemDefaultDevice();
#define MTLCreateSystemDefaultDevice ResolveCountedDevice
    if (!proxy.underlying) throw std::runtime_error("Regression requires an accessible Metal device");
    weakDevice = proxy;
    return (id<MTLDevice>)proxy;
}

void CheckRuntimeDevices(id<MTLDevice> expected);

template<class F>
void ExpectError(F action, const char* expected)
{
    bool threw = false;
    try { action(); }
    catch (const std::runtime_error& error)
    {
        threw = true;
        assert(std::string(error.what()) == expected);
    }
    assert(threw);
}

template<class T>
void CheckOffsetsAndReuse()
{
    // Use a previously untouched byte size for a deterministic pool hit.
    const size_t count = 19;
    void* handle;
    {
        Memory<T> memory(count);
        assert(!memory.IsShared() && memory.Offset() == 0);
        for (size_t i = 0; i < count; ++i) memory.MutableRawMemory()[i] = T(i) + T(0.125);
        auto view = memory.Shift(3).Shift(4);
        assert(memory.IsShared() && view.IsShared());
        assert(view.Size() == count - 7 && view.Offset() == 7);
        assert(view.RawMemory() == memory.RawMemory() + 7);
        assert(view.NativeHandle() == memory.NativeHandle());
        view.MutableRawMemory()[1] = T(42.25);
        assert(memory.RawMemory()[8] == T(42.25));
        handle = memory.NativeHandle();
    }
    const auto reuse = g_metal_buffer_reuse_count.load();
    Memory<T> reused(count);
    assert(reused.NativeHandle() == handle && reused.Offset() == 0 && reused.Size() == count);
    assert(g_metal_buffer_reuse_count.load() == reuse + 1);
    assert(reused.RawMemory()[8] == T(42.25));
}

void CheckConcurrentColdStart()
{
    std::mutex mutex;
    std::condition_variable cv;
    size_t ready = 0;
    bool start = false;
    std::set<void*> active;
    std::vector<std::thread> threads;
    for (size_t t = 0; t < 8; ++t)
    {
        threads.emplace_back([&, t] {
            @autoreleasepool
            {
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    ++ready;
                    cv.notify_all();
                    cv.wait(lock, [&] { return start; });
                }
                for (size_t i = 0; i < 500; ++i)
                {
                    // Mix pool hits, pool misses, types and unpooled sizes.
                    Memory<float> f(i % 4 == 0 ? 2048 : 32 + i % 8);
                    Memory<double> d(i % 4 == 0 ? 1024 : 16 + i % 8);
                    {
                        std::lock_guard<std::mutex> lock(mutex);
                        assert(active.insert(f.NativeHandle()).second);
                        assert(active.insert(d.NativeHandle()).second);
                    }
                    const float value = float(t * 1000 + i) + 0.25f;
                    f.MutableRawMemory()[0] = value;
                    f.MutableRawMemory()[f.Size() - 1] = -value;
                    d.MutableRawMemory()[0] = double(value) + 0.125;
                    d.MutableRawMemory()[d.Size() - 1] = -double(value);
                    auto view = f.Shift(1);
                    assert(view.Offset() == 1 && view.RawMemory() == f.RawMemory() + 1);
                    assert(f.RawMemory()[0] == value && f.RawMemory()[f.Size() - 1] == -value);
                    assert(d.RawMemory()[0] == double(value) + 0.125);
                    assert(d.RawMemory()[d.Size() - 1] == -double(value));
                    {
                        std::lock_guard<std::mutex> lock(mutex);
                        active.erase(f.NativeHandle());
                        active.erase(d.NativeHandle());
                    }
                }
            }
        });
    }
    {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&] { return ready == 8; });
        start = true;
        cv.notify_all();
    }
    for (auto& thread : threads) thread.join();
    assert(active.empty());
    assert(deviceResolutionCount.load() == 1);
    assert(g_metal_buffer_live_count.load() == 0);
    assert(g_metal_buffer_live_bytes.load() == 0);
}

void CheckPoolLimitAndSharedOwnership()
{
    const auto returns = g_metal_buffer_pool_return_count.load();
    const auto drops = g_metal_buffer_pool_drop_count.load();
    std::vector<Memory<float>> buffers;
    for (size_t i = 0; i < 300; ++i) buffers.emplace_back(20);
    buffers.clear();
    assert(g_metal_buffer_pool_return_count.load() - returns == 256);
    assert(g_metal_buffer_pool_drop_count.load() - drops == 44);
    const auto reuse = g_metal_buffer_reuse_count.load();
    for (size_t i = 0; i < 256; ++i) buffers.emplace_back(20);
    assert(g_metal_buffer_reuse_count.load() - reuse == 256);
    buffers.clear();

    std::unique_ptr<Memory<double>> view;
    void* handle;
    {
        Memory<double> parent(31);
        parent.MutableRawMemory()[5] = 123.125;
        handle = parent.NativeHandle();
        view = std::make_unique<Memory<double>>(parent.Shift(5));
    }
    Memory<double> distinct(31);
    assert(distinct.NativeHandle() != handle);
    assert(view->RawMemory()[0] == 123.125 && !view->IsShared());
    view.reset();
    Memory<double> reused(31);
    assert(reused.NativeHandle() == handle);
}

void CheckPoolBoundariesAndTypes()
{
    void* handle;
    {
        Memory<float> f(76);
        handle = f.NativeHandle();
    }
    {
        Memory<double> d(38);
        assert(d.NativeHandle() == handle); // Same byte bucket across types.
    }
    const auto reused = g_metal_buffer_reuse_count.load();
    {
        Memory<float> boundary(1024);
        handle = boundary.NativeHandle();
    }
    {
        Memory<double> boundary(512);
        assert(boundary.NativeHandle() == handle);
    }
    assert(g_metal_buffer_reuse_count.load() >= reused + 1);
    const auto unpooled = g_metal_buffer_reuse_count.load();
    { Memory<float> large(1026); }
    { Memory<double> large(513); }
    assert(g_metal_buffer_reuse_count.load() == unpooled);
}

template<class T>
void CheckTensorValues()
{
    Tensor<T, DeviceTags::Metal, 2> matrix(3, 5);
    for (size_t r = 0; r < 3; ++r)
        for (size_t c = 0; c < 5; ++c) matrix.SetValue(r, c, T(r * 5 + c) + T(0.125));
    auto row = matrix[2];
    for (size_t c = 0; c < 5; ++c) assert(row(c) == T(10 + c) + T(0.125));
    assert(!matrix.AvailableForWrite());
    Tensor<T, DeviceTags::Metal, 0> scalar(T(2.25));
    assert(scalar.Value() == T(2.25));
}

void CheckMatMulNumerics()
{
    // Offset-backed inputs/output exercise existing MPS offset semantics.
    Memory<float> baseA(8), baseB(8), baseC(8);
    auto a = baseA.Shift(4), b = baseB.Shift(4), c = baseC.Shift(4);
    for (size_t i = 0; i < 4; ++i)
    {
        a.MutableRawMemory()[i] = float(i + 1);
        b.MutableRawMemory()[i] = float(i + 5);
        baseC.MutableRawMemory()[i] = -123.f;
    }
    NSMetalMatMul::MatMul(a, b, c, 2, 2, 2);
    const float expected[] = {19, 22, 43, 50};
    for (size_t i = 0; i < 4; ++i)
    {
        assert(c.RawMemory()[i] == expected[i]);
        assert(baseC.RawMemory()[i] == -123.f);
    }
}

int main(int argc, char** argv)
{
    ExpectError([] { Memory<float> invalid(0); }, "Metal ContinuousMemory allocated with size 0");
    assert(deviceResolutionCount.load() == 0);
    if (argc == 2 && std::string(argv[1]) == "unavailable")
    {
        simulateUnavailableDevice = true;
        for (int i = 0; i < 3; ++i)
        {
            ExpectError([] { Memory<float> invalid(32); }, "Metal device not available.");
            ExpectError([] { Memory<double> invalid(32); }, "Metal device not available.");
        }
        assert(deviceResolutionCount.load() == 1);
        assert(g_metal_buffer_alloc_count.load() == 0);
        std::cout << "PASS unavailable device: one resolution, unchanged errors\n";
        return 0;
    }
    const bool lifetime = argc == 2 && std::string(argv[1]) == "lifetime";
    const bool delayedDiagnostics = argc == 2 && std::string(argv[1]) == "diagnostics";
    if (lifetime) resolveTestDevice = MakeLifetimeDevice;
    @autoreleasepool
    {
        std::ostringstream suppressed;
        auto* output = std::cout.rdbuf();
        if (delayedDiagnostics)
        {
            diagnosticLogging = false;
            std::cout.rdbuf(suppressed.rdbuf());
        }
        CheckConcurrentColdStart();
        if (delayedDiagnostics)
        {
            std::cout.rdbuf(output);
            assert(suppressed.str().empty());
            diagnosticLogging = true;
        }
        CheckOffsetsAndReuse<float>();
        CheckOffsetsAndReuse<double>();
        CheckPoolLimitAndSharedOwnership();
        CheckPoolBoundariesAndTypes();
        CheckTensorValues<float>();
        CheckTensorValues<double>();
        CheckRuntimeDevices(NSMetal::DefaultDevice());
        if (lifetime)
        {
            weakDevice.failBufferAllocation = true;
            ExpectError([] { Memory<float> invalid(10000); }, "Failed to allocate Metal buffer.");
            weakDevice.failBufferAllocation = false;
            Memory<double> retry(10000);
        }
        else CheckMatMulNumerics();
    }
    // Remove every pooled buffer and drain autoreleases: only the cache owns
    // the proxy now, while the underlying GPU device has separate ownership.
    for (auto& bucket : g_metal_small_buffer_pool) bucket.clear();
    @autoreleasepool
    {
        if (lifetime) assert(weakDevice != nil);
        Memory<float> afterDrain(33);
        afterDrain.MutableRawMemory()[0] = 9.5f;
        assert(afterDrain.RawMemory()[0] == 9.5f);
        CheckRuntimeDevices(NSMetal::DefaultDevice());
    }
    assert(deviceResolutionCount.load() == 1);
    assert(g_metal_buffer_live_count.load() == 0 && g_metal_buffer_live_bytes.load() == 0);
    assert(g_metal_buffer_alloc_count.load() == g_metal_buffer_free_count.load());
    std::cout << "PASS " << (lifetime ? "cache lifetime and allocation failure" : "real GPU and offset matmul")
              << ": resolutions=" << deviceResolutionCount.load()
              << ",allocations=" << g_metal_buffer_alloc_count.load()
              << ",reuse=" << g_metal_buffer_reuse_count.load() << "\n";
}
