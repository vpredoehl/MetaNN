Run from the checkout on a Mac with an accessible Metal device. C++20 is required
by the existing tensor headers; the allocator-only benchmark also builds as C++17.
All outputs must go outside the repository.

```sh
python3 Tests/MetalDeviceCache/run_checks.py /private/tmp/metann-metal-checks --tsan
python3 Tests/MetalDeviceCache/run_benchmark.py /private/tmp/metann-metal-bench
```

Use `run_checks.py --focused-only` to repeat only the focused tests and the
before/after numerical checks. The focused tests cover concurrent cold start,
float/double allocations and tensor values, shared views and offsets, reuse
across element types, the 4096-byte pool boundary and 256-buffer bucket limit,
device ownership after buffer/autorelease drainage, unavailable-device and
allocation-failure errors, shared runtime accessors in separate translation
units, and configuration diagnostics enabled after earlier suppressed calls.
ThreadSanitizer runs the same focused cases with default diagnostics enabled.

The older test suites contain compilation failures at the specified base commit.
The checker records each source's compile status, verifies failed sources against
an exact base snapshot, and runs all entry points from successful sources against
both original and current Metal implementations. It reports those limitations
separately and fails on a new compile or runtime regression. Four additional
successful source files consist of compile-time assertions, without entry points.

The benchmark uses the original allocator from commit
`a270e7a5dd239b524fd7d34ad3bb73b646dd7fd6`. Both builds use `-O3`, ARC, and identical
allocation workloads. Each worker warms up with 100 allocations. A pilot chooses
iteration counts for approximately 0.4-second samples, followed by seven paired
rounds with alternating version order and shuffled workload order. Initialization
is excluded; worker release and join are included. Different iteration counts
keep both versions measurable. Threads allocate and free their own buffers and
touch both ends; this measures allocation throughput rather than GPU computation.

Timing binaries have no device-call interception, and disable configuration and
buffer log output. Separate counted binaries intercept device resolution and
read the allocator's existing private counters. No instrumentation or diagnostic
API is added to production code. Artifacts include sample timings, medians/ranges,
allocation and reuse counts, device-resolution counts, build metadata, and the
implementation diff. A short counted workload can still miss the pool when its
maximum simultaneous demand exceeds that reached during warmup.
