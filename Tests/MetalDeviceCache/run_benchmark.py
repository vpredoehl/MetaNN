#!/usr/bin/env python3
"""Build original/current allocators and collect paired, isolated GPU samples."""
import argparse
import json
import pathlib
import random
import statistics
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[2]
BASE = "a270e7a5dd239b524fd7d34ad3bb73b646dd7fd6"
CASES = [(256, 1, "float"), (256, 4, "float"), (256, 8, "float"),
         (256, 1, "double"), (256, 4, "double"),
         (4096, 1, "float"), (4096, 4, "float"),
         (16384, 1, "float"), (16384, 4, "float"), (16384, 1, "double")]


def run(command):
    return subprocess.check_output(command, cwd=ROOT, text=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=pathlib.Path)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seconds", type=float, default=0.4)
    args = parser.parse_args()
    artifacts = args.artifacts.resolve()
    if artifacts == ROOT or ROOT in artifacts.parents:
        parser.error("Benchmark artifacts must be outside the repository")
    artifacts.mkdir(parents=True, exist_ok=True)
    original = artifacts / "original_allocator.mm"
    original.write_text(run(["git", "show", f"{BASE}:MetaNN/data/facilities/continuous_memory_metal.mm"]))
    metadata = {"base": BASE, "head": run(["git", "rev-parse", "HEAD"]).strip(),
                "compiler": run(["xcrun", "clang++", "--version"]),
                "os": run(["sw_vers"]), "flags": "-std=c++17 -O3 -fobjc-arc -pthread; diagnostics off",
                "method": "warm 100 allocations per worker; timing excludes initialization; adaptive iterations; paired alternating order; no timing interception",
                "repeats": args.repeats, "target_seconds": args.seconds}
    try:
        metadata["hardware"] = run(["sysctl", "-n", "machdep.cpu.brand_string", "hw.memsize", "hw.ncpu"])
    except subprocess.CalledProcessError:
        metadata["hardware"] = "unavailable"
    (artifacts / "metadata.json").write_text(json.dumps(metadata, indent=2))
    implementation = run(["git", "diff", BASE, "--", "MetaNN/data/facilities/continuous_memory_metal.mm",
                          "MetaNN/metal/metal_add.mm", "MetaNN/metal/metal_matmul.mm", "MetaNN/metal/metal_device.h"])
    tracked_header = subprocess.run(["git", "ls-files", "--error-unmatch", "MetaNN/metal/metal_device.h"],
                                    cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if tracked_header.returncode:
        header = subprocess.run(["git", "diff", "--no-index", "--", "/dev/null", "MetaNN/metal/metal_device.h"],
                                cwd=ROOT, text=True, capture_output=True)
        if header.returncode != 1:
            raise RuntimeError("Failed to capture the new shared device header")
        implementation += header.stdout
    (artifacts / "implementation.diff").write_text(implementation)
    for version in ("baseline", "optimized"):
        for counted in (False, True):
            binary = artifacts / (version + ("_counted" if counted else ""))
            command = ["xcrun", "clang++", "-std=c++17", "-O3", "-fobjc-arc", "-pthread",
                       "-DMETAL_CONFIG_DIAG=0", "-I", str(ROOT), "-I", str(ROOT / "MetaNN/data/facilities")]
            if counted:
                command += ["-DCOUNT_DEVICE_RESOLUTIONS"]
            if version == "baseline":
                command += [f'-DMETANN_ALLOCATOR_SOURCE="{original}"']
            command += [str(ROOT / "Tests/MetalDeviceCache/benchmark.mm"), "-framework", "Metal",
                        "-framework", "Foundation", "-o", str(binary)]
            run(command)

    def sample(version, case, iterations, counted=False):
        size, threads, elem = case
        name = version + ("_counted" if counted else "")
        output = run([str(artifacts / name), str(size), str(threads), str(iterations), elem])
        result = json.loads(output)
        result.update(version=version, iterations_per_thread=iterations)
        assert result["live_after"] == result["live_bytes_after"] == 0
        return result

    iterations = {}
    for case in CASES:
        for version in ("baseline", "optimized"):
            pilot = sample(version, case, 5000)
            iterations[version, case] = max(500, min(10_000_000, round(5000 * args.seconds / pilot["seconds"])))
    rng = random.Random(20261010)
    samples = []
    with (artifacts / "samples.jsonl").open("w") as output:
        for repeat in range(args.repeats):
            cases = CASES.copy()
            rng.shuffle(cases)
            for case in cases:
                order = ("baseline", "optimized") if repeat % 2 == 0 else ("optimized", "baseline")
                for version in order:
                    result = sample(version, case, iterations[version, case])
                    result["repeat"] = repeat
                    samples.append(result)
                    output.write(json.dumps(result) + "\n")
                    output.flush()
            print(f"Completed paired round {repeat + 1}/{args.repeats}", flush=True)
    counters = [sample(version, case, 2000, True) for case in CASES for version in ("baseline", "optimized")]
    (artifacts / "counters.json").write_text(json.dumps(counters, indent=2))
    summary = []
    for size, threads, elem in CASES:
        row = dict(bytes=size, threads=threads, type=elem)
        for version in ("baseline", "optimized"):
            matching = [s for s in samples if (s["bytes"], s["threads"], s["type"], s["version"]) == (size, threads, elem, version)]
            rates = [s["allocations_per_second"] for s in matching]
            row[version] = {"median_per_second": statistics.median(rates),
                            "min_per_second": min(rates), "max_per_second": max(rates),
                            "total_allocations": sum(s["allocations"] for s in matching),
                            "total_new_buffers": sum(s["new_buffers"] for s in matching),
                            "total_reuses": sum(s["reuses"] for s in matching)}
        row["speedup"] = row["optimized"]["median_per_second"] / row["baseline"]["median_per_second"]
        summary.append(row)
    (artifacts / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
