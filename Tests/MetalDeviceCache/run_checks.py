#!/usr/bin/env python3
"""Run focused Metal checks and the repository's existing C++ test suites."""
import argparse
import io
import json
import pathlib
import re
import subprocess
import tarfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
BASE = "a270e7a5dd239b524fd7d34ad3bb73b646dd7fd6"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=pathlib.Path)
    parser.add_argument("--tsan", action="store_true", help="Also run focused checks under ThreadSanitizer")
    parser.add_argument("--focused-only", action="store_true", help="Skip the audit of existing C++ suites")
    args = parser.parse_args()
    out = args.artifacts.resolve()
    if out == ROOT or ROOT in out.parents:
        parser.error("Test artifacts must be outside the repository")
    out.mkdir(parents=True, exist_ok=True)
    includes = [ROOT, ROOT / "Tests", ROOT / "Tests/DataOpTest", ROOT / "Tests/LayerTest",
                ROOT / "Tests/OtherTests", ROOT / "MetaNN", ROOT / "MetaNN/data/facilities"]
    flags = ["xcrun", "clang++", "-std=c++20", "-O2", "-g", "-pthread", "-fobjc-arc"]
    for path in includes:
        flags += ["-I", str(path)]
    frameworks = ["-framework", "Metal", "-framework", "Foundation", "-framework", "MetalPerformanceShaders"]

    def execute(command, log):
        with (out / log).open("w") as stream:
            result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            print((out / log).read_text()[-12000:], flush=True)
            raise SystemExit(f"Failed ({result.returncode}): {log}")
        print(f"PASS {log}", flush=True)

    for sanitizer in ([False, True] if args.tsan else [False]):
        name = "regression_tsan" if sanitizer else "regression"
        extra = ["-fsanitize=thread", "-fno-omit-frame-pointer"] if sanitizer else []
        execute(flags + extra + ["Tests/MetalDeviceCache/regression.mm", "Tests/MetalDeviceCache/runtime_peer.mm"]
                + frameworks + ["-o", str(out / name)], name + "_build.log")
        for mode in ("unavailable", "lifetime", "normal", "diagnostics"):
            execute([str(out / name)] + ([] if mode == "normal" else [mode]), f"{name}_{mode}.log")
            if mode != "unavailable":
                assert (out / f"{name}_{mode}.log").read_text().count("METAL_CONFIG") == 2

    # Audit the old suites individually: some cannot compile at the base commit.
    # Preserve their sources and compare every failure against an exact snapshot.
    baseline = out / "base_snapshot"
    baseline.mkdir(exist_ok=True)
    archive = subprocess.check_output(["git", "archive", BASE], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as files:
        files.extractall(baseline, filter="data")
    for version, root in (("baseline", baseline), ("optimized", ROOT)):
        version_flags = [str(root / pathlib.Path(f).relative_to(ROOT))
                         if f.startswith(str(ROOT)) else f for f in flags]
        metal = [str(root / p) for p in ("MetaNN/data/facilities/continuous_memory_metal.mm",
                                        "MetaNN/metal/metal_add.mm", "MetaNN/metal/metal_matmul.mm")]
        execute(version_flags + ["-DMETAL_CONFIG_DIAG=0", str(ROOT / "Tests/MetalDeviceCache/numerics.mm")]
                + metal + frameworks + ["-o", str(out / f"numerics_{version}")], f"numerics_{version}_build.log")
        execute([str(out / f"numerics_{version}")], f"numerics_{version}.log")
    if args.focused_only:
        return
    legacy = out / "legacy"
    legacy.mkdir(exist_ok=True)
    sources = [path for suite in ("DataOpTest", "LayerTest", "OtherTests")
               for path in sorted((ROOT / "Tests" / suite).rglob("*.cpp")) if path.name != "main.cpp"]
    audit, objects, entries = [], [], []
    for index, source in enumerate(sources):
        relative = source.relative_to(ROOT)
        obj = legacy / f"unit_{index}.o"
        with (legacy / f"unit_{index}_optimized.log").open("w") as stream:
            result = subprocess.run(flags + ["-ferror-limit=2", "-c", str(source), "-o", str(obj)],
                                    cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        entry = {"source": str(relative), "optimized_compile": result.returncode}
        if result.returncode:
            base_flags = [str(baseline / pathlib.Path(f).relative_to(ROOT))
                          if f.startswith(str(ROOT)) else f for f in flags]
            with (legacy / f"unit_{index}_baseline.log").open("w") as stream:
                before = subprocess.run(base_flags + ["-ferror-limit=2", "-fsyntax-only", str(baseline / relative)],
                                        cwd=baseline, stdout=stream, stderr=subprocess.STDOUT)
            entry["baseline_compile"] = before.returncode
            if before.returncode == 0:
                raise SystemExit(f"New compilation regression: {relative}")
            def errors(version):
                log = (legacy / f"unit_{index}_{version}.log").read_text()
                return [line.split("error:", 1)[1].strip() for line in log.splitlines() if "error:" in line]
            entry["same_errors"] = errors("baseline") == errors("optimized")
            if not entry["same_errors"]:
                raise SystemExit(f"Changed compiler diagnostics: {relative}")
        else:
            objects.append(str(obj))
            text = source.read_text()
            namespaces = list(re.finditer(r"namespace (Test(?:::\w+)*)\s*\{", text))
            if namespaces:
                last = namespaces[-1]
                for match in re.finditer(r"^    void (test_\w+)\(\)", text[last.end():], re.M):
                    entries.append((last.group(1), match.group(1)))
        audit.append(entry)
        if (index + 1) % 10 == 0:
            print(f"Audited {index + 1}/{len(sources)} existing test sources", flush=True)
    (out / "legacy_compile_audit.json").write_text(json.dumps(audit, indent=2))
    driver = legacy / "driver.cpp"
    declarations = "\n".join(f"namespace {ns} {{ void {fn}(); }}" for ns, fn in entries)
    calls = "\n".join(f'if (name == "{ns}::{fn}") {{ {ns}::{fn}(); return 0; }}' for ns, fn in entries)
    driver.write_text('#include <string>\n' + declarations + '\nint main(int argc, char** argv) {\n'
                      'if (argc != 2) return 2; std::string name = argv[1];\n' + calls + '\nreturn 2; }\n')
    results = []
    for version, root in (("baseline", baseline), ("optimized", ROOT)):
        version_flags = [str(root / pathlib.Path(f).relative_to(ROOT))
                         if f.startswith(str(ROOT)) else f for f in flags]
        binary = legacy / version
        metal = [str(root / p) for p in ("MetaNN/data/facilities/continuous_memory_metal.mm",
                                        "MetaNN/metal/metal_add.mm", "MetaNN/metal/metal_matmul.mm")]
        execute(version_flags + ["-DMETAL_CONFIG_DIAG=0", str(driver)] + objects + metal + frameworks
                + ["-o", str(binary)], f"legacy_{version}_link.log")
        for ns, fn in entries:
            name = f"{ns}::{fn}"
            with (legacy / f"{version}_{name.replace('::', '_')}.log").open("w") as stream:
                result = subprocess.run([str(binary), name], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            results.append({"version": version, "test": name, "returncode": result.returncode})
    (out / "legacy_results.json").write_text(json.dumps(results, indent=2))
    for ns, fn in entries:
        matching = [r["returncode"] for r in results if r["test"] == f"{ns}::{fn}"]
        if matching[0] != matching[1]:
            raise SystemExit(f"New runtime regression: {ns}::{fn}: {matching}")
    passed = sum(r["version"] == "optimized" and r["returncode"] == 0 for r in results)
    failed_compile = sum(r["optimized_compile"] != 0 for r in audit)
    print(f"Existing suites: {passed}/{len(entries)} entry points passed on both versions; "
          f"{failed_compile}/{len(sources)} sources fail to compile on both; no new regressions", flush=True)


if __name__ == "__main__":
    main()
