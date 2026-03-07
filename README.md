# warpx-metal

Run [WarpX](https://github.com/ECP-WarpX/WarpX) on Apple Silicon GPUs through:

```text
WarpX -> AMReX (SYCL) -> AdaptiveCpp SSCP -> Metal
```

This repo packages the patches, build scripts, validation tests, and benchmark
workflow needed to get that stack working on macOS.

## Status

- End-to-end WarpX runs on Apple GPU through Metal.
- SYCL smoke tests pass on the AdaptiveCpp Metal backend.
- AMReX validation passes on Metal with the included patches.
- WarpX Langmuir tests pass in 2D and 3D on the Metal path.
- Benchmarks on an Apple M4 Pro show GPU wins at larger 3D problem sizes.

The work has been tested on an Apple M4 Pro with 16 GPU compute units.

## Requirements

- Apple Silicon Mac
- macOS 14+
- Xcode 16+ and command-line tools
- Homebrew
- Internet access to clone upstream repos and download `metal-cpp`

The scripts install and use:

- `llvm@20` for the AdaptiveCpp Metal backend build
- `llvm@18` for `ld64.lld`, which is not shipped in the Homebrew `llvm@20` bottle
- `boost`, `cmake`, `ninja`, and `libomp`

## Quick Start

Run the full GPU build and validation flow from the repo root:

```bash
./scripts/00-install-deps.sh
./scripts/01-build-adaptivecpp.sh
./scripts/02-validate-metal.sh
./scripts/03-build-amrex.sh
./scripts/04-validate-amrex.sh
./scripts/05-build-warpx.sh
./scripts/06-validate-warpx.sh
```

What you should expect:

- First GPU runs are slow because AdaptiveCpp JIT-compiles LLVM IR to Metal.
- JIT artifacts are cached under `~/.acpp/apps/global/jit-cache/`.
- Build products and cloned sources live under `opt/` and `extern/`.

## Build Flow

### 1. Install dependencies

[`scripts/00-install-deps.sh`](scripts/00-install-deps.sh) installs Homebrew
packages, checks the Metal toolchain, and downloads `metal-cpp` headers into
`opt/metal-cpp`.

If `xcrun metal` exists but is not functional, the script will point you to:

```bash
xcodebuild -downloadComponent MetalToolchain
```

### 2. Build AdaptiveCpp with Metal enabled

[`scripts/01-build-adaptivecpp.sh`](scripts/01-build-adaptivecpp.sh):

- clones `AdaptiveCpp` into `extern/AdaptiveCpp`
- applies patches from `patches/adaptivecpp/`
- configures CMake with `WITH_METAL_BACKEND=ON`
- installs into `opt/adaptivecpp`

### 3. Validate the SYCL-to-Metal path

[`scripts/02-validate-metal.sh`](scripts/02-validate-metal.sh) compiles and
runs the SYCL smoke tests in `tests/sycl/`:

- `device_query`
- `vector_add`
- `usm_test`
- `reduction_test`

These confirm device discovery, basic kernels, shared USM, and atomics.

### 4. Build and validate AMReX

[`scripts/03-build-amrex.sh`](scripts/03-build-amrex.sh):

- clones `AMReX` into `extern/amrex`
- applies the AMReX patch set and file replacements from `patches/amrex/`
- configures a SYCL build with single precision and MPI/Fortran disabled
- installs into `opt/amrex`

[`scripts/04-validate-amrex.sh`](scripts/04-validate-amrex.sh) builds and runs
the AMReX HeatEquation test from `tests/amrex/heat_equation`.

### 5. Build and validate WarpX

[`scripts/05-build-warpx.sh`](scripts/05-build-warpx.sh):

- clones `WarpX` into `extern/warpx`
- re-applies the Metal-compatible AMReX source changes used by the WarpX subbuild
- configures WarpX with:
  - `WarpX_COMPUTE=SYCL`
  - `WarpX_PRECISION=SINGLE`
  - `WarpX_PARTICLE_PRECISION=SINGLE`
  - `WarpX_MPI=OFF`
  - `WarpX_FFT=OFF`
  - `WarpX_QED=OFF`
  - `WarpX_OPENPMD=OFF`

[`scripts/06-validate-warpx.sh`](scripts/06-validate-warpx.sh) runs reduced
Langmuir tests on the Metal backend:

- 2D: `inputs_test_2d_langmuir_multi`
- 3D: `inputs_test_3d_langmuir_multi`

Logs are written under `tests/warpx/results/`.

## Benchmarking

Build a CPU baseline and compare it to the GPU build:

```bash
./scripts/07-build-warpx-cpu.sh
./scripts/08-benchmark.sh
```

Useful variants:

```bash
./scripts/08-benchmark.sh --quick
./scripts/08-benchmark.sh --gpu-only
./scripts/08-benchmark.sh --cpu-only
```

Current benchmark summary from [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md):

| Test | CPU 12T s/step | GPU s/step | Speedup |
|------|----------------|------------|---------|
| 128x128 2D | 0.0027 | 0.0172 | 0.16x |
| 512x512 2D | 0.0173 | 0.0208 | 0.83x |
| 64^3 3D | 0.0081 | 0.0181 | 0.45x |
| 128^3 3D | 0.0560 | 0.0402 | 1.39x |

On the measured M4 Pro system, the GPU only pulls ahead once the problem is
large enough to amortize Metal submission overhead.

## Profiling

Capture a Metal System Trace with Instruments:

```bash
./scripts/09-profile-metal.sh
./scripts/09-profile-metal.sh langmuir_3d_large
```

This records traces under `benchmarks/profiles/`.

## Limitations

- Apple GPUs do not support FP64, so all builds are single precision.
- `__int128` is disabled for the Metal path.
- `host_task` is not available in AdaptiveCpp here; the AMReX integration uses
  alternate cleanup paths.
- PSATD / FFT-based solvers are disabled. This setup supports the FDTD path.
- The first execution of a new binary pays JIT compilation cost.

## Patch Set

This repo keeps the portability and performance fixes as patches instead of as a
forked monorepo snapshot.

Key patches:

- `patches/adaptivecpp/0008-metal-all-warpx-fixes.patch`
  Fixes Metal codegen issues needed for WarpX correctness, including thread
  address-space handling.
- `patches/adaptivecpp/0009-metal-batch-command-buffer.patch`
  Batches Metal command buffer submission to remove large per-kernel overhead.
- `patches/adaptivecpp/0010-metal-dtoh-fast-path.patch`
  Adds a faster device-to-host path for CPU-accessible USM allocations.
- `patches/amrex/0002-amrex-sscp-atomic-fix.patch`
  Restores correct atomic behavior under AdaptiveCpp SSCP.
- `patches/amrex/0003-amrex-redistribute-no-mpi-sync.patch`
  Removes redundant synchronizations in the no-MPI redistribute path.

The AMReX directory also includes patched replacement files for SYCL CMake and
RNG-related sources.

## Repository Layout

```text
scripts/       Build, validation, benchmark, and profiling entry points
patches/       AdaptiveCpp and AMReX patches plus AMReX replacement files
tests/         SYCL, AMReX, and local WarpX validation inputs
benchmarks/    Benchmark inputs, raw results, and summary report
docs/          Notes, issues, and implementation background
extern/        Cloned upstream source trees (created by scripts)
opt/           Installed toolchains and local build artifacts (created by scripts)
```

## Notes

- [`scripts/env.sh`](scripts/env.sh) defines the shared paths used by all other
  scripts.
- The build scripts reset the cloned `extern/` trees before re-applying patches.
  Treat those directories as generated workspace state.
- More detailed debugging notes and historical context live in
  [`docs/known-issues.md`](docs/known-issues.md).

## License

BSD-3-Clause. See [`LICENSE`](LICENSE).
