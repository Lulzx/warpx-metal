# WarpX on Apple Silicon GPU: Technical Specification

**Project:** GPU-accelerated WarpX for Apple Silicon via AdaptiveCpp Metal backend
**Target Stack:** WarpX → AMReX (SYCL) → AdaptiveCpp (Metal backend) → Apple GPU
**License:** Open source (BSD-3-Clause, matching WarpX/AMReX upstream)
**Status:** Feasibility confirmed — AdaptiveCpp Metal backend merged Feb 2026
**Estimated Effort:** 6–12 person-months

---

## 1. Motivation

WarpX is the leading open-source electromagnetic particle-in-cell (PIC) code, used for laser-plasma interaction, particle accelerator design, and fusion research. It currently supports GPU acceleration on NVIDIA (CUDA), AMD (HIP/ROCm), and Intel (SYCL/oneAPI) hardware. Apple Silicon Macs — with unified memory, high memory bandwidth, and increasingly powerful GPUs — have no path to run WarpX on their GPUs. This project creates that path and open-sources the result.

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────┐
│                    WarpX                         │
│         (PIC algorithms, field solvers)          │
├─────────────────────────────────────────────────┤
│                    AMReX                         │
│  (AMR framework, ParallelFor, memory arenas)     │
│          compiled with -DSYCL backend            │
├─────────────────────────────────────────────────┤
│               AdaptiveCpp (acpp)                 │
│     SSCP compiler: C++ → generic LLVM IR         │
│     Metal runtime: LLVM IR → MSL → GPU dispatch  │
├─────────────────────────────────────────────────┤
│              Apple Metal API                      │
│         MTLComputePipelineState                   │
│         MTLBuffer (shared storage mode)           │
├─────────────────────────────────────────────────┤
│           Apple Silicon GPU                       │
│      (M1/M2/M3/M4 family, TBDR arch)            │
└─────────────────────────────────────────────────┘
```

The key enabler is AdaptiveCpp's SSCP (Single-Source, Single Compiler Pass) flow. The compiler embeds device-independent LLVM IR into the binary. At runtime, the Metal backend JIT-compiles this IR into Metal Shading Language (MSL), creates compute pipelines, and dispatches work to the Apple GPU. This sidesteps the dual-source problem that blocked all previous attempts to use Metal for scientific computing.

## 3. Hardware Constraints and Mitigations

### 3.1 No FP64 (Double Precision)

**Constraint:** Apple GPUs have zero hardware support for 64-bit floating point arithmetic. There is no emulation path — MSL simply doesn't expose `double` for GPU computation.

**Mitigation:**
- WarpX supports compile-time precision selection via `WarpX_PRECISION=SINGLE` and `WarpX_PARTICLE_PRECISION=SINGLE`.
- AMReX's `amrex::Real` and `amrex::ParticleReal` types respect this setting.
- All internal literals use user-defined literals (`0.0_rt`, `0.0_prt`) precisely to enable this.

**Required work:**
- Audit WarpX and AMReX SYCL codepaths for any hardcoded `double` that bypasses the precision typedef. Known locations include some coordinate transforms, reduction operations, and I/O routines.
- Any remaining `double` in host-only code is fine — only device-side code must be FP32.
- Validate physics accuracy: run the WarpX CI test suite in single precision, compare against double-precision CPU reference. Document which test cases require double precision and are therefore excluded.

**Risk:** MEDIUM. Most PIC codes run fine in FP32 for the field solve and particle push. Spectral solvers (PSATD) may accumulate error faster — validate carefully or restrict to FDTD initially.

### 3.2 No Unified Shared Memory (USM)

**Constraint:** Despite Apple Silicon having physically unified memory (CPU and GPU share the same DRAM), Apple's Metal API does not expose Unified Shared Memory in the CUDA/SYCL sense. The GPU has its own TLB and maps buffers at different virtual addresses than the CPU. You cannot dereference a CPU pointer on the GPU or vice versa.

**Mitigation:**
- Metal provides `MTLBuffer` with `MTLResourceStorageModeShared`, which allows both CPU and GPU to access the same physical memory through *different* virtual addresses. No explicit data transfer is needed — only synchronization.
- AdaptiveCpp's Metal backend must map SYCL buffer/accessor patterns to shared `MTLBuffer` objects.
- For USM-style allocations that AMReX's SYCL backend uses (`sycl::malloc_shared`, `sycl::malloc_device`), AdaptiveCpp's Metal backend needs to back these with `MTLBuffer` shared storage and handle pointer translation.

**Required work:**
- Determine what AdaptiveCpp's Metal backend currently implements for USM. If `sycl::malloc_shared` is not yet supported, implement it using `MTLBuffer` with shared storage mode.
- Modify AMReX's memory arena system to use Metal-compatible allocation. Create an `ArenaAllocator_Metal` that:
  - Uses `MTLBuffer` shared storage for all allocations
  - Returns the CPU-visible pointer for host access
  - Provides the GPU-visible pointer (via `[buffer contents]`) for kernel arguments
  - Handles synchronization via `MTLCommandBuffer` completion handlers
- AMReX's `The_Arena()`, `The_Device_Arena()`, `The_Pinned_Arena()`, and `The_Managed_Arena()` can all map to the same shared `MTLBuffer` pool on Apple Silicon — the hardware has no concept of separate device/host memory.

**Risk:** HIGH. This is the single most likely phase-2 blocker. AMReX's SYCL backend was developed against Intel's USM implementation. If AdaptiveCpp's Metal backend cannot emulate USM semantics sufficiently, AMReX's memory management will need significant refactoring.

### 3.3 Limited Atomic Operations

**Constraint:** Apple GPUs (M1 / Apple7 family) support only 32-bit atomics on buffer pointers. 64-bit atomics are limited: M2+ (Apple8+) has a single non-returning `UInt64` max/min instruction but not the full `cl_khr_int64_base_atomics` set. There are no texture atomics on any Apple GPU generation.

**Mitigation:**
- WarpX's current deposition (particles → grid) and field gather use atomics for thread safety. These are 32-bit `float` atomics when running in single precision — which Apple GPUs support.
- AMReX's internal operations (e.g., reductions) use atomics. Verify these are 32-bit compatible in FP32 mode.
- If any 64-bit atomics are required (e.g., for particle ID generation or some reduction paths), implement a CAS-based emulation loop using 32-bit atomics.

**Required work:**
- Grep AMReX SYCL backend for `atomic<int64_t>`, `atomic<uint64_t>`, `atomic<double>`. Catalog all uses.
- Implement fallback CAS emulation for any required 64-bit atomic operations.
- Performance-test atomic-heavy kernels (current deposition) on Apple Silicon to check for bottlenecks — Apple's TBDR architecture may have different atomic throughput characteristics than immediate-mode GPUs.

**Risk:** LOW for FP32 mode. Most atomic usage is float or int32.

### 3.4 TBDR Architecture

**Constraint:** Apple GPUs use a Tile-Based Deferred Rendering (TBDR) architecture, fundamentally different from NVIDIA/AMD immediate-mode compute architectures. Compute shaders bypass the tiling/rendering pipeline, but the GPU core structure still differs: Apple GPU cores are wider (32 threads per SIMD group), have different register file sizes, and different memory hierarchy (tile memory, threadgroup memory, device memory).

**Mitigation:**
- For compute workloads (which is what WarpX uses), TBDR vs IMR matters less than for graphics. Compute dispatches go through the same `dispatchThreadgroups` path regardless.
- However, optimal threadgroup sizes, occupancy, and memory access patterns differ.

**Required work:**
- After functional correctness is achieved, profile with Xcode Metal System Trace and Metal GPU Profiler.
- Tune AMReX's `ParallelFor` launch parameters for Apple GPU occupancy. Current defaults are tuned for NVIDIA (256 threads/block) — Apple GPUs prefer threadgroup sizes that are multiples of 32 (SIMD width) and benefit from 256–1024 threads per threadgroup for compute.
- Investigate tile memory / threadgroup memory for particle sorting and current deposition scratch buffers.

**Risk:** LOW for correctness, MEDIUM for performance. Initial runs will work but may underperform. This is Phase 4 optimization work.

## 4. Phase 1: AdaptiveCpp Metal Backend Stabilization

### 4.1 Objective

Build AdaptiveCpp from source with Metal backend enabled. Run the AdaptiveCpp test suite and SYCL conformance tests on Apple Silicon. Identify and fix blocking bugs.

### 4.2 Prerequisites

- macOS 14+ (Sonoma or newer) on Apple Silicon (M1 or later)
- Xcode 15+ (provides Metal compiler toolchain, `metal` and `metallib` CLI tools)
- LLVM 17–19 (AdaptiveCpp links against LLVM for JIT)
- Boost >= 1.69 (Boost.Fiber, Boost.Context)
- CMake 3.24+

### 4.3 Build Procedure (Expected)

```bash
# Install LLVM (AdaptiveCpp needs LLVM libraries for JIT)
brew install llvm@18

# Clone AdaptiveCpp develop branch (contains Metal backend)
git clone --branch develop https://github.com/AdaptiveCpp/AdaptiveCpp.git
cd AdaptiveCpp
mkdir build && cd build

# Configure with Metal backend
# The exact CMake flags will depend on how resetius's Metal backend
# is integrated — check CMakeLists.txt and doc/installing.md
cmake .. \
  -DCMAKE_INSTALL_PREFIX=$HOME/opt/adaptivecpp \
  -DCMAKE_C_COMPILER=$(brew --prefix llvm@18)/bin/clang \
  -DCMAKE_CXX_COMPILER=$(brew --prefix llvm@18)/bin/clang++ \
  -DLLVM_DIR=$(brew --prefix llvm@18)/lib/cmake/llvm \
  -DWITH_METAL_BACKEND=ON \
  -DWITH_CPU_BACKEND=ON

make -j$(sysctl -n hw.ncpu)
make install
```

### 4.4 Validation Targets

Run these in order of increasing complexity:

1. **Vector add kernel** — basic ParallelFor, no atomics, no USM
2. **BabelStream** — standard HPC memory bandwidth benchmark, tests basic SYCL buffer operations
3. **Reduction** — tests atomic operations and workgroup-level synchronization
4. **USM allocation test** — `sycl::malloc_shared`, `sycl::malloc_device`, pointer arithmetic
5. **nd_range kernel with barriers** — tests workgroup synchronization

### 4.5 Expected Issues to Fix

- Metal shader compilation errors from unsupported LLVM IR constructs
- Missing SYCL 2020 features (sub-groups, specific atomic operations)
- Memory management bugs (buffer lifetime, synchronization)
- JIT cache invalidation issues

### 4.6 Deliverable

A stable AdaptiveCpp installation where `acpp-info` reports a Metal device and basic SYCL programs execute correctly. All fixes upstreamed to AdaptiveCpp.

## 5. Phase 2: AMReX on AdaptiveCpp/Metal

### 5.1 Objective

Compile and run AMReX's test suite and tutorials using AdaptiveCpp's Metal backend as the SYCL provider.

### 5.2 Starting Point

AMReX already has a SYCL backend (`-DAMReX_GPU_BACKEND=SYCL`). It was developed for Intel's DPC++ (icpx). The farscape-project demonstrated that AMReX's SYCL can target AMD/NVIDIA via AdaptiveCpp plug-in patches. We follow the same strategy for Metal.

### 5.3 Build Procedure (Expected)

```bash
git clone https://github.com/AMReX-Codes/amrex.git
cd amrex
mkdir build && cd build

# Use AdaptiveCpp's compiler wrapper
export PATH=$HOME/opt/adaptivecpp/bin:$PATH

cmake .. \
  -DCMAKE_CXX_COMPILER=acpp \
  -DAMReX_GPU_BACKEND=SYCL \
  -DAMReX_MPI=OFF \
  -DAMReX_OMP=OFF \
  -DAMReX_PRECISION=SINGLE \
  -DAMReX_PARTICLES_PRECISION=SINGLE \
  -DCMAKE_BUILD_TYPE=Release

make -j$(sysctl -n hw.ncpu)
```

### 5.4 Known Modifications Required

#### 5.4.1 Memory Arena Adaptation

AMReX's SYCL backend allocates memory via:
- `sycl::malloc_device()` — device-only memory
- `sycl::malloc_shared()` — USM shared memory
- `sycl::malloc_host()` — host-pinned memory

On Metal, all three should map to `MTLBuffer` with `MTLResourceStorageModeShared`. Create a preprocessor-guarded path:

```cpp
// In AMReX_Arena.cpp, add Metal-specific arena
#ifdef AMREX_USE_SYCL
#if defined(ACPP_METAL_BACKEND)
    // All arenas use shared Metal buffers on Apple Silicon
    // No separate device/host distinction needed
    void* metal_alloc_shared(std::size_t sz, sycl::queue& q) {
        return sycl::malloc_shared(sz, q);
        // AdaptiveCpp Metal backend implements this via MTLBuffer shared
    }
#endif
#endif
```

The ideal outcome is that AdaptiveCpp's Metal backend transparently handles `sycl::malloc_shared` using Metal shared buffers, requiring zero AMReX changes. The fallback is explicit Metal interop code.

#### 5.4.2 ParallelFor Launch Configuration

AMReX's `ParallelFor` maps to `sycl::parallel_for` with a `sycl::range` or `sycl::nd_range`. The work-group size defaults need adjustment for Apple GPUs.

In `AMReX_GpuLaunch.H`:
```cpp
#if defined(ACPP_METAL_BACKEND)
    // Apple GPU SIMD width is 32; prefer larger threadgroups for occupancy
    constexpr int AMREX_GPU_MAX_THREADS = 256;
    constexpr int AMREX_GPU_WARP_SIZE = 32;
#endif
```

#### 5.4.3 Atomic Operation Compatibility

Grep and patch:
```bash
# Find all atomic usage in AMReX SYCL paths
grep -rn "atomic_ref\|atomic_fetch\|atomic_compare" Src/ --include="*.H" --include="*.cpp"
```

Replace any `sycl::atomic_ref<double, ...>` with `sycl::atomic_ref<float, ...>` guarded by precision typedef. For integer atomics, verify they're 32-bit.

#### 5.4.4 Sub-group Operations

AMReX's SYCL backend may use `sycl::sub_group` for reductions and scans. Verify AdaptiveCpp's Metal backend supports sub-group operations (which map to SIMD-group operations in Metal). If not, provide fallback implementations using threadgroup-level operations.

### 5.5 Validation Targets

Run in order:

1. **AMReX HeatEquation tutorial** — simplest GPU test, single MultiFab, explicit stencil
2. **AMReX Advection tutorial** — tests ghost cell filling and MFIter
3. **AMReX ElectromagneticPIC tutorial** — this is the direct precursor to WarpX. If this runs, WarpX will likely work. Tests particles, field deposition, current deposition, and the field solve.
4. **AMReX regression test suite** — `make test` on the full AMReX test suite

### 5.6 Deliverable

AMReX ElectromagneticPIC tutorial running correctly on Apple GPU with single-precision output matching CPU reference within FP32 tolerance. Patches organized as clean commits suitable for upstream PR.

## 6. Phase 3: WarpX on Metal

### 6.1 Objective

Build and run WarpX electromagnetic PIC simulations on Apple Silicon GPU.

### 6.2 Build Procedure

```bash
git clone https://github.com/BLAST-WarpX/warpx.git
cd warpx
mkdir build && cd build

cmake -S .. -B . \
  -DCMAKE_CXX_COMPILER=acpp \
  -DWarpX_COMPUTE=SYCL \
  -DWarpX_PRECISION=SINGLE \
  -DWarpX_PARTICLE_PRECISION=SINGLE \
  -DWarpX_DIMS="2;3" \
  -DWarpX_MPI=OFF \
  -DWarpX_FFT=OFF \
  -DWarpX_QED=OFF \
  -DCMAKE_BUILD_TYPE=Release

cmake --build . -j $(sysctl -n hw.ncpu)
```

Note: FFT is disabled initially (see 6.3.1). MPI is disabled for single-node development. QED module is disabled to reduce surface area.

### 6.3 Known Challenges

#### 6.3.1 FFT / Spectral Solver

WarpX's PSATD (Pseudo-Spectral Analytical Time-Domain) solver requires FFT. On GPU, this uses cuFFT (NVIDIA), rocFFT (AMD), or oneMKL (Intel). None of these exist for Metal.

**Options (ranked by feasibility):**

1. **Skip PSATD, use FDTD only** — The finite-difference solver requires no FFT. This is the pragmatic first target. Most WarpX users use FDTD for production runs anyway.

2. **Use Apple's Accelerate/vDSP on CPU** — Run FFTs on CPU via `vDSP_fft_zripD`, transfer results to GPU. Latency will be high but the memory is physically shared, so bandwidth is not a bottleneck. Viable for development but not competitive for production.

3. **Implement Metal compute shader FFT** — Write a high-performance FFT in MSL. This is substantial work (Cooley-Tukey, Stockham, or Bluestein depending on sizes). Libraries like Metal Performance Shaders do not expose general-purpose complex FFT. This is Phase 5+ work.

4. **Port VkFFT to Metal** — VkFFT is an open-source GPU FFT library supporting Vulkan, CUDA, HIP, and OpenCL. Adding a Metal backend to VkFFT, then integrating with AMReX's FFT interface, is a large but well-defined project. This is the best long-term solution.

**Decision:** Start with FDTD (option 1). Implement option 2 as a fallback for users who need spectral. Pursue option 4 as a separate workstream if there's demand.

#### 6.3.2 Particle Sorting

WarpX sorts particles by cell for cache-efficient deposition. AMReX uses `thrust::sort` on CUDA or equivalent on other backends. On SYCL, this uses `oneapi::dpl::sort`. Verify that AdaptiveCpp provides a sorting primitive or implement a GPU radix sort in SYCL.

#### 6.3.3 Boundary Conditions and Communication

With `WarpX_MPI=OFF`, boundary conditions are periodic or absorbing, handled locally. Once single-GPU is working, enabling MPI requires:
- Verifying that MPI can access Metal shared buffer memory directly (MPI implementations on macOS may not be CUDA-aware, but shared `MTLBuffer` memory is regular virtual memory accessible from CPU, so standard MPI should work without GPU-aware MPI).
- Testing with Open MPI from Homebrew.

### 6.4 Validation Targets

WarpX has a comprehensive CI test suite. Run these benchmarks in single precision:

| Test | Physics | What it validates |
|------|---------|-------------------|
| `Langmuir_2d` | Electrostatic oscillation | Basic particle push + field solve |
| `Langmuir_multi_2d` | Multi-species | Species handling on GPU |
| `LaserAcceleration_2d` | Laser-plasma | Full EM PIC loop, deposition, gather |
| `PlasmaAcceleration_3d` | Beam-driven wakefield | 3D performance, large particle count |
| `uniform_plasma_3d` | Uniform plasma | Load balancing, GPU memory management |

For each test: run on CPU (OpenMP), run on Apple GPU (Metal), compare field and particle diagnostics within FP32 tolerance (relative error < 1e-5 for fields, < 1e-3 for particle statistics).

### 6.5 Deliverable

WarpX 2D and 3D FDTD simulations running on Apple Silicon GPU, with physics validation against CPU reference. Functional PICMI Python interface for interactive use.

## 7. Phase 4: Optimization and Benchmarking

### 7.1 Profiling Tools

- **Xcode Metal System Trace** — GPU timeline, kernel execution time, occupancy
- **Xcode Metal Debugger** — per-kernel analysis, register pressure, threadgroup memory usage
- **Metal Performance HUD** — real-time GPU utilization overlay
- **Custom timing** — AMReX has built-in `BL_PROFILE` timers; enable and analyze

### 7.2 Optimization Targets

#### 7.2.1 Current Deposition (Highest Priority)

Current deposition (J += qv * S, where S is the shape function) is the single hottest kernel in WarpX. It involves:
- Random memory access (particle positions map to grid cells)
- Atomic float additions to shared grid
- High arithmetic intensity per particle

Optimizations for Apple GPU:
- Use threadgroup memory for local accumulation before global atomic write-back
- Experiment with particle tiling / binning to improve spatial locality
- Profile SIMD-group reductions vs atomic adds

#### 7.2.2 Field Solve (Second Priority)

The FDTD Yee solver is a structured stencil — high memory bandwidth demand, low arithmetic intensity. Apple Silicon's memory bandwidth (M3 Max: 400 GB/s, M4 Max: 546 GB/s) should map well.

Optimizations:
- Ensure coalesced memory access patterns in AMReX's ParallelFor traversal order
- Test impact of `MTLResourceStorageModeShared` vs `MTLResourceStorageModePrivate` for read-only field data

#### 7.2.3 Particle Push (Third Priority)

Boris push is embarrassingly parallel with no data dependencies between particles. Should scale linearly with GPU cores. Verify this is the case; if not, investigate register pressure or memory access patterns.

### 7.3 Benchmark Matrix

Measure and report wall-clock time per step for:

| Configuration | Mesh Size | Particles/cell | Metric |
|--------------|-----------|-----------------|--------|
| M3 Max GPU (40 cores) | 256³ | 8 | steps/sec |
| M3 Max GPU (40 cores) | 512³ | 8 | steps/sec |
| M4 Max GPU (40 cores) | 256³ | 8 | steps/sec |
| M3 Max CPU (16 cores, OpenMP) | 256³ | 8 | steps/sec |
| Reference: A100 (CUDA) | 256³ | 8 | steps/sec |

The goal is not to beat an A100. The goal is to demonstrate meaningful speedup over CPU-only execution on the same machine, and to establish where Apple Silicon GPU sits in the performance landscape.

### 7.4 Deliverable

Published benchmark results. Performance analysis document identifying bottlenecks and Apple GPU-specific optimization strategies. Upstream performance patches.

## 8. Repository and Release Plan

### 8.1 Repository Structure

```
warpx-metal/
├── README.md                 # Project overview, build instructions
├── LICENSE                   # BSD-3-Clause
├── docs/
│   ├── spec.md              # This document
│   ├── build-guide.md       # Step-by-step build on macOS
│   ├── benchmarks.md        # Performance results
│   └── known-issues.md      # FP64, FFT, and other limitations
├── patches/
│   ├── adaptivecpp/         # Patches/fixes for AdaptiveCpp Metal backend
│   ├── amrex/               # AMReX Metal compatibility patches
│   └── warpx/               # WarpX build system and Metal-specific fixes
├── scripts/
│   ├── build-all.sh         # One-click build script
│   ├── run-tests.sh         # Validation test runner
│   └── benchmark.sh         # Performance benchmark suite
├── examples/
│   ├── langmuir_2d/         # Minimal physics example with input deck
│   ├── laser_wakefield_2d/  # Laser-plasma acceleration example
│   └── uniform_plasma_3d/   # 3D scaling benchmark
└── spack/
    └── spack-macos-metal.yaml  # Spack environment for dependency management
```

### 8.2 Upstream Strategy

The long-term goal is zero patches — everything merged upstream:

| Component | Upstream Target | Strategy |
|-----------|----------------|----------|
| AdaptiveCpp Metal fixes | AdaptiveCpp/AdaptiveCpp | Direct PR to `develop` branch |
| AMReX Metal support | AMReX-Codes/amrex | PR after ElectromagneticPIC tutorial passes |
| WarpX Metal CI | BLAST-WarpX/warpx | PR after physics validation passes |

Until upstream accepts, maintain a thin patch layer. Use `git format-patch` for clean, rebuildable patch sets.

### 8.3 Release Milestones

| Milestone | Criteria | Target |
|-----------|----------|--------|
| M0: Environment | AdaptiveCpp + Metal builds, vector-add runs | Month 1 |
| M1: AMReX basic | HeatEquation tutorial on Apple GPU | Month 2–3 |
| M2: AMReX PIC | ElectromagneticPIC tutorial on Apple GPU | Month 3–5 |
| M3: WarpX FDTD | Langmuir 2D passes physics validation | Month 5–7 |
| M4: WarpX 3D | 3D simulations, Python interface | Month 7–9 |
| M5: Optimized | Benchmarks published, upstream PRs filed | Month 9–12 |

## 9. Dependencies and Minimum Versions

| Dependency | Minimum Version | Notes |
|------------|----------------|-------|
| macOS | 14.0 (Sonoma) | Metal 3 API required |
| Xcode | 15.0 | Metal compiler toolchain |
| Apple Silicon | M1 | All M-series supported |
| LLVM | 17 | For AdaptiveCpp JIT |
| AdaptiveCpp | develop branch (Feb 2026+) | Metal backend |
| Boost | 1.74 | Fiber, Context |
| CMake | 3.24 | AMReX/WarpX requirement |
| Python | 3.9+ | PICMI interface |

## 10. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| AdaptiveCpp Metal backend too unstable | Medium | Critical | Budget Phase 1 time generously; contribute fixes upstream; maintain fork if needed |
| AMReX USM dependency blocks Metal port | High | Critical | Implement buffer-based fallback; propose AMReX refactor to decouple from USM |
| FP32 insufficient for some physics | Low | Medium | Document limitations; keep FDTD focus; let users fall back to CPU for FP64 |
| Apple changes Metal API | Low | Low | Target stable Metal 3 API; avoid private/undocumented features |
| Performance uncompetitive vs CPU | Medium | Medium | Apple GPU memory bandwidth is excellent; if compute-bound, profile and optimize; some speedup over CPU is expected even without heroic optimization |
| AdaptiveCpp Metal maintainer abandons project | Medium | High | Engage with resetius early; offer to co-maintain; worst case, fork and maintain Metal backend independently |

## 11. Open Questions

1. **Does AdaptiveCpp's Metal backend support `sycl::malloc_shared`?** This determines how much AMReX memory management code needs modification. Must be answered in Phase 1.

2. **What SYCL 2020 features are missing from the Metal backend?** Sub-groups, group algorithms, extended atomics — catalog the gaps against AMReX's usage.

3. **Can MPI on macOS (Open MPI / MPICH) directly access Metal shared buffer memory for send/recv?** If yes, multi-GPU (multi-socket Mac Pro, or multi-process on single GPU) is straightforward. If no, explicit staging buffers are needed.

4. **What is the maximum `MTLBuffer` size on current Apple Silicon?** M-series GPUs with unified memory should allow buffers up to the physical RAM limit, but verify against Metal API documentation.

5. **Does the Metal JIT compilation add measurable overhead per timestep?** AdaptiveCpp caches compiled kernels, but the first invocation may be slow. Measure cold-start and warm-start times.

## 12. Contact and Collaboration

- **AdaptiveCpp Metal backend:** resetius (GitHub) — author of the Metal backend PR
- **AdaptiveCpp maintainer:** Aksel Alpay (illuhad, Heidelberg University)
- **AMReX SYCL:** Weiqun Zhang, Andrew Myers (AMReX team, LBNL)
- **WarpX:** Axel Huebl, Jean-Luc Vay (WarpX team, LBNL)
- **AMReX SYCL plug-in (farscape):** Joe Todd (farscape-project)
- **Apple GPU microarchitecture:** Philip Turner (metal-benchmarks, original Metal backend tracker)

Engage all of these early. File issues, join discussions, offer PRs. This project sits at the intersection of multiple open-source communities and succeeds or fails based on collaboration.
