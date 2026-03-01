# Known Issues — WarpX on Metal

## Phase 1: AdaptiveCpp Metal Backend Validation

### Test Results

| Test | Status | Notes |
|------|--------|-------|
| `device_query.cpp` | **PASS** | Apple M4 Pro detected: 16 CUs, 15974 MB, 1024 max work-group |
| `vector_add.cpp` | **PASS** | 1M-element buffer/accessor parallel_for correct |
| `usm_test.cpp` | **PASS** | `sycl::malloc_shared` works — AMReX compatibility confirmed |
| `reduction_test.cpp` | **PASS** | `atomic_ref<float>` fetch_add correct; multi-cell deposition simulation correct |

### Build Environment

- **macOS:** 26.4 (Tahoe)
- **Xcode:** 26.3
- **LLVM:** 20.1.8 (via Homebrew) — Metal backend requires LLVM 20 APIs
- **LLVM 18:** Also needed for `ld64.lld` (not shipped in LLVM 20 Homebrew bottle)
- **AdaptiveCpp:** develop branch (Metal backend merged Feb 2026)
- **metal-cpp:** macOS 15 / iOS 18 headers (from Apple developer site)

### Build Workarounds

1. **LLVM 20 required (not 18):** AdaptiveCpp develop branch Metal backend uses LLVM 20 APIs (`MemSetPatternInst`, `Intrinsic::scmp/ucmp`, `CmpIntrinsic`). LLVM 18 will not compile it.

2. **ld64.lld from LLVM 18:** The LLVM 20 Homebrew bottle does not include `ld64.lld`. Install `llvm@18` alongside `llvm@20` and pass `-DACPP_LLD_PATH=$(brew --prefix llvm@18)/bin/ld64.lld`.

3. **ACPP_COMPILER_FEATURE_PROFILE=full:** Must be set explicitly. The default (`none`) disables the SSCP compiler, which prevents the `llvm-to-metal` translation library from being built.

4. **CMAKE_OSX_SYSROOT:** Must be set to `$(xcrun --sdk macosx --show-sdk-path)`. Homebrew LLVM does not auto-detect the macOS SDK, causing SSCP libkernel bitcode compilation to fail with a broken `-isysroot` flag.

5. **MSL 3.2 fallback patch:** metal-cpp headers from macOS 15 do not define `MTL::LanguageVersion4_0`. Patch `metal_code_object.cpp` to fall back to `LanguageVersion3_2`. See `patches/adaptivecpp/0001-metal-fallback-msl-3.2-for-older-metal-cpp-headers.patch`.

6. **libomp:** Required by AdaptiveCpp runtime. Must pass explicit OpenMP flags to CMake since `libomp` is keg-only on Homebrew.

### Linker Warnings (Non-blocking)

- `ld: warning: building for macOS-16.0, but linking with dylib ... which was built for newer version 26.0` — LLVM 20 and libomp Homebrew bottles were built for macOS 26 but our deployment target is 16.0. No functional impact observed.

### Implications for Phase 2 (AMReX)

- **USM works:** `sycl::malloc_shared` is functional on Metal. AMReX SYCL backend should work with minimal changes.
- **Atomics work:** `atomic_ref<float>` and `atomic_ref<int>` fetch_add are correct. WarpX current deposition kernels should work in FP32 mode.
- **JIT compilation:** First kernel invocation triggers JIT compilation (LLVM IR → MSL). Subsequent runs use cached binaries. AdaptiveCpp warns about this — expect slower first timestep.

## Phase 2: AMReX on AdaptiveCpp Metal Backend

### Build Configuration

AMReX is built with the AdaptiveCpp `acpp` compiler as the SYCL provider, replacing Intel oneAPI's `icpx`/`dpcpp`. Key CMake flags:

```
-DCMAKE_CXX_COMPILER=acpp
-DAMReX_GPU_BACKEND=SYCL
-DAMReX_PRECISION=SINGLE
-DAMReX_PARTICLES_PRECISION=SINGLE
-DAMReX_SYCL_SUB_GROUP_SIZE=32
-DAMReX_MPI=OFF -DAMReX_OMP=OFF -DAMReX_FORTRAN=OFF
-DAMReX_SYCL_AOT=OFF -DAMReX_SYCL_SPLIT_KERNEL=OFF -DAMReX_SYCL_ONEDPL=OFF
```

### Test Results

| Test | Status | Notes |
|------|--------|-------|
| HeatEquation (16^3, 10 steps) | **PASS** | All 10 timesteps complete on Metal GPU |

### Patches Applied

**1. AMReXSYCL.cmake** (file replacement)
Replaces `Tools/CMake/AMReXSYCL.cmake` to support AdaptiveCpp alongside Intel oneAPI. The original unconditionally adds Intel-specific flags (`-fsycl`, `-qmkl`, `-fsycl-device-lib`, `-mlong-double-64`) that break under `acpp`. The patched version:
- Detects AdaptiveCpp by checking if the CXX compiler basename is `acpp` or `syclcc`
- **AdaptiveCpp path:** Minimal SYCL interface target with `cxx_std_17`, `-Wno-tautological-constant-compare`, and `-DAMREX_NO_INT128`
- **Intel path:** Preserves all original Intel-specific flags unchanged

**2. AMReX_RandomEngine.H / AMReX_Random.cpp** (file replacements)
Guards `oneapi::mkl::rng` includes and usage with `!defined(SYCL_IMPLEMENTATION_ACPP)`. Provides stub RNG types for AdaptiveCpp (GPU RNG not yet supported; falls through to CPU RNG).

**3. AMReX_INT.H** (surgical patch)
Disables `__int128` / `AMREX_INT128_SUPPORTED` when `AMREX_NO_INT128` is defined. The Metal emitter cannot translate `i128` (maps to `uint4` in MSL with unsupported casts). All i128 codepaths (`umulhi`, `FastDivmodU64`) have safe fallbacks.

**4. AMReX_Math.H** (no longer needed — handled by `AMREX_NO_INT128`)
Previously patched `sycl::mul_hi` → UInt128_t fallback, but disabling INT128 entirely removes the function.

**5. AMReX_GpuAsyncArray.H / AMReX_GpuElixir.cpp** (surgical patches)
Replaces `host_task` (SYCL 2020 optional, not in AdaptiveCpp) with synchronous `q.wait()` + direct cleanup.

**6. AMReX_GpuLaunchFunctsG.H / AMReX_GpuLaunchMacrosG.nolint.H / AMReX_TagParallelFor.H / AMReX_FBI.H** (surgical patches)
Removes `[[sycl::reqd_sub_group_size(...)]]` and `[[sycl::reqd_work_group_size(...)]]` attributes. These caused: (a) compile warnings "unknown attribute", (b) runtime `uint4` type cast errors in the Metal emitter. Apple GPUs have fixed 32-thread sub-groups, so these are redundant.

### AdaptiveCpp Metal Emitter Bug Fix

**0003-metal-emit-struct-defs-for-array-elements.patch** — Fixes a bug in `Emitter.cpp` where struct types used as array elements inside other structs were not having their definitions emitted in MSL. The `addDeps` lambda only checked direct `StructType` members but didn't recurse into `ArrayType`. This caused "undeclared identifier" errors for types like `FastDivmodU64` used in `BoxIndexerND::fdm[2]`.

### Metal GPU Constraints

1. **No `double`:** Metal GPUs have zero double-precision support. All builds use `SINGLE` precision. Device code must avoid double literals (`2.` → `Real(2.0)` or `2.0f`).
2. **No `__int128`:** The Metal emitter maps `i128` to `uint4` in MSL but only supports limited cast patterns. Disabled via `AMREX_NO_INT128`.
3. **No `host_task`:** SYCL 2020 `host_task` is not supported by AdaptiveCpp. Replaced with synchronous `q.wait()`.
4. **JIT compilation:** First kernel run triggers LLVM IR → MSL JIT compilation (AdaptiveCpp warns about this). Cached after first run.

---

### Known Limitations (Hardware)

1. **No FP64:** Apple GPUs have zero double-precision support. All builds must use `SINGLE` precision.
2. **32-bit atomics only:** M1/M2 Apple GPUs support only 32-bit atomic operations on buffer pointers. FP32 mode makes this sufficient for WarpX.
3. **No FFT on GPU:** No Metal-native FFT library exists. FDTD solver only (no PSATD spectral solver on GPU).
