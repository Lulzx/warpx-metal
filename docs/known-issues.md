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

### Known Limitations (Hardware)

1. **No FP64:** Apple GPUs have zero double-precision support. All builds must use `SINGLE` precision.
2. **32-bit atomics only:** M1/M2 Apple GPUs support only 32-bit atomic operations on buffer pointers. FP32 mode makes this sufficient for WarpX.
3. **No FFT on GPU:** No Metal-native FFT library exists. FDTD solver only (no PSATD spectral solver on GPU).
