# Known Issues — WarpX on Metal

## Phase 1: AdaptiveCpp Metal Backend Validation

### Test Results

| Test | Status | Notes |
|------|--------|-------|
| `device_query.cpp` | Pending | Metal GPU should appear as SYCL GPU device |
| `vector_add.cpp` | Pending | Buffer/accessor model, no USM dependency |
| `usm_test.cpp` | Pending | **Critical** — AMReX requires `sycl::malloc_shared` |
| `reduction_test.cpp` | Pending | `atomic_ref<float>` fetch_add for current deposition |

### Build Environment

- **macOS:** 26.4 (Tahoe)
- **Xcode:** 26.3
- **LLVM:** 18 (via Homebrew) — system LLVM 21 is incompatible with AdaptiveCpp
- **AdaptiveCpp:** develop branch (Metal backend merged Feb 2026)

### Build Workarounds

_None required yet — update after first build attempt._

### Implications for Phase 2 (AMReX)

- If `usm_test` **passes**: AMReX SYCL backend should work with minimal changes. `sycl::malloc_shared` backed by Metal shared storage mode provides the unified memory AMReX expects.
- If `usm_test` **fails**: AMReX memory arenas need a buffer-based fallback path. This would be a significant refactoring effort — see spec.md §3.2.
- If `reduction_test` **fails**: WarpX current deposition kernels need restructuring to avoid `atomic_ref<float>`. Possible workaround: threadgroup-local accumulation followed by non-atomic writeback.

### Known Limitations (Hardware)

1. **No FP64:** Apple GPUs have zero double-precision support. All builds must use `SINGLE` precision.
2. **32-bit atomics only:** M1/M2 Apple GPUs support only 32-bit atomic operations on buffer pointers. FP32 mode makes this sufficient for WarpX.
3. **No FFT on GPU:** No Metal-native FFT library exists. FDTD solver only (no PSATD spectral solver on GPU).
