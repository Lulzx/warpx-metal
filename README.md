# WarpX on Apple Silicon GPU (Metal)

GPU-accelerated [WarpX](https://github.com/BLAST-WarpX/warpx) electromagnetic particle-in-cell (PIC) simulations on Apple Silicon via the [AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp) Metal backend.

**Stack:** WarpX → AMReX (SYCL) → AdaptiveCpp SSCP → Metal → Apple GPU

**Hardware:** Apple M-series (validated on M4 Pro)
**Status:** ✅ Full 2D Langmuir wave simulation running on Metal GPU

---

## What This Is

This repo documents and automates building WarpX — a production electromagnetic PIC code used for plasma physics and accelerator design — to run on Apple Silicon GPUs. The path involves:

1. Building [AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp) (formerly hipSYCL) from the `develop` branch, enabling the LLVM SSCP → Metal backend
2. Patching and building [AMReX](https://amrex-codes.github.io/) (the mesh-refinement framework WarpX uses) with the `acpp` SYCL compiler
3. Building WarpX itself against the Metal-enabled AMReX

The result: a WarpX binary that dispatches GPU kernels as Metal shaders on the local Apple GPU, with no CUDA/ROCm required.

---

## Quick Start

```bash
# 1. Install dependencies (LLVM 20, LLVM 18 for ld64.lld, Boost, Ninja, metal-cpp)
./scripts/00-install-deps.sh

# 2. Build AdaptiveCpp with Metal backend (takes ~10 min)
./scripts/01-build-adaptivecpp.sh

# 3. Validate SYCL/Metal backend (device query, vector add, USM, atomics)
./scripts/02-validate-metal.sh

# 4. Build AMReX with SYCL/Metal support (takes ~5 min)
./scripts/03-build-amrex.sh

# 5. Validate AMReX on Metal (HeatEquation test)
./scripts/04-validate-amrex.sh

# 6. Build WarpX against Metal-enabled AMReX (takes ~5 min)
./scripts/05-build-warpx.sh

# 7. Run WarpX validation tests on Metal GPU
./scripts/06-validate-warpx.sh
```

### Prerequisites

- Apple Silicon Mac (M1 or later; validated on M4 Pro)
- macOS 14+ (Sonoma) or macOS 15 (Sequoia)
- Xcode 16+ with command-line tools (`xcode-select --install`)
- Homebrew

---

## Project Structure

```
├── docs/
│   ├── known-issues.md      # Build workarounds, bug fixes, GPU constraints
│   └── spec.md              # Technical specification
├── patches/
│   ├── adaptivecpp/
│   │   └── 0008-metal-all-warpx-fixes.patch   # All Metal emitter bug fixes
│   └── amrex/
│       ├── AMReXSYCL.cmake                    # AdaptiveCpp build system support
│       ├── AMReX_Random{Engine}.{H,cpp}       # RNG stubs for Metal
│       └── 0002-amrex-sscp-atomic-fix.patch   # Fix GPU atomics in SSCP mode
├── scripts/
│   ├── env.sh               # Shared paths (source before other scripts)
│   ├── 00-install-deps.sh   # Homebrew deps
│   ├── 01-build-adaptivecpp.sh
│   ├── 02-validate-metal.sh
│   ├── 03-build-amrex.sh
│   ├── 04-validate-amrex.sh
│   ├── 05-build-warpx.sh
│   └── 06-validate-warpx.sh
├── tests/
│   ├── sycl/                # SYCL validation tests (device_query, vector_add, USM, atomics)
│   ├── amrex/               # AMReX HeatEquation test
│   ├── metal_direct/        # Direct Metal MSL tests for address-space behavior
│   └── warpx/               # WarpX input files (Langmuir wave, two-species)
├── extern/                  # Cloned upstream repos (gitignored)
└── opt/                     # Installed software (gitignored)
```

---

## Validation Results

| Phase | Test | Status |
|-------|------|--------|
| Phase 1 | SYCL device query (M4 Pro: 16 CUs, applegpu_g16s) | ✅ PASS |
| Phase 1 | SYCL vector add (1M elements, buffer/accessor) | ✅ PASS |
| Phase 1 | SYCL USM (`malloc_shared`) | ✅ PASS |
| Phase 1 | SYCL atomics (`atomic_ref<float>` fetch_add) | ✅ PASS |
| Phase 2 | AMReX HeatEquation (10 steps on Metal GPU) | ✅ PASS |
| Phase 3 | WarpX 2D Langmuir wave (65,536 particles/species, 80 steps) | ✅ PASS |
| Phase 3 | WarpX two-species test with particle sort (sort every step) | ✅ PASS |

---

## Key Constraints and Bug Fixes

### Metal GPU Constraints

| Constraint | Impact | Workaround |
|-----------|--------|------------|
| No FP64 | All physics must be single-precision | `-DAMReX_PRECISION=SINGLE -DAMReX_PARTICLES_PRECISION=SINGLE` |
| No `__int128` | AMReX `umulhi`/`FastDivmodU64` fail | `-DAMREX_NO_INT128` |
| 32-bit atomics only | OK for FP32 WarpX deposition | Single-precision sufficient |
| No FFT on GPU | PSATD spectral solver unavailable | FDTD only |
| JIT compilation | First kernel run slow (LLVM IR → MSL) | Cached after first run |

### AdaptiveCpp Metal Emitter Fixes (`patches/adaptivecpp/0008-metal-all-warpx-fixes.patch`)

1. **Array-element struct types** — `addDeps` didn't recurse into `ArrayType`, causing missing MSL struct definitions
2. **AS5 (thread) pointer identity cast** *(Critical)* — `addrspacecast AS5→AS0` must emit `x = x;` not `(device void*)((ulong)x)`. Apple Silicon returns zeros when thread memory is accessed via a device pointer, causing particle bounds checks to fail and all 65,536 particles to be removed with ID=0
3. **PHI cycle address-space resolution** — Loop counter PHI with AS5 edge caused address-space conflict; SENTINEL must return `0` (unknown) not `1` (device)
4. **Thread-parameter IPA** — Two-pass inter-procedural analysis to identify function params that always receive AS5 pointers

### AMReX SSCP Atomic Fix (`patches/amrex/0002-amrex-sscp-atomic-fix.patch`)

AdaptiveCpp SSCP (Single Source Compilation Pipeline) never defines `__SYCL_DEVICE_ONLY__` (it uses a unified host-device pass). AMReX's `Gpu::Atomic::Add` and related wrappers use `AMREX_IF_ON_DEVICE` — which checks `__SYCL_DEVICE_ONLY__` — to dispatch to the atomic `sycl::atomic_ref` path. In SSCP mode this always fell through to a non-atomic `*sum += value` load-modify-store, causing race conditions in `DenseBins::buildGPU` (particle sort), corrupting permutation arrays, and dropping 3/4 of positrons at step 4.

Fix: detect SSCP mode via `__ACPP_ENABLE_LLVM_SSCP_TARGET__` (always injected by the `acpp` compiler script) and call `_device` atomic variants directly.

---

## Architecture Notes

AdaptiveCpp's SSCP (Single Source Compilation Pipeline) compiles SYCL kernels to LLVM IR bitcode at build time. At runtime, the LLVM IR is JIT-translated to Metal Shading Language (MSL), compiled by the system Metal compiler, and cached for subsequent runs. This means:

- **No specialized compiler needed at WarpX build time** — `acpp` handles LLVM IR generation; Metal compilation happens on the target machine
- **Cached after first run** — JIT cache at `~/.acpp/apps/global/jit-cache/`
- **Portable** — same binary runs on any Apple Silicon Mac

---

## License

BSD-3-Clause — see [LICENSE](LICENSE)
