# WarpX on Apple Silicon GPU (Metal)

GPU-accelerated [WarpX](https://github.com/BLAST-WarpX/warpx) electromagnetic PIC simulations on Apple Silicon via the AdaptiveCpp Metal backend.

**Stack:** WarpX → AMReX (SYCL) → AdaptiveCpp → Metal → Apple GPU

**Status:** Phase 1 — AdaptiveCpp Metal backend build and validation

## Prerequisites

- Apple Silicon Mac (M1 or later)
- macOS 14+ (Sonoma)
- Xcode 15+ (`xcode-select --install`)

## Quick Start

```bash
# 1. Install dependencies (LLVM 18, Boost, Ninja, metal-cpp)
./scripts/00-install-deps.sh

# 2. Build AdaptiveCpp with Metal backend
./scripts/01-build-adaptivecpp.sh

# 3. Run SYCL validation tests
./scripts/02-validate-metal.sh
```

## Project Structure

```
├── docs/              # Specification and documentation
├── patches/           # git format-patch files for upstream repos
│   ├── adaptivecpp/
│   ├── amrex/
│   └── warpx/
├── scripts/           # Build and validation scripts
├── tests/sycl/        # SYCL validation tests for Metal backend
├── extern/            # Cloned upstream repos (gitignored)
└── opt/               # Installed software (gitignored)
```

## Key Constraints

- **No FP64:** Apple GPUs have no double-precision support. All builds use single precision.
- **USM:** Metal has unified physical memory but different virtual address spaces for CPU/GPU. USM support via AdaptiveCpp is being validated.
- **Atomics:** Only 32-bit atomics are fully supported on Apple GPUs. Single-precision mode makes this sufficient for WarpX.

## Documentation

- [Technical Specification](docs/spec.md)
- [Known Issues](docs/known-issues.md)

## License

BSD-3-Clause — see [LICENSE](LICENSE)
