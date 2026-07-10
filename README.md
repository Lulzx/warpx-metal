# WarpX Metal GPU Backend

This repository packages an Apple-Metal GPU backend path for
[WarpX](https://github.com/ECP-WarpX/WarpX):

```text
WarpX -> AMReX SYCL -> AdaptiveCpp SSCP -> Metal
```

The patch set covers the AdaptiveCpp SSCP/Metal emitter and runtime, AMReX
particle/parser/reduction/base support, and the WarpX source changes needed for
standard PIC workloads on Apple Silicon GPUs.

## Current Status

- WarpX PIC workloads execute on Apple GPUs through AMReX SYCL and AdaptiveCpp
  SSCP-generated Metal.
- Validation has been run across three Apple Silicon generations: M3 Ultra,
  M4 Max, and M5 Max.
- The current source patch set is synchronized to the field-validated source
  tree, with debug-only instrumentation removed.

## Validation

Current validation is field-level, not only run-completion based:

- Langmuir oscillation: GPU and CPU fields (`Ex`, `jx`) agree to about `1e-6`
  at both 1 and 4 particles per cell, and particle counts match.
- Cyclotron gyration and `E x B` drift: full 640-step runs complete with
  particle datasets bit-identical to CPU output; analytic relative error is
  about `1e-4` to `1e-3`.
- Vacuum electromagnetic wave: GPU and CPU agree at relative `Linf` error near
  `1e-6`.
- Convergence and conservation checks show about second-order convergence and
  conserved charge, energy, momentum, and particle count within the validated
  tolerances.

These checks cover both functional agreement and physics-level behavior for the
standard PIC benchmark suite used during validation.

## Reliability

The backend includes reliability controls for heavy runs:

- Runtime Metal JIT compilation is serialized to bound system compiler-service
  memory and process pressure.
- The system-memory guard uses reclaimable file-cache aware macOS memory
  accounting, so cached file data no longer causes a false abort at the default
  guard threshold.

Known limitation: sustained Metal particle-redistribution runs can still stall
stochastically when a command buffer is committed and scheduled but its
completion shared event never signals. Long production runs are not yet reliable;
short runs and Metal validation-layer runs complete.

## Remaining Notes

- GPU atomic deposition is not expected to be run-to-run bitwise deterministic;
  this is standard for GPU PIC. Validated comparisons use physics observables
  and CPU agreement.
- Particle sorting is disabled on the Metal backend pending an upstream sort
  fix.
- On the single-precision path, parser execution uses host-side momentum
  evaluation for particle injection so GPU and CPU setup match exactly.

## Bugfix Reports

Detailed reports describe the failure mechanism, correction, validation scope,
and remaining limitations for the keeper fixes:

- [AddPlasma parser-momentum out-of-bounds access](reports/addplasma-parser-momentum-oob.md)
- [AdaptiveCpp Metal device-to-host completion hardening](reports/adaptivecpp-d2h-shared-event-completion.md)
- [macOS system-memory crashguard accounting](reports/macos-memory-crashguard.md)

## Requirements

- Apple Silicon Mac
- macOS 14 or newer
- Xcode 16 or newer with command-line tools
- Homebrew
- Internet access to clone upstream sources and fetch `metal-cpp`

The scripts use Homebrew packages including `llvm@20`, `llvm@18`, `boost`,
`cmake`, `ninja`, and `libomp`.

## Build Flow

From the repository root:

```bash
./scripts/00-install-deps.sh
./scripts/01-build-adaptivecpp.sh
./scripts/02-validate-metal.sh
./scripts/03-build-amrex.sh
./scripts/04-validate-amrex.sh
./scripts/05-build-warpx.sh
./scripts/06-validate-warpx.sh
```

Build products and cloned upstream sources live under `opt/` and `extern/`.
AdaptiveCpp JIT artifacts are cached by the runtime.

## Repository Layout

- `patches/adaptivecpp/` - AdaptiveCpp SSCP/Metal source patch.
- `patches/amrex/` - AMReX source patch and replacement files used by the build
  scripts.
- `patches/warpx/` - WarpX source patch.
- `reports/` - technical bugfix reports and validation boundaries.
- `scripts/` - dependency, build, and validation helpers.
- `tests/` and `benchmarks/` - small validation inputs and benchmark fixtures.
