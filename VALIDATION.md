# Validation

This branch publishes an Apple-Metal GPU backend path for WarpX:

```text
WarpX -> AMReX (SYCL) -> AdaptiveCpp SSCP -> Metal
```

## Build Path

Run the build flow from the repository root:

```bash
./scripts/00-install-deps.sh
./scripts/01-build-adaptivecpp.sh
./scripts/02-validate-metal.sh
./scripts/03-build-amrex.sh
./scripts/04-validate-amrex.sh
./scripts/05-build-warpx.sh
./scripts/06-validate-warpx.sh
```

The scripts clone upstream source trees into `extern/`, apply the patch sets in
`patches/`, and install local build products into `opt/`.

## Validated Results

Validation was performed on Apple M3 Ultra, Apple M4 Max, and Apple M5 Max
systems.

- Deterministic GPU-to-CPU crosstest: maximum relative error `1.68e-6` over 100 steps.
- Statistical RNG deck: moment differences around `1e-18`.
- Charge conservation: `2.67e-8` vs CPU.
- Field energy: `1.81e-6` vs CPU.
- Particle energy: `3.25e-7` vs CPU.
- Particle count: exact over 100 steps.
- x/y momentum: exact over 100 steps.
- On-chip GPU-to-CPU equivalence: validated on all three systems.

## Post-Merge Revalidation (Apple M4 Pro, 2026-07-19)

The full stack was rebuilt from a clean state after merging PR #1 with the
review follow-up fixes (AdaptiveCpp pinned to develop@`3733a56`, AMReX/WarpX
pinned to their 26.06 revisions, MSL 3.2 fallback restored, install prefix
wiped before reinstall, corrected 64-bit atomic emulation, multipass SYCL
scan with the two-word `BlockStatus` on Metal, and the reworked memory
guard). Results on Apple M4 Pro (12 CPU cores, 16 GPU cores, 24 GB):

- SYCL smoke tests (`02-validate-metal.sh`): 4/4 pass
  (device query, vector add, USM, reduction).
- AMReX HeatEquation on Metal GPU (`04-validate-amrex.sh`): PASS.
- **WarpX Langmuir 2D: 40/40 steps complete on Metal GPU.**
- **WarpX Langmuir 3D: 20/20 steps complete on Metal GPU.**

Two silent-corruption classes were found and eliminated during this
revalidation:

1. A stale `libllvm-to-metal.dylib` from an older install layout paired an
   old JIT emitter with the new runtime, making every kernel a silent no-op.
   The build scripts now wipe the install prefix before reinstalling.
2. `AMREX_SYCL_NO_MULTIPASS_SCAN` forced AMReX's decoupled-lookback scan,
   whose packed `BlockStatus` requires true 64-bit atomic exchange/load that
   Metal does not provide; the previous emulation touched only the low 32
   bits, silently corrupting prefix sums used by particle redistribution and
   sorting. The Metal path now uses the multipass scan, and the lookback
   kernel (single-block only) uses the fence-based two-word `BlockStatus`.

## Nondeterminism Note

GPU charge/current deposition uses atomic adds. That means bitwise-identical
run-to-run output is not expected for GPU particle-in-cell deposition. This is
normal for GPU PIC backends that use atomics, including AMReX GPU backends such
as CUDA.

Correctness is therefore judged by conservation, analytic checks, and GPU-to-CPU
observable equivalence, not by bitwise reproducibility of every deposited cell.
The validated observables above are conserved and match CPU within tolerance.
