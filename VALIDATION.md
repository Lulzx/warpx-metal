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

## Nondeterminism Note

GPU charge/current deposition uses atomic adds. That means bitwise-identical
run-to-run output is not expected for GPU particle-in-cell deposition. This is
normal for GPU PIC backends that use atomics, including AMReX GPU backends such
as CUDA.

Correctness is therefore judged by conservation, analytic checks, and GPU-to-CPU
observable equivalence, not by bitwise reproducibility of every deposited cell.
The validated observables above are conserved and match CPU within tolerance.
