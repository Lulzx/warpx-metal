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
- Validation has been run across four Apple Silicon parts: M3 Ultra, M4 Pro,
  M4 Max, and M5 Max (see [`VALIDATION.md`](VALIDATION.md)).
- The current source patch set is synchronized to the field-validated source
  tree, with debug-only instrumentation removed.
- Upstream sources are pinned: AdaptiveCpp `develop@3733a56`, AMReX and WarpX
  at their `26.06` revisions. The patches are generated against those pins and
  the build scripts check them out explicitly.
- Device-side `double` is correct: the Metal emitter lowers FP64 to
  [VF64-metal](https://github.com/Lulzx/VF64-metal) correctly rounded software
  binary64 instead of silently demoting it to `float`, so host-written
  doubles keep their layout and `WarpX_PRECISION=DOUBLE` builds become
  possible at about 13x the FP32 cost.

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

- Runtime Metal JIT compilation and queue submission are serialized to preserve
  SYCL in-order semantics and bound system compiler-service pressure.
- The system-memory guard uses reclaimable file-cache aware macOS memory
  accounting, so cached file data no longer causes a false abort at the default
  guard threshold.
- Device-to-host readback no longer depends on the completion shared-event
  callback cycle. Producer and blit completion are explicit and bounded; a
  Metal command buffer that does not reach a terminal state returns a timeout
  error instead of hanging indefinitely.
- Any remaining host-side shared-event wait is bounded by the same deadline and
  reports the requested and last observed event values when it expires.
- Long jobs can run as checkpointed process generations with
  `scripts/10-run-warpx-resilient.py`. Each generation creates fresh Metal
  device/queue state, and a completion timeout automatically retries from the
  last checkpoint that was verified after a clean child-process exit.
- If the GPU binary keeps wedging, the supervisor can demote the run to a
  CPU build (`--cpu-fallback-executable`). AMReX checkpoints are backend
  independent, so the CPU binary resumes the GPU-written checkpoint directly;
  demotion is permanent for the rest of the run.

The supervisor is the production-safe client workaround for the macOS driver
defect: it never replays uncertain GPU state inside the affected process.
Checkpoint frequency controls the recovery window and I/O overhead.

Example for a 10,000-step 2D run, using 100-step process generations:

```bash
./scripts/10-run-warpx-resilient.py \
  --max-step 10000 \
  --chunk-steps 100 \
  --work-dir /path/to/run \
  --cpu-fallback-executable extern/warpx/build-cpu/bin/warpx.2d.NOMPI.OMP.SP.PSP.EB \
  extern/warpx/build-acpp/bin/warpx.2d.NOMPI.SYCL.SP.PSP.EB \
  /path/to/inputs
```

See [Metal process-isolated recovery](reports/metal-process-isolated-recovery.md)
for recovery semantics and tuning.

## Remaining Notes

- GPU atomic deposition is not expected to be run-to-run bitwise deterministic;
  this is standard for GPU PIC. Validated comparisons use physics observables
  and CPU agreement.
- Particle sorting is enabled (default `sort_intervals = 4`). The earlier
  sort corruption traced to the decoupled-lookback scan's 64-bit atomics,
  which Metal lacks; the Metal path now uses the multipass scan and sorting
  validates cleanly (exact particle counts, energies within atomic-deposition
  tolerance).
- On the single-precision path, parser execution uses host-side momentum
  evaluation for particle injection so GPU and CPU setup match exactly.
- FP64 device code runs through software binary64 (see
  `docs/known-issues.md`, "FP64 on Metal via VF64"). Keep production builds
  in `SINGLE` precision; use `double` for setup-time code only.
- **Open:** parser expressions evaluated on the device (`parse_density_function`,
  parsed external fields) silently inject zero particles because of the
  nested-pointer translation gap; only the momentum parser is worked around.
  See `docs/known-issues.md`, "Device-side parser evaluation injects zero
  particles".

## Bugfix Reports

Detailed reports describe the failure mechanism, correction, validation scope,
and remaining limitations for the keeper fixes:

- [AddPlasma parser-momentum out-of-bounds access](reports/addplasma-parser-momentum-oob.md)
- [AdaptiveCpp Metal device-to-host completion hardening](reports/adaptivecpp-d2h-shared-event-completion.md)
- [AdaptiveCpp Metal in-order readback and bounded completion](reports/adaptivecpp-metal-inorder-readback.md)
- [Metal process-isolated checkpoint recovery](reports/metal-process-isolated-recovery.md)
- [macOS system-memory crashguard accounting](reports/macos-memory-crashguard.md)
- [FP64 on Metal through VF64 software binary64](reports/metal-vf64-double.md)

## Performance

`scripts/08-benchmark.sh` runs the Langmuir cases on the Metal build and the
OpenMP CPU build back to back and writes
[`benchmarks/RESULTS.md`](benchmarks/RESULTS.md). On an M4 Pro (12 CPU
cores, 16 GPU cores) the GPU is behind the 12-thread CPU on small 2D grids
and ahead only at 128^3 in 3D; per-step launch overhead dominates the small
cases. `scripts/09-profile-metal.sh` records a Metal System Trace for
Instruments.

## Requirements

- Apple Silicon Mac
- macOS 15 or newer (validated on macOS 26.4 and 27.0; see
  `docs/known-issues.md` for the macOS 27 SDK / LLVM 20 libc++ workaround)
- Xcode 16 or newer with command-line tools
- Homebrew
- Internet access to clone upstream sources and fetch `metal-cpp`

The scripts use Homebrew packages including `llvm@20`, `llvm@18`, `boost`,
`cmake`, `ninja`, and `libomp`.

## Local CI

Run the pinned host-platform CPU build from the repository root:

```bash
nice -n 15 ./ci/run-local-ci.sh
```

The script explicitly applies the packaged AMReX and WarpX portability patches,
then configures, compiles, and link-checks a non-Metal CPU build. See
[`ci/README.md`](ci/README.md) for dependencies and the precise host-platform
and Linux-coverage boundaries.

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

Optional, for the CPU baseline, benchmarks, and profiling:

```bash
./scripts/07-build-warpx-cpu.sh   # Apple Clang + OpenMP build of the same patched tree
./scripts/08-benchmark.sh         # GPU vs CPU, writes benchmarks/RESULTS.md
./scripts/09-profile-metal.sh     # xctrace Metal System Trace
```

`scripts/env.sh` holds every shared path and toolchain probe; the numbered
scripts source it, and so should any ad-hoc shell that runs `acpp`. Build
products and cloned upstream sources live under `opt/` and `extern/`.
AdaptiveCpp JIT artifacts are cached by the runtime, so the first kernel
launch after a rebuild is slow.

## Documentation

- [`VALIDATION.md`](VALIDATION.md) - validated hardware and dated
  revalidation runs.
- [`docs/known-issues.md`](docs/known-issues.md) - build workarounds, Metal
  constraints, patch inventory, and open defects.
- [`readme-apple-silicon-port-upstream.md`](readme-apple-silicon-port-upstream.md)
  - portability contract for the AMReX and WarpX changes (every shared code
  path is behind an Apple/Metal gate).
- [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md) - GPU vs CPU numbers.
- [`ci/README.md`](ci/README.md) - local CI scope.
- [`docs/spec.md`](docs/spec.md) - the original planning document, kept for
  history.

## Repository Layout

- `patches/adaptivecpp/` - AdaptiveCpp SSCP/Metal source patches (applied in
  order); `tools/` holds the VF64 header generator.
- `patches/amrex/` - AMReX whole-file replacements.
- `patches/amrex-post/` - AMReX source patch applied after the replacements
  and in-place edits.
- `patches/warpx/` - WarpX source patches (source identity, host-side parser
  momentum, macOS memory guard).
- `reports/` - technical bugfix reports and validation boundaries.
- `scripts/` - dependency, build, validation, benchmark, and profiling
  helpers plus the checkpointed supervisor; `scripts/lib/` holds the shared
  AMReX patcher and toolchain shims.
- `tests/sycl/` - AdaptiveCpp/Metal smoke tests, including FP64 and a
  FP64-vs-FP32 throughput benchmark.
- `tests/amrex/` - standalone HeatEquation test used by `04-validate-amrex.sh`.
- `tests/warpx/` - small WarpX input decks (Langmuir, field-only, two-species).
- `tests/metal_direct/` - plain Metal/Swift reproducer for the thread-to-device
  pointer defect, independent of AdaptiveCpp.
- `tests/supervisor/` - unit tests for the process-isolated supervisor, run by
  local CI.
- `benchmarks/` - benchmark input decks, results, and captured profiles.
- `ci/` - pinned CPU portability build.
- `docs/` - known issues and the historical spec.

## License

BSD 3-Clause, see [`LICENSE`](LICENSE). The embedded VF64-metal shader source
in patch 0023 is the author's own work and is covered by the same terms.
