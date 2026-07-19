# Metal process-isolated checkpoint recovery

## Why recovery needs a process boundary

On affected macOS 26.5.x/M3 Ultra systems, Metal can leave a committed command
buffer in `scheduled` state indefinitely while `error == nil`. AdaptiveCpp now
bounds every relevant host wait, but a timeout cannot establish which GPU writes
ran. Replaying the operation or replacing its command queue inside the same
process could therefore duplicate partial particle or field updates.

`scripts/10-run-warpx-resilient.py` uses the last boundary at which correctness
is knowable: a WarpX process that exited successfully after finishing a
checkpoint.

## Recovery protocol

The supervisor divides the requested run into process generations:

1. Start WarpX with a generation-specific `max_step`.
2. Add a dedicated Full diagnostic in checkpoint format while preserving the
   input deck's existing diagnostics.
3. Require WarpX to exit successfully at the generation boundary.
4. Verify that `WarpXHeader` and `Level_0` exist.
5. Atomically write `.warpx-metal-complete.json` inside the checkpoint.
6. Start the next generation with `amr.restart` pointing to that verified
   checkpoint.

Only the external marker makes a checkpoint eligible for automatic resume.
Directories left by a failed or killed child are never promoted. Before retrying
the same generation, they are moved under the supervisor's `quarantine/`
directory for postmortem inspection.

On an AdaptiveCpp Metal completion timeout, or when the child stops producing
log progress past the watchdog deadline, the supervisor terminates the entire
child process group. The retry starts from the last marked checkpoint in a new
process, recreating the Metal device, command queues, shared events, allocations,
and AdaptiveCpp runtime state.

Failures that do not look like a Metal timeout are not retried. This prevents an
invalid input deck or deterministic physics failure from becoming a restart
loop.

## Usage

```bash
./scripts/10-run-warpx-resilient.py \
  --max-step 10000 \
  --chunk-steps 100 \
  --max-retries 3 \
  --command-timeout-seconds 120 \
  --stall-timeout-seconds 300 \
  --work-dir /path/to/run \
  --checkpoint-prefix .warpx-metal-checkpoints/chk \
  /path/to/warpx.3d.NOMPI.SYCL.SP.PSP.EB \
  /path/to/inputs \
  -- \
  my_constants.PPC=4
```

The supervisor owns its checkpoint prefix. Use a distinct prefix if the input
deck already writes checkpoints. Existing diagnostic names are read from the
input file, including `FILE = ...` includes, and retained alongside the recovery
checkpoint diagnostic.

The supervisor automatically resumes the highest checkpoint carrying a valid
completion marker. Logs for every process generation and retry are stored under
`.warpx-metal-supervisor/` in the work directory.

Important controls:

- `--chunk-steps`: smaller values recreate Metal state more frequently and
  reduce lost work, but increase checkpoint and startup overhead.
- `--command-timeout-seconds`: sets `ACPP_METAL_COMMAND_TIMEOUT_MS` for each
  child.
- `--stall-timeout-seconds`: kills a child process group after this much time
  without log growth. Set to `0` only when an external scheduler supplies a
  watchdog.
- `--max-retries`: bounds retries for one generation. Exhaustion exits with
  status 75.
- `--initial-restart`: imports a trusted pre-existing checkpoint; its directory
  name must end with the checkpoint step.
- `--no-resume`: ignores prior completion markers; use it with a fresh
  checkpoint prefix.

Relative paths referenced by the WarpX input retain their normal meaning
relative to `--work-dir`.

## Validation

The deterministic supervisor tests cover:

- following an included input file to preserve diagnostic names;
- a simulated scheduled-command-buffer timeout on the first attempt;
- retry from the last committed generation;
- atomic promotion of only successful checkpoints; and
- immediate failure without retry for a non-Metal error.

An end-to-end Metal run using the real 2D WarpX executable completed in
two-step process generations, produced verified checkpoints at steps 2 and 4,
then a second supervisor invocation discovered step 4 and restarted through
step 6.

This architecture does not repair Apple's driver. It converts the driver defect
into bounded lost work with a correctness-preserving restart, which is the
strongest recovery guarantee available to client code.
