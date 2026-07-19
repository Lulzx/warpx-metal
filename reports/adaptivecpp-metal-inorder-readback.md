# AdaptiveCpp Metal in-order readback and bounded completion

## Findings

AdaptiveCpp exposed the Metal backend through `multi_queue_executor`. On the
tested M4 Pro this created one memcpy lane and 64 kernel lanes. Consecutive
operations from a SYCL queue could therefore be submitted to different
`metal_inorder_queue` instances:

```text
kernel N   -> kernel lane K
D2H copy N -> memcpy lane 0
kernel N+1 -> kernel lane K+1
```

USM operations did not add DAG requirements between those lanes. The memcpy
lane consequently had no producer command buffer to wait for, and a readback
could observe the previous kernel's value.

A targeted stress test made the ordering bug deterministic enough to observe:
before this fix, repeated kernel/readback pairs returned iteration `N-1` during
iteration `N` after tens of iterations.

That AdaptiveCpp ordering defect is distinct from the completion-loss observed
under sustained particle redistribution on macOS 26.5.x/M3 Ultra. Evidence from
that system indicates a Metal driver/command-queue failure: a committed and
scheduled command buffer can remain non-terminal with `error == nil`, and its
shared event never signals. Client code cannot repair or safely replay such a
buffer because its GPU side effects may be unknown.

## Correction

Patch:
[`patches/adaptivecpp/0019-metal-inorder-d2h-timeout.patch`](../patches/adaptivecpp/0019-metal-inorder-d2h-timeout.patch)

The Metal backend now uses one `inorder_executor`, so kernels and copies from a
SYCL queue share the same Metal command queue. The queue retains its latest
submitted producer command buffer and establishes an explicit completion
boundary before device-to-host readback.

Device-to-host staging no longer uses the
GPU-signal → CPU-listener → GPU-wait shared-event cycle. It commits the blit,
waits for a terminal command-buffer state, and copies staging bytes only after
successful completion.

Producer, blit, and queue-completion waits use retained command-buffer status
instead of an unbounded host shared-event wait. Fine-grained shared-event waits
are also bounded and report both requested and observed values on expiry. The
default timeout is 120 seconds and can be changed with:

```bash
export ACPP_METAL_COMMAND_TIMEOUT_MS=300000
```

If Metal remains committed or scheduled past the deadline, AdaptiveCpp returns
an error that includes the last command-buffer status instead of waiting
forever. A timed-out buffer is never treated as valid readback data.

## Validation

- `rt-backend-metal` compiles and links after the change.
- Existing device query, buffer vector-add, shared-USM, and atomic reduction
  tests pass on Apple M4 Pro.
- `tests/sycl/d2h_stress_test.cpp` completed 10,000 consecutive
  private-device kernel → host-readback iterations with exact expected values.
- Before the single-lane correction, the same stress test reproduced stale
  readback at iterations 46–67.

The bounded timeout prevents an unreported driver/GPU wedge from hanging the
process. It cannot make a genuinely wedged command buffer complete; such a
condition is surfaced as an error so the application can terminate or restart
at a higher level. Sustained particle-redistribution reliability remains
unproven until the workload passes long-duration testing on the affected macOS
26.5.x/M3 Ultra system.
