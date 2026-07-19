# AdaptiveCpp Metal device-to-host completion hardening

> Historical note: this defensive change did not fix the hang. The later
> [in-order readback correction](adaptivecpp-metal-inorder-readback.md)
> fixes a separate cross-lane ordering defect, removes this shared-event
> callback path, and bounds the wait. Evidence on macOS 26.5.x/M3 Ultra still
> points to an underlying Metal completion-loss issue.

## Summary

The Metal device-to-host staging path now establishes both completion callbacks
before commit, serializes them through an atomic once-gate, and makes queue waits
fail closed when Metal reports a command-buffer error. This is defensive
completion hardening, not an established root-cause fix for the observed long-run
stall.

The causal run produced no missed shared-event notification. The change did not
recover that run because the command buffer itself never completed. Metal shared
event notifications are threshold-based: a listener is notified when the event
value reaches or exceeds its requested value, so the previous commit-before-
listener ordering did not by itself demonstrate an edge-triggered notification
loss.

Patch: [`patches/adaptivecpp/0018-metal-d2h-shared-event-completion.patch`](../patches/adaptivecpp/0018-metal-d2h-shared-event-completion.patch)

## Completion model

The transfer uses two shared-event values:

- `val_blit` indicates that the GPU blit into the staging allocation completed.
- `val_done` indicates that the host copy from staging into the destination is
  complete, or that the transfer failed and the failure was recorded.

The shared-event listener and command-buffer completion handler are both
registered before `commit()`. They are equivalent success contenders: either can
win the compare-and-swap gate. If the listener wins, it first waits for the
command buffer to reach an authoritative terminal status. The winning callback
then checks the command-buffer error before exposing staging bytes.

On success, the winner copies all requested staging ranges to host memory and
publishes `val_done`. The once-gate prevents a second callback from repeating
the copy or signal. A zero-sized transfer naturally skips the copy loops and
still publishes completion.

## Fail-closed error semantics

On command-buffer error, the winner does not copy from the staging buffer. It
stores the first failure in queue-owned shared state, registers the asynchronous
error, and still publishes `val_done` so dependants and queue drains cannot hang
behind an unreachable event value.

`metal_inorder_queue::wait()` snapshots the recorded failure before waiting and
checks again after the completion event is reached. It returns that failure to
the SYCL queue wait path. AMReX synchronizes its SYCL streams with
`wait_and_throw()`, so the registered failure is thrown before reduction code can
consume unchanged or stale destination bytes.

Completion publication therefore means progress, not data validity. Data is
valid only when the queue wait also reports success.

## Scope and validation boundary

The change is confined to AdaptiveCpp's Metal queue header and implementation.
Other AdaptiveCpp backends are not modified.

The patch applies cleanly to the preserved AdaptiveCpp source revision used by
this package. Both `acpp-rt` and the `rt-backend-metal` target that directly
compiles `metal_queue.cpp` compile and link successfully. This proves the code
path is syntactically and link-time compatible with that revision. It does not
reproduce a Metal command-buffer failure, establish a missed-listener root
cause, or correct a command buffer that never reaches completion.
