# AdaptiveCpp Metal device-to-host completion race

## Summary

The Metal device-to-host staging path could commit a command buffer before its
shared-event listener was registered. A fast command buffer could signal the
intermediate event during that interval. The listener would then miss the
notification, the final completion value would never be published, and a
dependent wait could hang indefinitely.

The keeper patch registers completion paths before commit and funnels them
through an atomic once-gate.

Patch: [`patches/adaptivecpp/0018-metal-d2h-shared-event-completion.patch`](../patches/adaptivecpp/0018-metal-d2h-shared-event-completion.patch)

## Defect mechanism

The transfer uses two shared-event values:

- `val_blit` indicates that the GPU blit into the staging allocation completed.
- `val_done` indicates that the host copy from staging into the destination is
  complete.

Previously, the queue encoded `val_blit`, committed the command buffer, and
only then registered the listener that performs the host copy and publishes
`val_done`. Completion before listener registration created a missed-completion
hang class: the blit was finished, but no callback advanced the event to
`val_done`.

## Correction

The patch introduces a shared `std::atomic<bool>` and a `finish_d2h_once`
closure. The closure uses `compare_exchange_strong` to claim completion once.
The winning successful path copies all requested staging ranges to host memory
and then publishes `val_done`; a later callback returns without repeating the
copy or signal.

Both callback paths are established before `commit()`:

1. The shared-event listener remains the preferred path after `val_blit`.
2. The command-buffer completion handler is a fallback when listener delivery
   is missed.

Either successful callback performs the same copy-then-publish sequence. The
once-gate prevents double completion if both callbacks run. A zero-sized
transfer naturally skips the copy loops and still publishes completion.

## Error-path semantics

If Metal reports a command-buffer error, the completion handler registers that
error and claims the once-gate without copying the staging buffer. It then
publishes `val_done` so queue progress cannot remain blocked forever.

This is an explicit error-versus-hang trade-off. Host destination bytes are not
valid on that path and may remain unchanged or stale. Correct consumers must
observe the registered asynchronous error rather than treat event completion as
proof that data is valid.

## Scope and validation boundary

The change is confined to AdaptiveCpp's Metal queue implementation. Other
AdaptiveCpp backends are not modified.

The patch applies cleanly to the pinned AdaptiveCpp source used by this package,
and the patched `acpp-rt` target builds successfully. The ordering proof closes
the commit-before-listener race and the completion-handler fallback closes the
missed-listener hang class. It does not claim to correct unrelated cases where a
Metal command buffer itself never reaches completion.
