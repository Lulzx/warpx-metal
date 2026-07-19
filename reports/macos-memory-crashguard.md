# macOS system-memory crashguard accounting

## Summary

The original macOS guard treated only `free_count` pages as available. That
made reclaimable file cache appear permanently used and could abort a healthy
run even though macOS could release cached pages under pressure.

The keeper patch derives an overlap-safe reclaimable-memory estimate, uses the
host-reported page size, and adds independent pressure and sustained
compressor/swap gates.

Patch: [`patches/warpx/0003-correct-macos-memory-guard-available-memory.patch`](../patches/warpx/0003-correct-macos-memory-guard-available-memory.patch)

## False-trip mechanism

Darwin's `free_count` is intentionally small on a system using otherwise idle
RAM for caches. Treating `physical - free` as used memory therefore counts
reclaimable file-backed and purgeable cache against the deck. The guard can
cross its configured fraction even when substantial memory can be reclaimed
without compression or swap.

## Available-memory derivation

The corrected page estimate is:

```text
available = free + external + max(purgeable - external, 0)
```

The terms are:

- `free`: pages immediately unused.
- `external`: file-backed pages that the operating system can reclaim and
  re-read from their backing store.
- `purgeable`: pages that can be discarded, some of which can overlap the
  file-backed population.

Adding all purgeable pages to all external pages could double-count their
overlap. The final term therefore includes only purgeable pages beyond the
external count. The subtraction is guarded before unsigned arithmetic.

Inactive pages are deliberately excluded. The inactive population can contain
dirty anonymous memory that still requires compression or swap; inactivity is
not equivalent to cheap reclaimability. Counting it would make the guard
optimistic under the pressure conditions it is intended to catch.

The page total is multiplied by the value returned by `host_page_size()` and
clamped to physical memory. Apple Silicon systems commonly report 16,384-byte
pages, so a fixed 4,096-byte assumption would undercount by a factor of four.

## Independent pressure gates

The fraction gate remains configurable with `WARPX_MAX_MEM_FRACTION` and uses
the corrected available-memory estimate. Two additional signals catch cases
where a single reclaimable-memory snapshot is insufficient:

- A Darwin memory-pressure level of critical (4) or above is treated as
  abnormal. Warning level (2) is routine on macOS and does not abort on its
  own.
- Compressor growth of at least 512 MiB together with swapout growth of at
  least 128 MiB must occur for two consecutive sampling intervals.

The first compressor/swap sample establishes a baseline and cannot trigger the
gate. Requiring both deltas and two consecutive intervals reduces sensitivity
to isolated counter movement while retaining a bounded response to sustained
pressure.

## Fail-close behavior

On Apple platforms, failure to obtain physical memory, host page size, or VM
statistics aborts the run. Continuing without those required inputs would
silently disable the protection on the only platform where it is active. The
pressure-level sysctl is an additive signal: if it is unavailable, the required
VM accounting and compressor/swap checks still operate.

The startup check and runtime check use the same reclaimable-memory derivation.
Abort paths use the normal WarpX/AMReX error mechanism rather than terminating
the process directly.

## Platform scope

Darwin headers and APIs are enclosed by `__APPLE__` guards. Non-Apple builds
compile no-op sampling/check stubs and do not execute macOS memory policy. The
repository's Linux CPU workflow applies this patch to pinned upstream WarpX and
compiles the non-Apple path as a portability gate.

## Validation and limitations

The accounting and guard behavior were verified on Apple M3 Ultra hardware
running macOS 26.5.2, including the observed 16,384-byte host page size. The
published formula, pressure threshold, two-sample gate, and failure behavior
were also checked directly against the packaged patch.

This is a coarse host-memory safety guard, not a predictor for all Metal driver
or compiler behavior. Fixed thresholds can miss shorter pressure bursts, and
Darwin's page categories do not provide perfect ownership information. This
guard is independent of the later
[Metal in-order readback correction](adaptivecpp-metal-inorder-readback.md).
