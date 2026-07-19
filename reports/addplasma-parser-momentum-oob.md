# AddPlasma parser-momentum out-of-bounds access

## Summary

The SYCL/XZ particle-injection path could dereference parser-backed momentum
state inside an AdaptiveCpp Metal device kernel. That state contains nested
pointers whose translation was not valid for this path. Metal shader validation
reported out-of-bounds loads and stores during `AddPlasma`; one diagnostic
described a user allocation with `length:1`.

The keeper patch avoids parser-momentum evaluation in the affected device
kernels and uses the existing host-side parser backfill for the final particle
momenta.

Patch: [`patches/warpx/0002-fix-addplasma-metal-parser-momentum-oob.patch`](../patches/warpx/0002-fix-addplasma-metal-parser-momentum-oob.patch)

## Defect mechanism

Two `AddPlasma` operations crossed the unsafe nested-pointer boundary:

1. The particle pre-count path called `applyBallisticCorrection()`. Even for an
   unboosted simulation, that function called `getBulkMomentum()` before
   returning the unchanged longitudinal position.
2. The particle-creation kernel called the parser-backed `getMomentum()` method
   directly.

Both calls occurred in device code for the Metal SYCL/XZ configuration. The
validation layer attributed 128 invalid loads and 128 invalid stores to the
affected `AddPlasma` kernels before the correction. The `length:1` report was a
translation-failure symptom, not evidence of a one-particle sizing error.

## Correction

The patch makes three bounded changes:

- `applyBallisticCorrection()` returns the input position immediately when the
  boost is the identity, avoiding an unnecessary bulk-momentum dereference.
- For parser momentum under GPU+SYCL+XZ, both device momentum branches write a
  temporary zero momentum instead of dereferencing parser state.
- The pre-existing host backfill evaluates the parser and copies `ux`, `uy`,
  and `uz` to the device after the kernel completes.

For the validated unboosted configuration, density and particle weight do not
depend on the temporary momentum. The host backfill therefore restores the
intended momentum values before subsequent particle processing.

## Backend scope

The bypass is compiled only when all of `AMREX_USE_GPU`, `AMREX_USE_SYCL`, and
`WARPX_DIM_XZ` are defined, and it activates only for parser momentum.

- CPU builds do not define `AMREX_USE_GPU`.
- CUDA and HIP builds do not define `AMREX_USE_SYCL`.
- Non-parser SYCL/XZ injection continues to evaluate momentum on the device.

Those paths retain their previous behavior.

## Validation

The corrected build was verified on Apple M3 Ultra hardware running macOS
26.5.2 with Metal shader validation enabled. A 2,500-step injection run
completed with:

- invalid device loads: 0
- invalid device stores: 0
- out-of-bounds buffer reports: 0

This establishes removal of the reported memory-safety defect in the validated
configuration. The separate runtime stall is contained—but an underlying driver
completion loss is not repaired—by the
[Metal in-order readback correction](adaptivecpp-metal-inorder-readback.md).

## Residual limitation

Boosted-frame parser injection remains outside the supported scope of this
fix. With a non-identity boost, `applyBallisticCorrection()` can still require
parser-backed bulk momentum in device code. In addition, the device-side
boosted density/weight transformation runs before the host momentum backfill;
the backfill repairs momentum but not particle weight.

Until host-side boosted density/weight recomputation and a boost-safe ballistic
path are implemented and validated, this patch must not be cited as support for
boosted-frame parser injection on Metal.
