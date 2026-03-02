# warpx-metal

WarpX electromagnetic particle-in-cell simulation running on Apple GPU via SYCL.

The build chain is WarpX -> AMReX (SYCL backend) -> AdaptiveCpp SSCP -> Metal.
AdaptiveCpp compiles SYCL kernels to LLVM IR at build time and JIT-translates
them to Metal Shading Language at runtime. Tested on an M4 Pro (applegpu_g16s,
16 compute units).

## Status

WarpX — a full electromagnetic particle-in-cell (PIC) plasma physics code — is
running SYCL kernels on Apple Silicon Metal GPU via AdaptiveCpp. The entire
stack works end to end:

    WarpX → AMReX → AdaptiveCpp SSCP → Metal → Apple M4 Pro GPU

Key achievements that required non-trivial engineering:

- Built AdaptiveCpp's Metal emitter with 4 patches (array deps, AS5 identity
  cast, PHI cycle, thread-param IPA)
- Fixed AMReX's atomic operations for SSCP mode (atomics were silently falling
  to no-ops)
- Worked around Metal's zero FP64 support, no __int128, no host_task, no
  sub-group size hints

## Build

Requires macOS 14+, Xcode 16, Homebrew, Apple Silicon.

    ./scripts/00-install-deps.sh
    ./scripts/01-build-adaptivecpp.sh
    ./scripts/02-validate-metal.sh
    ./scripts/03-build-amrex.sh
    ./scripts/04-validate-amrex.sh
    ./scripts/05-build-warpx.sh
    ./scripts/06-validate-warpx.sh

LLVM 20 is required; the Metal backend uses LLVM 20 APIs not present in 18.
LLVM 18 is also needed because the LLVM 20 Homebrew bottle omits ld64.lld.
The first run is slow due to JIT compilation (LLVM IR -> MSL). Subsequent
runs use the cache at ~/.acpp/apps/global/jit-cache/.

## Limitations

Apple GPUs have no FP64. All builds use single precision
(-DAMReX_PRECISION=SINGLE -DAMReX_PARTICLES_PRECISION=SINGLE).
__int128 is disabled (-DAMREX_NO_INT128); the Metal emitter maps i128 to
uint4 with incomplete cast support. The PSATD spectral solver is unavailable
(no Metal FFT library). Only the FDTD solver is supported.

## Patches

### patches/adaptivecpp/0008-metal-all-warpx-fixes.patch

Fixes four bugs in the AdaptiveCpp Metal emitter (Emitter.cpp):

**1. Missing struct definitions for array-element types.**
addDeps() checked direct struct members for MSL type dependencies but did not
recurse into ArrayType. Struct types only referenced as array elements had no
MSL definition emitted, causing compile errors in generated shaders.

**2. Thread-address-space pointer reads returning zero (critical).**
addrspacecast AS5->AS0 was emitted as `(device void*)((ulong)x)`. Apple
Silicon returns zeros when thread-address-space memory is read through a
device pointer. WarpX's AddPlasma kernel stores cell bounds in thread-local
GpuArray<> variables and accesses them through an AS5->AS0 cast. The emitter
produced a device pointer, every bounds check returned zero, all 65536
particles were injected with ID=0, and Redistribute removed them all.
The correct emission is `x = x` (identity cast, preserving the thread
qualifier). Fixed by treating addrspacecast AS5->AS0 as a no-op in the
emitter and propagating the AS5 type through all downstream GEPs.

**3. PHI cycle address-space resolution returning wrong address space.**
getPhysicalPointerAddressSpace() used a SENTINEL value to detect cycles but
returned 1 (device) on a SENTINEL cache hit. A loop counter PHI with one AS5
incoming edge and one GEP back-edge resolved to device instead of unknown,
propagating the wrong address space to downstream loads.
Fixed by returning 0 (unknown) on SENTINEL hit and not caching 0 in
GEP/BitCast/Select nodes so re-evaluation is possible after the PHI resolves.

**4. Incorrect address-space signatures for helper functions (thread params).**
Added analyzeThreadASParams(): a two-pass inter-procedural analysis that
identifies function parameters always receiving AS5 pointers. Without this,
helper functions were given device void* signatures and the Metal compiler
rejected the calls.

### patches/amrex/0002-amrex-sscp-atomic-fix.patch

Fixes non-atomic GPU operations under AdaptiveCpp SSCP.

AdaptiveCpp SSCP uses a unified host-device compilation pass and never defines
__SYCL_DEVICE_ONLY__. AMReX's Gpu::Atomic::Add and all related wrapper
functions (Min, Max, If, Multiply, Divide, HostDevice::Atomic::Add) use the
AMREX_IF_ON_DEVICE macro, which tests __SYCL_DEVICE_ONLY__, to dispatch
between sycl::atomic_ref and a plain load-modify-store. In SSCP mode the
condition was always false: every atomic call in a GPU kernel executed
`auto old = *sum; *sum += value; return old` instead.

DenseBins::buildGPU increments bin counters with Gpu::Atomic::Add to build a
counting-sort permutation. With non-atomic increments and 256 threads racing
on shared counters, the permutation array was corrupt. ReorderParticles
assigned garbage source indices to output slots; particles received ID=0 and
were removed by Redistribute. The symptom was exactly 3/4 of positrons lost
at the first sort step (default sort period is 4 steps; each corrupt sort
removes a fixed fraction depending on the collision pattern).

The fix adds `|| defined(__ACPP_ENABLE_LLVM_SSCP_TARGET__)` to every
`#if defined(__SYCL_DEVICE_ONLY__)` guard inside the _device functions
(Add_device, atomic_op, atomic_op_if, AddNoRet, Min_device, Max_device,
LogicalOr, LogicalAnd, Exch, CAS), and adds a direct SSCP dispatch path to
each wrapper that calls the _device variant without going through
AMREX_IF_ON_DEVICE.

__ACPP_ENABLE_LLVM_SSCP_TARGET__ is injected by the acpp compiler script
before any headers are processed and is the correct compile-time indicator
for SSCP mode.

Patching AMREX_IF_ON_DEVICE globally in AMReX_GpuQualifiers.H is not viable:
AMReX_GpuRange.H uses AMREX_IF_ON_DEVICE around CUDA-specific identifiers
(blockDim, blockIdx, threadIdx, gridDim) that do not exist under SYCL.

## License

BSD-3-Clause.
