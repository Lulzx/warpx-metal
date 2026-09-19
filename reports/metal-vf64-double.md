# FP64 on Metal through VF64 software binary64

## Summary

The AdaptiveCpp Metal emitter mapped the LLVM `double` type to MSL `float`.
Apple GPUs have no FP64 hardware and Metal Shading Language has no `double`,
so this demotion was the only way to get FP64-carrying kernels to compile at
all. It is sound for values that never leave registers. It is wrong for any
`double` that the host writes to memory and the device reads back: the device
struct is laid out with 4-byte fields where the host wrote 8-byte ones.

The keeper change carries `double` as a 64-bit IEEE-754 bit pattern (`ulong`)
and lowers every FP64 operation to the correctly rounded integer soft-float
runtime from [VF64-metal](https://github.com/Lulzx/VF64-metal). Layout now
matches the host, results are bit-identical to the CPU, and the cost is paid
only by kernels that actually use `double`.

## Failure mechanism

`amrex::Parser` compiles an expression into a byte stream of `alignas(8)`
nodes. A literal is

```cpp
struct alignas(8) ParserExeNumber { enum parser_exe_t type; double v; };
```

The host serializes it with `sizeof == 16` and `v` at offset 8. The emitter
produced

```metal
struct ParserExeNumber { uint field0; float field1; };
```

so the device read `field1` from offset 4, which is padding, and interpreted
four of the eight mantissa bytes as a `float`. Every literal in a
device-evaluated parser expression was garbage. The same stride and alignment
error applied to kernel-argument structs, arrays and constant globals holding
`double`, and `bitcast i64 <-> double` emitted invalid MSL.

Scope in this repository: the single-precision AdaptiveCpp build was already
shielded for the parser specifically, because
`patches/amrex-post/0004` redefines `ParserExeReal` as `float` under
`AMREX_USE_SYCL && AMREX_USE_FLOAT` with AdaptiveCpp, so host and device
agreed on a 4-byte literal. The demotion therefore bit `WarpX_PRECISION=DOUBLE`
builds (unbuildable in practice), any other host-written `double` reaching a
kernel, and every device computation that legitimately needed binary64
range or precision. The parser example above is the mechanism, not a
regression that the validated SP decks were exposed to.

## Correction

`patches/adaptivecpp/0023-metal-vf64-double.patch`:

- `mapType(double)` returns `ulong`.
- `fadd/fsub/fmul/fdiv/frem/fneg`, all sixteen `fcmp` predicates,
  `fpext/fptrunc/sitofp/uitofp/fptosi/fptoui`, `bitcast` and `double`
  constants are emitted as calls into a `__vf64_*` wrapper layer.
- The f64 builtins that have an exact soft-float implementation (`sqrt`,
  `fma`, `fabs`, `copysign`, `fmin`, `fmax`, `floor`, `ceil`, `trunc`, `rint`,
  `round`, `fmod`) have their libkernel bodies, which round through `float`,
  stripped in `LLVMToMetal` before inlining so the emitter can lower the call
  itself. The remaining transcendental f64 builtins still evaluate in `float`.
- `VF64Support.hpp` embeds VF64-metal's `IEEE/Arithmetic.metal` (commit
  `7290217`, about 1,100 lines of MSL) as a string. It is prepended to the
  generated shader only when the module references `double`.
- `patches/adaptivecpp/tools/gen-vf64-support.py` regenerates the header from
  a VF64-metal checkout so the two repositories can be kept in step.

`frem` is C `fmod` (truncated quotient); VF64 provides the IEEE remainder
(nearest quotient). The wrapper derives `fmod` from the remainder with one
exact correction: both values are exactly representable, and when the
remainder's sign differs from the dividend's, adding `copysign(b, a)` yields
the exact `fmod` result.

Unsupported constructs fail loudly in the emitter instead of miscompiling:
vector-of-double arithmetic and `double` atomics.

## Validation scope

Apple M4 Pro, macOS 27.0, Xcode 27, AdaptiveCpp `develop@3733a56` + patches
0008–0023, 2026-09-19.

`tests/sycl/double_test.cpp` (now part of `02-validate-metal.sh`), 4,096
elements:

- add, sub, mul, div, `fma`, `sqrt(fabs)`, and a `{int, double}` struct
  member load: 0 bitwise mismatches against the CPU.
- `<`, `==`, `double -> int`, `float -> double -> compare`,
  `double -> float`, `long -> double -> long`: 0 mismatches.
- 8-term Horner polynomial in `double`: 7.8e-16 maximum relative difference
  from the CPU (the CPU contracts to FMA; the GPU path does not).

`tests/sycl/double_bench.cpp`, 4M elements, 32 dependent multiply-adds:

| Type | ns per element | Relative |
|------|---------------:|---------:|
| `float` | 0.354 | 1x |
| `double` (VF64 ieee64) | 4.824 | 13.6x |

The existing smoke tests (device query, vector add, USM, reduction, D2H
stress) pass unchanged with the new emitter. With the rebuilt translator,
AMReX HeatEquation (`04-validate-amrex.sh`) passes, and WarpX Langmuir 2D
completes 40/40 steps and 3D 20/20 steps on the GPU (`06-validate-warpx.sh`);
the SP PIC loop contains no `double`, so it is unaffected by construction.

## Limitations and follow-ups

- Performance: a `WarpX_PRECISION=DOUBLE` build runs the whole PIC loop at
  soft-float speed. Keep production builds in `SINGLE`; `double` is now a
  correctness tool for setup-time code such as parser evaluation and for
  cross-checking FP32 sensitivity.
- VF64's reduced-precision pair modes (`fast48`, `wide48`, roughly 4–15x
  faster than `ieee64`) are not wired in. They are not binary64, and using
  them safely needs the pair-residency representation that VF64 documents for
  its own compiler; that is a separate project.
- Transcendental f64 builtins (`sin`, `cos`, `exp`, `log`, `pow`, ...) round
  through `float`. Exact versions would need a soft-float libm.
- Device-side parser evaluation is still broken for a different reason.
  A Langmuir 2D deck with `profile = parse_density_function` and
  `density_function = "n0"` (or `"4.e24"`) injects **zero** particles and
  exits 0 with no warning, whereas `profile = constant` injects 131,072. That
  is the nested-pointer translation defect of
  `reports/addplasma-parser-momentum-oob.md`, which `patches/warpx/0002`
  works around for momentum only; density, external-field and boundary
  expressions evaluated on the device remain unsupported. VF64 does not
  change this; it removes one of the two obstacles to fixing it.
