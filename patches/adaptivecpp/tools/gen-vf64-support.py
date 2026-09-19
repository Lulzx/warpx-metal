#!/usr/bin/env python3
"""Regenerate VF64Support.hpp for the AdaptiveCpp Metal emitter.

Usage: gen-vf64-support.py <VF64-metal checkout> <output .hpp>

Embeds VF64-metal's integer IEEE-754 binary64 runtime
(Sources/VF64Metal/Shaders/IEEE/Arithmetic.metal) as an MSL string plus the
thin __vf64_* wrappers the emitter calls. After regenerating, rebuild
patches/adaptivecpp/0023-metal-vf64-double.patch from the AdaptiveCpp tree.
"""
import subprocess
import sys
from pathlib import Path

WRAPPERS = r'''
// ---------------------------------------------------------------------------
// Thin wrappers used by the AdaptiveCpp Metal emitter. `double` values are
// carried as `ulong` IEEE-754 binary64 bit patterns; all arithmetic is
// correctly rounded (nearest-even) software binary64 from VF64-metal.
// ---------------------------------------------------------------------------
inline ulong __vf64_add(ulong a, ulong b) { uint f = 0; return soft_add64_status(a, b, soft_round_near_even, f); }
inline ulong __vf64_sub(ulong a, ulong b) { uint f = 0; return soft_sub64_status(a, b, soft_round_near_even, f); }
inline ulong __vf64_mul(ulong a, ulong b) { uint f = 0; return soft_mul64_status(a, b, soft_round_near_even, f); }
inline ulong __vf64_div(ulong a, ulong b) { uint f = 0; return soft_div64_status(a, b, soft_round_near_even, f); }
inline ulong __vf64_sqrt(ulong a) { uint f = 0; return soft_sqrt64_status(a, soft_round_near_even, f); }
inline ulong __vf64_fma(ulong a, ulong b, ulong c) { uint f = 0; return soft_fma64_status(a, b, c, soft_round_near_even, f); }
inline ulong __vf64_neg(ulong a) { return a ^ 0x8000000000000000ul; }
inline ulong __vf64_fabs(ulong a) { return a & 0x7ffffffffffffffful; }
inline ulong __vf64_copysign(ulong a, ulong b) { return (a & 0x7ffffffffffffffful) | (b & 0x8000000000000000ul); }
inline bool __vf64_isnan(ulong a) { return soft_is_nan(a); }
inline bool __vf64_eq(ulong a, ulong b) { uint f = 0; return soft_equal64_status(a, b, false, f); }
inline bool __vf64_lt(ulong a, ulong b) { uint f = 0; return soft_less64_status(a, b, false, true, f); }
inline bool __vf64_le(ulong a, ulong b) { uint f = 0; return soft_less64_status(a, b, true, true, f); }
inline ulong __vf64_fmin(ulong a, ulong b) {
    if (soft_is_nan(a)) return b;
    if (soft_is_nan(b)) return a;
    return __vf64_lt(b, a) ? b : a;
}
inline ulong __vf64_fmax(ulong a, ulong b) {
    if (soft_is_nan(a)) return b;
    if (soft_is_nan(b)) return a;
    return __vf64_lt(a, b) ? b : a;
}
inline ulong __vf64_floor(ulong a) { uint f = 0; return soft_round_to_int64_status(a, soft_round_min, false, f); }
inline ulong __vf64_ceil(ulong a) { uint f = 0; return soft_round_to_int64_status(a, soft_round_max, false, f); }
inline ulong __vf64_trunc(ulong a) { uint f = 0; return soft_round_to_int64_status(a, soft_round_min_mag, false, f); }
inline ulong __vf64_rint(ulong a) { uint f = 0; return soft_round_to_int64_status(a, soft_round_near_even, false, f); }
inline ulong __vf64_round(ulong a) { uint f = 0; return soft_round_to_int64_status(a, soft_round_near_max_mag, false, f); }
// C fmod (truncated quotient) derived from the exact IEEE remainder: both are
// exactly representable, and the correction term keeps the result exact.
inline ulong __vf64_fmod(ulong a, ulong b) {
    uint f = 0;
    ulong r = soft_remainder64_status(a, b, f);
    if (soft_is_nan(r)) return r;
    if ((r & 0x7ffffffffffffffful) == 0) return a & 0x8000000000000000ul;
    if (((r ^ a) >> 63) != 0) r = __vf64_add(r, __vf64_copysign(b, a));
    return r;
}
inline ulong __vf64_from_f32(float x) { uint f = 0; return soft_format_to_f64_status(ulong(as_type<uint>(x)), 8u, 23u, 127, f); }
inline float __vf64_to_f32(ulong x) { uint f = 0; return as_type<float>(uint(soft_f64_to_format_status(x, soft_round_near_even, 8u, 23u, 127, f))); }
inline ulong __vf64_from_u64(ulong v) { uint f = 0; return soft_uint_to_f64_status(v, false, soft_round_near_even, f); }
inline ulong __vf64_from_i64(ulong v) {
    bool sign = (v >> 63) != 0;
    ulong magnitude = sign ? (~v + 1ul) : v;
    uint f = 0;
    return soft_uint_to_f64_status(magnitude, sign, soft_round_near_even, f);
}
// LLVM fptoui/fptosi truncate toward zero; out-of-range inputs are poison in
// LLVM, so the SoftFloat saturation result is an acceptable definition.
inline ulong __vf64_to_u64(ulong x) { uint f = 0; return soft_f64_to_int_status(x, soft_round_min_mag, false, false, 64u, f); }
inline ulong __vf64_to_i64(ulong x) { uint f = 0; return soft_f64_to_int_status(x, soft_round_min_mag, false, true, 64u, f); }
'''

HEADER = '''// Generated from VF64-metal Sources/VF64Metal/Shaders/IEEE/Arithmetic.metal
// (https://github.com/Lulzx/VF64-metal, commit {commit}). Do not edit by hand;
// regenerate with patches/adaptivecpp/tools/gen-vf64-support.py.
//
// Correctly rounded software IEEE-754 binary64 for Apple GPUs, which have no
// native FP64. The Metal emitter inlines this text into the generated MSL
// whenever a kernel module references the LLVM `double` type.
#ifndef HIPSYCL_METAL_VF64_SUPPORT_HPP
#define HIPSYCL_METAL_VF64_SUPPORT_HPP

#include <string>
#include <unordered_map>

namespace hipsycl {{
namespace compiler {{

// f64 math builtins that have an exact VF64 implementation. LLVMToMetal strips
// the libkernel (float-rounding) bodies of these so the calls reach the
// emitter, which lowers them to the mapped __vf64_* function.
inline const std::unordered_map<std::string, std::string>& vf64ExactBuiltins() {{
  static const std::unordered_map<std::string, std::string> table = {{
    {{"__acpp_sscp_sqrt_f64", "__vf64_sqrt"}},
    {{"__acpp_sscp_fma_f64", "__vf64_fma"}},
    {{"__acpp_sscp_fabs_f64", "__vf64_fabs"}},
    {{"__acpp_sscp_copysign_f64", "__vf64_copysign"}},
    {{"__acpp_sscp_fmin_f64", "__vf64_fmin"}},
    {{"__acpp_sscp_fmax_f64", "__vf64_fmax"}},
    {{"__acpp_sscp_floor_f64", "__vf64_floor"}},
    {{"__acpp_sscp_ceil_f64", "__vf64_ceil"}},
    {{"__acpp_sscp_trunc_f64", "__vf64_trunc"}},
    {{"__acpp_sscp_rint_f64", "__vf64_rint"}},
    {{"__acpp_sscp_round_f64", "__vf64_round"}},
    {{"__acpp_sscp_fmod_f64", "__vf64_fmod"}},
  }};
  return table;
}}

static const char* const vf64_support_msl = R"vf64(
// ===================== VF64-metal IEEE/Arithmetic.metal =====================
'''

FOOTER = ''')vf64";

}
}

#endif
'''


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    root = Path(sys.argv[1])
    out = Path(sys.argv[2])
    src = root / "Sources/VF64Metal/Shaders/IEEE/Arithmetic.metal"
    body = src.read_text()
    if ')vf64"' in body:
        raise SystemExit("raw-string delimiter collision in " + str(src))
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    out.write_text(HEADER.format(commit=commit) + body + WRAPPERS + FOOTER)
    print(f"wrote {out} ({out.stat().st_size} bytes, VF64-metal {commit})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
