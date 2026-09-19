/* libcxx-infinity-shim.h — force-included by 03/05 when the toolchain needs it.
 *
 * Homebrew LLVM 20's libc++ <random> uses INFINITY. With the macOS 27 SDK and
 * -std=c++20, math.h defers INFINITY/NAN to <float.h> via the
 * __need_infinity_nan protocol (because __has_feature(modules) is true in
 * C++20), and clang 20's float.h does not implement that protocol, so neither
 * macro is ever defined. Apple's clang does not have this problem.
 *
 * These are the same expansions the SDK would produce. Detected by
 * acpp_libcxx_workaround_flags in scripts/env.sh; never applied blindly.
 */
#ifndef WARPX_METAL_LIBCXX_INFINITY_SHIM_H
#define WARPX_METAL_LIBCXX_INFINITY_SHIM_H
#ifndef INFINITY
#define INFINITY __builtin_inff()
#endif
#ifndef NAN
#define NAN __builtin_nanf("")
#endif
#endif
