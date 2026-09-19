#!/usr/bin/env bash
# env.sh — Shared paths and configuration for WarpX-on-Metal builds
# Source this file before running other scripts:  source scripts/env.sh
#
# Deliberately no `set -e` here: this file is sourced, and every script sets
# its own shell options. Arming -e in an interactive shell that sources this
# file would make the next failing command close the terminal.

# Project root (resolve relative to this script's location)
export WARPX_METAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# LLVM 20 (AdaptiveCpp develop branch Metal backend requires LLVM 20 APIs)
# LLVM 18 is also needed for ld64.lld (not shipped in LLVM 20 Homebrew bottle)
export LLVM_PREFIX="$(brew --prefix llvm@20 2>/dev/null || echo /opt/homebrew/opt/llvm@20)"
export LLVM_DIR="${LLVM_PREFIX}/lib/cmake/llvm"
export CC="${LLVM_PREFIX}/bin/clang"
export CXX="${LLVM_PREFIX}/bin/clang++"

# Metal-cpp headers
export METAL_CPP_DIR="${WARPX_METAL_ROOT}/opt/metal-cpp"

# AdaptiveCpp install prefix
export ACPP_INSTALL_PREFIX="${WARPX_METAL_ROOT}/opt/adaptivecpp"

# AMReX install prefix
export AMREX_INSTALL_PREFIX="${WARPX_METAL_ROOT}/opt/amrex"

# AMReX source
export AMREX_SOURCE_DIR="${WARPX_METAL_ROOT}/extern/amrex"

# WarpX source
export WARPX_SOURCE_DIR="${WARPX_METAL_ROOT}/extern/warpx"
export WARPX_GPU_BUILD_DIR="${WARPX_SOURCE_DIR}/build-acpp"
export WARPX_CPU_BUILD_DIR="${WARPX_SOURCE_DIR}/build-cpu"

# AdaptiveCpp source
export ACPP_SOURCE_DIR="${WARPX_METAL_ROOT}/extern/AdaptiveCpp"

# Patches
export PATCHES_DIR="${WARPX_METAL_ROOT}/patches"

# Add installed AdaptiveCpp to PATH
if [ -d "${ACPP_INSTALL_PREFIX}/bin" ]; then
    export PATH="${ACPP_INSTALL_PREFIX}/bin:${PATH}"
fi

# Build parallelism
export NPROC="$(sysctl -n hw.ncpu)"

# Homebrew LLVM 20's libc++ fails to compile <random> under -std=c++20 with
# the macOS 27 SDK: math.h defers INFINITY/NAN to <float.h> through the
# __need_infinity_nan protocol when __has_feature(modules) is true (which
# C++20 implies), and clang 20's float.h does not implement that protocol.
# Apple's clang does. AMReX and WarpX compile as C++20, so probe once and
# supply the builtin definitions only when the toolchain needs them.
acpp_libcxx_workaround_flags() {
    local acpp="${ACPP_INSTALL_PREFIX}/bin/acpp"
    [ -x "${acpp}" ] || return 0
    local tmp
    tmp="$(mktemp -d)"
    printf '#include <random>\nint main() { return 0; }\n' > "${tmp}/probe.cpp"
    if ! "${acpp}" -std=c++20 -c "${tmp}/probe.cpp" -o "${tmp}/probe.o" >/dev/null 2>&1; then
        # A force-included header rather than -D: parentheses/quotes in a
        # -D value do not survive CMake -> ninja -> /bin/sh.
        echo "-include ${WARPX_METAL_ROOT}/scripts/lib/libcxx-infinity-shim.h"
    fi
    rm -rf "${tmp}"
}

echo "=== WarpX-on-Metal Environment ==="
echo "  Project root:    ${WARPX_METAL_ROOT}"
echo "  LLVM prefix:     ${LLVM_PREFIX}"
echo "  LLVM_DIR:        ${LLVM_DIR}"
echo "  Metal-cpp:       ${METAL_CPP_DIR}"
echo "  AdaptiveCpp src: ${ACPP_SOURCE_DIR}"
echo "  AdaptiveCpp dst: ${ACPP_INSTALL_PREFIX}"
echo "  Parallelism:     ${NPROC} cores"
echo "=================================="
