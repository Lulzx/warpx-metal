#!/usr/bin/env bash
# env.sh — Shared paths and configuration for WarpX-on-Metal builds
# Source this file before running other scripts:  source scripts/env.sh

set -euo pipefail

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

echo "=== WarpX-on-Metal Environment ==="
echo "  Project root:    ${WARPX_METAL_ROOT}"
echo "  LLVM prefix:     ${LLVM_PREFIX}"
echo "  LLVM_DIR:        ${LLVM_DIR}"
echo "  Metal-cpp:       ${METAL_CPP_DIR}"
echo "  AdaptiveCpp src: ${ACPP_SOURCE_DIR}"
echo "  AdaptiveCpp dst: ${ACPP_INSTALL_PREFIX}"
echo "  Parallelism:     ${NPROC} cores"
echo "=================================="
