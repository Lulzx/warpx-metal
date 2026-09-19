#!/usr/bin/env bash
# 07-build-warpx-cpu.sh — Build WarpX with Apple Clang + OpenMP (no SYCL/Metal)
#
# Creates a CPU-only baseline for performance comparison with the Metal GPU build.
# Uses the same patched AMReX source tree; only the compiler and compute backend differ.
#
# Prerequisites: Run 03-build-amrex.sh first (to clone and patch AMReX source).
#
# Output: extern/warpx/build-cpu/bin/warpx.{2d,3d}.NOMPI.OMP.SP.PSP.EB
#
# Usage: ./scripts/07-build-warpx-cpu.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

# CPU build uses system Apple Clang (not LLVM 20)
CPU_CXX="$(xcrun -find clang++)"
CPU_CC="$(xcrun -find clang)"

LIBOMP_PREFIX="$(brew --prefix libomp 2>/dev/null || echo /opt/homebrew/opt/libomp)"
BUILD_DIR="${WARPX_SOURCE_DIR}/build-cpu"

echo ""
echo "=== CPU WarpX Build Configuration ==="
echo "  Compiler: ${CPU_CXX}"
echo "  OpenMP:   ${LIBOMP_PREFIX}"
echo "  Source:   ${WARPX_SOURCE_DIR}"
echo "  Build:    ${BUILD_DIR}"
echo "  AMReX:    ${AMREX_SOURCE_DIR}"
echo "======================================"

# Verify WarpX source is available
if [ ! -d "${WARPX_SOURCE_DIR}/.git" ]; then
    echo "[FAIL] WarpX source not found at ${WARPX_SOURCE_DIR}"
    echo "       Run ./scripts/05-build-warpx.sh first (which clones WarpX)."
    exit 1
fi

# Verify AMReX source is available (patched)
if [ ! -d "${AMREX_SOURCE_DIR}/.git" ]; then
    echo "[FAIL] AMReX source not found at ${AMREX_SOURCE_DIR}"
    echo "       Run ./scripts/03-build-amrex.sh first."
    exit 1
fi

if [ ! -d "${LIBOMP_PREFIX}" ]; then
    echo "[FAIL] libomp not found at ${LIBOMP_PREFIX}"
    echo "       Install with: brew install libomp"
    exit 1
fi

echo ""
echo "=== Step 1: Re-apply AMReX patches ==="
# Identical source state to the GPU build: the SYCL-only edits are macro-guarded
# and inert under OpenMP, so the CPU baseline differs only in compiler/backend.
"${SCRIPT_DIR}/lib/patch-amrex.sh"

echo ""
echo "=== Step 2: Configure WarpX CPU build ==="

rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

MACOS_SDK="$(xcrun --sdk macosx --show-sdk-path)"

cmake -S "${WARPX_SOURCE_DIR}" -B . \
    -G Ninja \
    -DCMAKE_CXX_COMPILER="${CPU_CXX}" \
    -DCMAKE_C_COMPILER="${CPU_CC}" \
    -DCMAKE_OSX_SYSROOT="${MACOS_SDK}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include" \
    -DCMAKE_EXE_LINKER_FLAGS="-L${LIBOMP_PREFIX}/lib -lomp" \
    -DCMAKE_SHARED_LINKER_FLAGS="-L${LIBOMP_PREFIX}/lib -lomp" \
    -DWarpX_COMPUTE=OMP \
    -DWarpX_PRECISION=SINGLE \
    -DWarpX_PARTICLE_PRECISION=SINGLE \
    -DWarpX_DIMS="2;3" \
    -DWarpX_MPI=OFF \
    -DWarpX_FFT=OFF \
    -DWarpX_QED=OFF \
    -DWarpX_OPENPMD=OFF \
    -DWarpX_amrex_src="${AMREX_SOURCE_DIR}"

echo ""
echo "=== Step 3: Build WarpX CPU ==="

ninja -j"${NPROC}"

echo ""
echo "=== Step 4: Verify build ==="

CPU_2D=""
CPU_3D=""
for dim in 2d 3d; do
    exe=$(find "${BUILD_DIR}/bin" -name "warpx.${dim}*" -type f -perm +111 2>/dev/null | head -1)
    if [ -n "${exe}" ]; then
        echo "  [OK] Found $(basename "${exe}")"
        if [ "${dim}" = "2d" ]; then CPU_2D="${exe}"; fi
        if [ "${dim}" = "3d" ]; then CPU_3D="${exe}"; fi
    else
        echo "  [WARN] WarpX ${dim} CPU executable not found in ${BUILD_DIR}/bin"
    fi
done

echo ""
echo "=== CPU WarpX build complete ==="
echo "  Build dir: ${BUILD_DIR}"
if [ -n "${CPU_2D}" ]; then echo "  2D exe: ${CPU_2D}"; fi
if [ -n "${CPU_3D}" ]; then echo "  3D exe: ${CPU_3D}"; fi
echo ""
echo "Next step: ./scripts/08-benchmark.sh"
