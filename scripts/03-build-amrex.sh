#!/usr/bin/env bash
# 03-build-amrex.sh — Clone, patch, and build AMReX with AdaptiveCpp SYCL backend
#
# Prerequisites: Run 01-build-adaptivecpp.sh first (need acpp compiler)
#
# Usage: ./scripts/03-build-amrex.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

AMREX_SOURCE_DIR="${WARPX_METAL_ROOT}/extern/amrex"
AMREX_INSTALL_PREFIX="${WARPX_METAL_ROOT}/opt/amrex"
ACPP="${ACPP_INSTALL_PREFIX}/bin/acpp"

# Verify acpp is available
if [ ! -x "${ACPP}" ]; then
    echo "[FAIL] acpp compiler not found at ${ACPP}"
    echo "       Run ./scripts/01-build-adaptivecpp.sh first."
    exit 1
fi

echo ""
echo "=== Step 1: Clone AMReX ==="

# Pinned 26.06 revision the patches in patches/amrex*/ are generated against
# (same pin as ci/run-local-ci.sh).
AMREX_REV="fa795322b44fff24fef3a795c3b00d24e015ee42"

if [ -d "${AMREX_SOURCE_DIR}/.git" ]; then
    echo "  [OK] AMReX already cloned at ${AMREX_SOURCE_DIR}"
    cd "${AMREX_SOURCE_DIR}"
    # Reset any previously applied patches before re-applying
    git checkout -- .
    git clean -fd
    git fetch origin
    git checkout "${AMREX_REV}"
else
    echo "  [..] Cloning AMReX..."
    git clone https://github.com/AMReX-Codes/amrex.git "${AMREX_SOURCE_DIR}"
    cd "${AMREX_SOURCE_DIR}"
    git checkout "${AMREX_REV}"
fi

echo ""
echo "=== Step 2: Apply patches ==="
# Single source of truth for the AMReX source state; shared with 05 and 07.
"${SCRIPT_DIR}/lib/patch-amrex.sh"

echo ""
echo "=== Step 3: Configure AMReX with CMake ==="

BUILD_DIR="${AMREX_SOURCE_DIR}/build-acpp"
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# macOS SDK sysroot (needed for Homebrew LLVM)
MACOS_SDK="$(xcrun --sdk macosx --show-sdk-path)"
ACPP_EXTRA_CXX_FLAGS="$(acpp_libcxx_workaround_flags)"
if [ -n "${ACPP_EXTRA_CXX_FLAGS}" ]; then
    echo "  [INFO] libc++/SDK workaround flags: ${ACPP_EXTRA_CXX_FLAGS}"
fi

cmake .. \
    -G Ninja \
    -DCMAKE_INSTALL_PREFIX="${AMREX_INSTALL_PREFIX}" \
    -DCMAKE_CXX_COMPILER="${ACPP}" \
    -DCMAKE_OSX_SYSROOT="${MACOS_SDK}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="${ACPP_EXTRA_CXX_FLAGS}" \
    -DAMReX_GPU_BACKEND=SYCL \
    -DAMReX_PRECISION=SINGLE \
    -DAMReX_PARTICLES_PRECISION=SINGLE \
    -DAMReX_SYCL_SUB_GROUP_SIZE=32 \
    -DAMReX_MPI=OFF \
    -DAMReX_OMP=OFF \
    -DAMReX_FORTRAN=OFF \
    -DAMReX_SYCL_AOT=OFF \
    -DAMReX_SYCL_SPLIT_KERNEL=OFF \
    -DAMReX_SYCL_ONEDPL=OFF

echo ""
echo "=== Step 4: Build AMReX ==="

ninja -j"${NPROC}"

echo ""
echo "=== Step 5: Install AMReX ==="

ninja install

echo ""
echo "=== Step 6: Verify installation ==="

if [ -f "${AMREX_INSTALL_PREFIX}/lib/cmake/AMReX/AMReXConfig.cmake" ]; then
    echo "  [OK] AMReX CMake config found"
else
    # Some installs put it in lib64
    if [ -f "${AMREX_INSTALL_PREFIX}/lib64/cmake/AMReX/AMReXConfig.cmake" ]; then
        echo "  [OK] AMReX CMake config found (lib64)"
    else
        echo "  [FAIL] AMReX CMake config not found in ${AMREX_INSTALL_PREFIX}"
        exit 1
    fi
fi

if ls "${AMREX_INSTALL_PREFIX}"/lib*/libamrex* 1>/dev/null 2>&1; then
    echo "  [OK] AMReX libraries installed:"
    ls -la "${AMREX_INSTALL_PREFIX}"/lib*/libamrex* 2>/dev/null || true
else
    echo "  [FAIL] AMReX libraries not found"
    exit 1
fi

echo ""
echo "=== AMReX build complete ==="
echo "  Source:  ${AMREX_SOURCE_DIR}"
echo "  Install: ${AMREX_INSTALL_PREFIX}"
echo ""
echo "Next step: ./scripts/04-validate-amrex.sh"
