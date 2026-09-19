#!/usr/bin/env bash
# 05-build-warpx.sh — Clone, patch, and build WarpX with AdaptiveCpp SYCL/Metal backend
#
# Prerequisites: Run 01-build-adaptivecpp.sh first (03-build-amrex.sh optional)
#
# WarpX builds AMReX from source (FetchContent or local source tree) so our
# pre-installed AMReX is only needed for standalone AMReX tests. Here we point
# WarpX at our Metal-patched AMReX source and let it build AMReX as a subproject.
#
# Usage: ./scripts/05-build-warpx.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

WARPX_SOURCE_DIR="${WARPX_METAL_ROOT}/extern/warpx"
ACPP="${ACPP_INSTALL_PREFIX}/bin/acpp"

# Verify acpp is available
if [ ! -x "${ACPP}" ]; then
    echo "[FAIL] acpp compiler not found at ${ACPP}"
    echo "       Run ./scripts/01-build-adaptivecpp.sh first."
    exit 1
fi

# Verify AMReX source is available (for patches)
if [ ! -d "${AMREX_SOURCE_DIR}/.git" ]; then
    echo "[FAIL] AMReX source not found at ${AMREX_SOURCE_DIR}"
    echo "       Run ./scripts/03-build-amrex.sh first to clone and patch AMReX."
    exit 1
fi

echo ""
echo "=== Step 1: Clone WarpX ==="

# Pinned 26.06 revision the patches in patches/warpx/ are generated against
# (same pin as ci/run-local-ci.sh).
WARPX_REV="f7db079f9a8ca96d179e02709a29fe6c027ed8ed"

if [ -d "${WARPX_SOURCE_DIR}/.git" ]; then
    echo "  [OK] WarpX already cloned at ${WARPX_SOURCE_DIR}"
    cd "${WARPX_SOURCE_DIR}"
    git checkout -- .
    git clean -fd
    git fetch origin
    git checkout "${WARPX_REV}"
else
    echo "  [..] Cloning WarpX..."
    git clone https://github.com/ECP-WarpX/WarpX.git "${WARPX_SOURCE_DIR}"
    cd "${WARPX_SOURCE_DIR}"
    git checkout "${WARPX_REV}"
fi

echo ""
echo "=== Step 2: Apply AMReX patches ==="
echo "  (WarpX builds AMReX from source — same tree and edits as 03/07)"
"${SCRIPT_DIR}/lib/patch-amrex.sh"

echo ""
echo "=== Step 3: Apply WarpX patches ==="

WARPX_PATCH_DIR="${PATCHES_DIR}/warpx"
mkdir -p "${WARPX_PATCH_DIR}"

if [ -d "${WARPX_PATCH_DIR}" ]; then
    PATCH_COUNT=$(find "${WARPX_PATCH_DIR}" -name '*.patch' 2>/dev/null | wc -l | tr -d ' ')
    if [ "${PATCH_COUNT}" -gt 0 ]; then
        cd "${WARPX_SOURCE_DIR}"
        for patch in "${WARPX_PATCH_DIR}"/*.patch; do
            PATCH_NAME="$(basename "${patch}")"
            if git apply --check "${patch}" 2>/dev/null; then
                echo "  [..] Applying ${PATCH_NAME}..."
                git apply "${patch}"
                echo "  [OK] Applied ${PATCH_NAME}"
            elif git apply --reverse --check "${patch}" 2>/dev/null; then
                echo "  [OK] ${PATCH_NAME} already applied"
            else
                echo "  [FAIL] ${PATCH_NAME} does not apply cleanly" >&2
                exit 1
            fi
        done
    else
        echo "  [OK] No WarpX patches to apply"
    fi
fi

echo ""
echo "=== Step 4: Configure WarpX with CMake ==="

BUILD_DIR="${WARPX_SOURCE_DIR}/build-acpp"
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# macOS SDK sysroot
MACOS_SDK="$(xcrun --sdk macosx --show-sdk-path)"
ACPP_EXTRA_CXX_FLAGS="$(acpp_libcxx_workaround_flags)"
if [ -n "${ACPP_EXTRA_CXX_FLAGS}" ]; then
    echo "  [INFO] libc++/SDK workaround flags: ${ACPP_EXTRA_CXX_FLAGS}"
fi

# Use our Metal-patched AMReX source tree (WarpX builds it as a subproject).
# WarpX will configure AMReX with the right components (2D/3D, PIC, EB, etc.).
cmake -S "${WARPX_SOURCE_DIR}" -B . \
    -G Ninja \
    -DCMAKE_CXX_COMPILER="${ACPP}" \
    -DCMAKE_OSX_SYSROOT="${MACOS_SDK}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="${ACPP_EXTRA_CXX_FLAGS}" \
    -DWarpX_COMPUTE=SYCL \
    -DWarpX_PRECISION=SINGLE \
    -DWarpX_PARTICLE_PRECISION=SINGLE \
    -DWarpX_DIMS="2;3" \
    -DWarpX_MPI=OFF \
    -DWarpX_FFT=OFF \
    -DWarpX_QED=OFF \
    -DWarpX_OPENPMD=OFF \
    -DWarpX_amrex_src="${AMREX_SOURCE_DIR}"

echo ""
echo "=== Step 5: Build WarpX ==="

ninja -j"${NPROC}"

echo ""
echo "=== Step 6: Verify build ==="

# Check for WarpX executables
for dim in 2d 3d; do
    exe=$(find "${BUILD_DIR}" -name "warpx.${dim}*" -type f -perm +111 2>/dev/null | head -1)
    if [ -n "${exe}" ]; then
        echo "  [OK] Found ${exe}"
    else
        exe=$(find "${BUILD_DIR}" -name "*warpx*${dim}*" -type f -perm +111 2>/dev/null | head -1)
        if [ -n "${exe}" ]; then
            echo "  [OK] Found ${exe}"
        else
            echo "  [WARN] WarpX ${dim} executable not found"
        fi
    fi
done

echo ""
echo "=== WarpX build complete ==="
echo "  Source: ${WARPX_SOURCE_DIR}"
echo "  Build:  ${BUILD_DIR}"
echo ""
echo "Next step: ./scripts/06-validate-warpx.sh"
