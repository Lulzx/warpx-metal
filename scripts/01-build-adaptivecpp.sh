#!/usr/bin/env bash
# 01-build-adaptivecpp.sh — Clone and build AdaptiveCpp with Metal backend
#
# Prerequisites: Run 00-install-deps.sh first
#
# Usage: ./scripts/01-build-adaptivecpp.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

echo ""
echo "=== Step 1: Clone AdaptiveCpp ==="

if [ -d "${ACPP_SOURCE_DIR}/.git" ]; then
    echo "  [OK] AdaptiveCpp already cloned at ${ACPP_SOURCE_DIR}"
    echo "  Updating to latest develop..."
    cd "${ACPP_SOURCE_DIR}"
    git fetch origin
    git checkout develop
    git pull origin develop
else
    echo "  [..] Cloning AdaptiveCpp develop branch..."
    git clone --branch develop https://github.com/AdaptiveCpp/AdaptiveCpp.git "${ACPP_SOURCE_DIR}"
fi

echo ""
echo "=== Step 2: Apply patches ==="

PATCH_DIR="${PATCHES_DIR}/adaptivecpp"
PATCH_COUNT=$(find "${PATCH_DIR}" -name '*.patch' 2>/dev/null | wc -l | tr -d ' ')
if [ "${PATCH_COUNT}" -gt 0 ]; then
    cd "${ACPP_SOURCE_DIR}"
    for patch in "${PATCH_DIR}"/*.patch; do
        PATCH_NAME="$(basename "${patch}")"
        if git apply --check "${patch}" 2>/dev/null; then
            echo "  [..] Applying ${PATCH_NAME}..."
            git apply "${patch}"
            echo "  [OK] Applied ${PATCH_NAME}"
        else
            echo "  [SKIP] ${PATCH_NAME} (already applied or conflicts)"
        fi
    done
else
    echo "  [OK] No patches to apply"
fi

echo ""
echo "=== Step 3: Configure with CMake ==="

BUILD_DIR="${ACPP_SOURCE_DIR}/build"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake .. \
    -G Ninja \
    -DCMAKE_INSTALL_PREFIX="${ACPP_INSTALL_PREFIX}" \
    -DCMAKE_C_COMPILER="${CC}" \
    -DCMAKE_CXX_COMPILER="${CXX}" \
    -DLLVM_DIR="${LLVM_DIR}" \
    -DWITH_METAL_BACKEND=ON \
    -DWITH_CPU_BACKEND=ON \
    -DWITH_CUDA_BACKEND=OFF \
    -DWITH_ROCM_BACKEND=OFF \
    -DWITH_OPENCL_BACKEND=OFF \
    -DWITH_LEVEL_ZERO_BACKEND=OFF \
    -DCMAKE_BUILD_TYPE=Release

echo ""
echo "=== Step 4: Build ==="

ninja -j"${NPROC}"

echo ""
echo "=== Step 5: Install ==="

ninja install

echo ""
echo "=== Step 6: Verify installation ==="

if [ -x "${ACPP_INSTALL_PREFIX}/bin/acpp-info" ]; then
    echo ""
    echo "--- acpp-info output ---"
    "${ACPP_INSTALL_PREFIX}/bin/acpp-info" || true
    echo "------------------------"
    echo ""

    if "${ACPP_INSTALL_PREFIX}/bin/acpp-info" 2>&1 | grep -qi "metal\|apple\|gpu"; then
        echo "  [OK] Metal device detected by acpp-info"
    else
        echo "  [WARN] acpp-info ran but no Metal device detected."
        echo "         Check that WITH_METAL_BACKEND=ON was effective."
    fi
else
    echo "  [FAIL] acpp-info not found at ${ACPP_INSTALL_PREFIX}/bin/acpp-info"
    exit 1
fi

echo ""
echo "=== AdaptiveCpp build complete ==="
echo "  Install prefix: ${ACPP_INSTALL_PREFIX}"
echo ""
echo "Next step: ./scripts/02-validate-metal.sh"
