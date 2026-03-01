#!/usr/bin/env bash
# 04-validate-amrex.sh — Build and run AMReX HeatEquation test on Metal GPU
#
# Prerequisites: Run 03-build-amrex.sh first (need installed AMReX)
#
# Usage: ./scripts/04-validate-amrex.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

ACPP="${ACPP_INSTALL_PREFIX}/bin/acpp"
AMREX_SOURCE_DIR="${WARPX_METAL_ROOT}/extern/amrex"
AMREX_INSTALL_PREFIX="${WARPX_METAL_ROOT}/opt/amrex"
TEST_SRC_DIR="${WARPX_METAL_ROOT}/tests/amrex/heat_equation"
BUILD_DIR="${TEST_SRC_DIR}/build"
MACOS_SDK="$(xcrun --sdk macosx --show-sdk-path)"

# Verify prerequisites
if [ ! -x "${ACPP}" ]; then
    echo "[FAIL] acpp compiler not found at ${ACPP}"
    echo "       Run ./scripts/01-build-adaptivecpp.sh first."
    exit 1
fi

# Find AMReX cmake config (could be in lib/ or lib64/)
AMREX_CMAKE_DIR=""
for candidate in "${AMREX_INSTALL_PREFIX}/lib/cmake/AMReX" "${AMREX_INSTALL_PREFIX}/lib64/cmake/AMReX"; do
    if [ -f "${candidate}/AMReXConfig.cmake" ]; then
        AMREX_CMAKE_DIR="${candidate}"
        break
    fi
done
if [ -z "${AMREX_CMAKE_DIR}" ]; then
    echo "[FAIL] AMReX not found in ${AMREX_INSTALL_PREFIX}"
    echo "       Run ./scripts/03-build-amrex.sh first."
    exit 1
fi

echo "========================================"
echo "  AMReX HeatEquation Validation Test"
echo "========================================"
echo ""
echo "Compiler:    ${ACPP}"
echo "AMReX:       ${AMREX_INSTALL_PREFIX}"
echo "Test source: ${TEST_SRC_DIR}"
echo ""

echo "=== Step 1: Build HeatEquation test ==="

rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake "${TEST_SRC_DIR}" \
    -G Ninja \
    -DCMAKE_CXX_COMPILER="${ACPP}" \
    -DCMAKE_OSX_SYSROOT="${MACOS_SDK}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="${AMREX_INSTALL_PREFIX}" \
    -DAMREX_SRC_DIR="${AMREX_SOURCE_DIR}"

ninja -j"${NPROC}"

if [ ! -x "${BUILD_DIR}/heat_equation" ]; then
    echo "  [FAIL] heat_equation binary not found after build"
    exit 1
fi
echo "  [OK] HeatEquation compiled successfully"

echo ""
echo "=== Step 2: Run HeatEquation test ==="

# Copy inputs file to build dir for execution
cp "${TEST_SRC_DIR}/inputs" "${BUILD_DIR}/inputs"
cd "${BUILD_DIR}"

echo "  [..] Running HeatEquation (10 steps, 16^3 grid)..."
echo ""

OUTPUT=$("${BUILD_DIR}/heat_equation" inputs 2>&1) || {
    echo "${OUTPUT}"
    echo ""
    echo "  [FAIL] HeatEquation crashed"
    exit 1
}

echo "${OUTPUT}" | sed 's/^/    /'
echo ""

echo "=== Step 3: Validate output ==="

# Check that all 10 steps completed
PASS=true
for step in $(seq 1 10); do
    if echo "${OUTPUT}" | grep -q "Advanced step ${step}"; then
        echo "  [OK] Step ${step} completed"
    else
        echo "  [FAIL] Step ${step} not found in output"
        PASS=false
    fi
done

echo ""
echo "========================================"
if [ "${PASS}" = true ]; then
    echo "  RESULT: PASS"
    echo "  AMReX HeatEquation ran successfully on Metal GPU"
else
    echo "  RESULT: FAIL"
    echo "  Some steps did not complete — check output above"
    exit 1
fi
echo "========================================"
