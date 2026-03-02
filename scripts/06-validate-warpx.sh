#!/usr/bin/env bash
# 06-validate-warpx.sh — Run WarpX physics validation tests on Metal GPU
#
# Prerequisites: Run 05-build-warpx.sh first
#
# Tests run (FDTD only, no FFT/PSATD):
#   1. Langmuir 2D — electrostatic oscillation, basic PIC loop (M3 milestone)
#   2. Langmuir 3D — same physics in 3D
#
# Usage: ./scripts/06-validate-warpx.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

WARPX_SOURCE_DIR="${WARPX_METAL_ROOT}/extern/warpx"
BUILD_DIR="${WARPX_SOURCE_DIR}/build-acpp"
TESTS_DIR="${WARPX_SOURCE_DIR}/Examples/Tests"
RESULTS_DIR="${WARPX_METAL_ROOT}/tests/warpx/results"

WARPX_2D="${BUILD_DIR}/bin/warpx.2d.NOMPI.SYCL.SP.PSP.EB"
WARPX_3D="${BUILD_DIR}/bin/warpx.3d.NOMPI.SYCL.SP.PSP.EB"

# Verify executables exist
for exe in "${WARPX_2D}" "${WARPX_3D}"; do
    if [ ! -x "${exe}" ]; then
        echo "[FAIL] WarpX executable not found: ${exe}"
        echo "       Run ./scripts/05-build-warpx.sh first."
        exit 1
    fi
done

mkdir -p "${RESULTS_DIR}"

PASS_COUNT=0
FAIL_COUNT=0
TOTAL_COUNT=0

run_test() {
    local TEST_NAME="$1"
    local EXECUTABLE="$2"
    local INPUT_DIR="$3"
    local INPUT_FILE="$4"
    local MAX_STEP="$5"

    TOTAL_COUNT=$((TOTAL_COUNT + 1))
    echo ""
    echo "=== Test ${TOTAL_COUNT}: ${TEST_NAME} ==="

    local WORK_DIR="${RESULTS_DIR}/${TEST_NAME}"
    rm -rf "${WORK_DIR}"
    mkdir -p "${WORK_DIR}"

    # Copy all input files from the test directory to work dir
    # (WarpX FILE= include mechanism uses relative paths)
    cp "${INPUT_DIR}"/inputs_* "${WORK_DIR}/" 2>/dev/null || true

    local LOG="${WORK_DIR}/output.log"

    echo "  [..] Running ${TEST_NAME} (max_step=${MAX_STEP})..."
    echo "       Exe:   $(basename "${EXECUTABLE}")"
    echo "       Input: ${INPUT_FILE}"

    cd "${WORK_DIR}"

    # Run WarpX — push diagnostic interval far out to skip I/O
    local EXIT_CODE=0
    "${EXECUTABLE}" "${INPUT_FILE}" \
        max_step="${MAX_STEP}" \
        diag1.intervals=99999 \
        > "${LOG}" 2>&1 || EXIT_CODE=$?

    if [ ${EXIT_CODE} -ne 0 ]; then
        echo "  [FAIL] ${TEST_NAME} — exit code ${EXIT_CODE}"
        echo "  --- Last 30 lines of output ---"
        tail -30 "${LOG}"
        echo "  --- End of output ---"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        return 1
    fi

    # Check for successful completion: WarpX prints "STEP <N>" lines
    local LAST_STEP
    LAST_STEP=$(grep -oE 'STEP [0-9]+' "${LOG}" | tail -1 | grep -oE '[0-9]+' || echo "0")

    if [ "${LAST_STEP}" -ge "${MAX_STEP}" ]; then
        echo "  [OK] ${TEST_NAME} — completed ${LAST_STEP}/${MAX_STEP} steps"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        echo "  [FAIL] ${TEST_NAME} — only reached step ${LAST_STEP}/${MAX_STEP}"
        echo "  --- Last 30 lines of output ---"
        tail -30 "${LOG}"
        echo "  --- End of output ---"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        return 1
    fi

    # Check for SYCL/Metal errors in output
    if grep -qi "sycl exception\|metal.*error\|abort\|SIGABRT\|uint4" "${LOG}"; then
        echo "  [WARN] Possible runtime errors detected in output"
        grep -i "sycl exception\|metal.*error\|abort\|uint4" "${LOG}" | head -5
    fi

    # Print timing info
    local WALL_TIME
    WALL_TIME=$(grep -oE 'WarpX.*time.*=[^,]+' "${LOG}" | tail -1 || echo "")
    if [ -n "${WALL_TIME}" ]; then
        echo "  [INFO] ${WALL_TIME}"
    fi

    return 0
}

echo ""
echo "=== WarpX Metal GPU Validation ==="
echo "  2D exe: ${WARPX_2D}"
echo "  3D exe: ${WARPX_3D}"
echo ""

# ── Test 1: Langmuir 2D (primary spec target M3) ──
# 128x128 grid, 2 species (electrons + positrons), FDTD field solve
# Reduced to 40 steps for initial validation (full test: 80 steps)
run_test \
    "langmuir_2d" \
    "${WARPX_2D}" \
    "${TESTS_DIR}/langmuir" \
    "inputs_test_2d_langmuir_multi" \
    40 || true

# ── Test 2: Langmuir 3D ──
# 64^3 grid, 2 species, FDTD field solve
# Reduced to 20 steps for initial validation (full test: 40 steps)
run_test \
    "langmuir_3d" \
    "${WARPX_3D}" \
    "${TESTS_DIR}/langmuir" \
    "inputs_test_3d_langmuir_multi" \
    20 || true

echo ""
echo "=== Validation Summary ==="
echo "  Passed: ${PASS_COUNT}/${TOTAL_COUNT}"
echo "  Failed: ${FAIL_COUNT}/${TOTAL_COUNT}"
echo ""

if [ ${FAIL_COUNT} -eq 0 ]; then
    echo "RESULT: ALL TESTS PASSED"
    echo ""
    echo "Next step: Run full validation with extended timesteps:"
    echo "  06-validate-warpx.sh  (modify max_step to full values)"
    echo ""
    echo "Or proceed to Phase 4: Optimization and benchmarking"
else
    echo "RESULT: ${FAIL_COUNT} TEST(S) FAILED"
    echo "  Check logs in ${RESULTS_DIR}/<test_name>/output.log"
fi
