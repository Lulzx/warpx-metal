#!/usr/bin/env bash
# 02-validate-metal.sh — Compile and run SYCL validation tests on Metal backend
#
# Prerequisites: Run 01-build-adaptivecpp.sh first
#
# Usage: ./scripts/02-validate-metal.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

ACPP="${ACPP_INSTALL_PREFIX}/bin/acpp"
TEST_DIR="${WARPX_METAL_ROOT}/tests/sycl"
PASS=0
FAIL=0
SKIP=0
RESULTS=()

if [ ! -x "${ACPP}" ]; then
    echo "[FAIL] acpp compiler not found at ${ACPP}"
    echo "       Run ./scripts/01-build-adaptivecpp.sh first."
    exit 1
fi

compile_and_run() {
    local name="$1"
    local src="${TEST_DIR}/${name}.cpp"
    local bin="${TEST_DIR}/${name}"
    local description="$2"

    echo ""
    echo "=== Test: ${name} ==="
    echo "    ${description}"

    if [ ! -f "${src}" ]; then
        echo "    [SKIP] Source file not found: ${src}"
        SKIP=$((SKIP + 1))
        RESULTS+=("SKIP  ${name}: source not found")
        return
    fi

    # Compile
    echo "    [..] Compiling..."
    if ! "${ACPP}" -O2 -o "${bin}" "${src}" 2>&1; then
        echo "    [FAIL] Compilation failed"
        FAIL=$((FAIL + 1))
        RESULTS+=("FAIL  ${name}: compilation error")
        return
    fi

    # Run
    echo "    [..] Running..."
    local output
    if output=$("${bin}" 2>&1); then
        echo "${output}" | sed 's/^/    /'
        if echo "${output}" | grep -q "PASS"; then
            echo "    [PASS]"
            PASS=$((PASS + 1))
            RESULTS+=("PASS  ${name}")
        else
            echo "    [FAIL] Ran but did not print PASS"
            FAIL=$((FAIL + 1))
            RESULTS+=("FAIL  ${name}: no PASS in output")
        fi
    else
        echo "${output}" | sed 's/^/    /'
        echo "    [FAIL] Runtime error (exit code $?)"
        FAIL=$((FAIL + 1))
        RESULTS+=("FAIL  ${name}: runtime error")
    fi
}

echo "========================================"
echo "  SYCL Metal Backend Validation Tests"
echo "========================================"
echo ""
echo "Compiler: ${ACPP}"
"${ACPP}" --version 2>&1 | head -1 || true

# Run tests in order of increasing complexity
compile_and_run "device_query"    "Enumerate SYCL devices, confirm Metal GPU"
compile_and_run "vector_add"      "Basic parallel_for with buffer/accessor model"
compile_and_run "usm_test"        "sycl::malloc_shared — critical for AMReX"
compile_and_run "reduction_test"  "Atomic float add — critical for deposition"

echo ""
echo "========================================"
echo "  Results Summary"
echo "========================================"
for r in "${RESULTS[@]}"; do
    echo "  ${r}"
done
echo ""
echo "  PASS: ${PASS}  FAIL: ${FAIL}  SKIP: ${SKIP}"
echo "========================================"

if [ "${FAIL}" -gt 0 ]; then
    echo ""
    echo "Some tests failed. See docs/known-issues.md for workarounds."
    exit 1
fi
