#!/usr/bin/env bash
# 09-profile-metal.sh — Capture Metal GPU timeline via xctrace
#
# Records a Metal System Trace while running WarpX on the GPU.
# Output: benchmarks/profiles/<test>.trace  (open in Instruments.app)
#
# Captures: GPU kernel occupancy, ALU utilization, memory bandwidth,
#           encoder gaps, command buffer timeline.
#
# Prerequisites: Run 05-build-warpx.sh first (GPU build).
#                Xcode must be installed (provides xctrace).
#
# Usage: ./scripts/09-profile-metal.sh [test_name]
#   test_name: langmuir_2d_small (default) | langmuir_2d_large | langmuir_3d_small | langmuir_3d_large

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

GPU_BUILD_DIR="${WARPX_SOURCE_DIR}/build-acpp/bin"
INPUTS_DIR="${WARPX_METAL_ROOT}/benchmarks/inputs"
PROFILES_DIR="${WARPX_METAL_ROOT}/benchmarks/profiles"

WARPX_GPU_2D="${GPU_BUILD_DIR}/warpx.2d.NOMPI.SYCL.SP.PSP.EB"
WARPX_GPU_3D="${GPU_BUILD_DIR}/warpx.3d.NOMPI.SYCL.SP.PSP.EB"

# ── Parse test name ───────────────────────────────────────────────────────────
TEST_NAME="${1:-langmuir_2d_small}"

case "${TEST_NAME}" in
    langmuir_2d_small) DIM=2; NX=128; PPC=2; NSTEP=40 ;;
    langmuir_2d_large) DIM=2; NX=512; PPC=2; NSTEP=40 ;;
    langmuir_3d_small) DIM=3; NX=64;  PPC=1; NSTEP=20 ;;
    langmuir_3d_large) DIM=3; NX=128; PPC=1; NSTEP=20 ;;
    *)
        echo "[FAIL] Unknown test: ${TEST_NAME}"
        echo "       Valid tests: langmuir_2d_small langmuir_2d_large langmuir_3d_small langmuir_3d_large"
        exit 1
        ;;
esac

if [ "${DIM}" = "2" ]; then
    WARPX_EXE="${WARPX_GPU_2D}"
else
    WARPX_EXE="${WARPX_GPU_3D}"
fi

INPUT="${INPUTS_DIR}/langmuir_${DIM}d_bench.txt"
TRACE_FILE="${PROFILES_DIR}/${TEST_NAME}.trace"
WORK_DIR="${PROFILES_DIR}/${TEST_NAME}_work"

echo ""
echo "=== Metal GPU Profiler ==="
echo "  Test:   ${TEST_NAME}"
echo "  Exe:    ${WARPX_EXE}"
echo "  Input:  ${INPUT}"
echo "  Trace:  ${TRACE_FILE}"
echo "=========================="

# ── Validate ──────────────────────────────────────────────────────────────────
if [ ! -x "${WARPX_EXE}" ]; then
    echo "[FAIL] WarpX GPU executable not found: ${WARPX_EXE}"
    echo "       Run ./scripts/05-build-warpx.sh first."
    exit 1
fi

if ! command -v xctrace &>/dev/null; then
    echo "[FAIL] xctrace not found. Install Xcode from the App Store."
    exit 1
fi

# ── Warm up JIT cache first (xctrace adds overhead — don't mix with JIT) ──────
echo ""
echo "=== Step 1: JIT warm-up (2 runs without tracing) ==="
mkdir -p "${WORK_DIR}/warmup"
for w in 0 1; do
    echo "  [..] Warm-up ${w}..."
    cd "${WORK_DIR}/warmup"
    "${WARPX_EXE}" "${INPUT}" \
        NX="${NX}" PPC="${PPC}" NSTEP="${NSTEP}" my_constants.NSTEP="${NSTEP}" \
        diag1.intervals=99999 \
        warpx.verbose=0 \
        > "${WORK_DIR}/warmup_${w}.log" 2>&1
    echo "  [OK] Warm-up ${w} complete"
done

# ── Record Metal System Trace ─────────────────────────────────────────────────
echo ""
echo "=== Step 2: Recording Metal System Trace ==="
echo "  (This may take ${NSTEP} simulation steps — be patient)"

mkdir -p "${PROFILES_DIR}"
mkdir -p "${WORK_DIR}/trace_run"
cd "${WORK_DIR}/trace_run"

# Remove old trace file if it exists
rm -rf "${TRACE_FILE}"

# Build the WarpX command as a string for xctrace
WARPX_CMD="${WARPX_EXE} ${INPUT} NX=${NX} PPC=${PPC} NSTEP=${NSTEP} my_constants.NSTEP=${NSTEP} diag1.intervals=99999 warpx.verbose=0"

xctrace record \
    --output "${TRACE_FILE}" \
    --template "Metal System Trace" \
    --launch -- \
    "${WARPX_EXE}" \
    "${INPUT}" \
    "NX=${NX}" \
    "PPC=${PPC}" \
    "NSTEP=${NSTEP}" \
    "my_constants.NSTEP=${NSTEP}" \
    "diag1.intervals=99999" \
    "warpx.verbose=0"

echo ""
if [ -e "${TRACE_FILE}" ]; then
    echo "  [OK] Trace written: ${TRACE_FILE}"
    echo ""
    echo "=== Step 3: Open in Instruments ==="
    echo "  open \"${TRACE_FILE}\""
    echo ""
    echo "  Key views in Instruments:"
    echo "    Metal > GPU Timeline — per-kernel execution timeline"
    echo "    Metal > Occupancy — threadgroup utilization per CU"
    echo "    Metal > Memory Bandwidth — reads/writes per cycle"
    echo "    Metal > ALU Utilization — compute throughput"
    echo "    Metal > Command Buffer — encoder gaps (CPU-to-GPU latency)"
    echo ""

    # Also try to export a summary via xctrace export (if available)
    if xctrace help export &>/dev/null 2>&1; then
        SUMMARY_JSON="${PROFILES_DIR}/${TEST_NAME}_summary.json"
        echo "  [..] Exporting summary to JSON..."
        xctrace export \
            --input "${TRACE_FILE}" \
            --xpath '//trace-toc/run/tracks/track[@name="Metal"]' \
            > "${SUMMARY_JSON}" 2>/dev/null || true
        if [ -s "${SUMMARY_JSON}" ]; then
            echo "  [OK] Summary JSON: ${SUMMARY_JSON}"
        fi
    fi

    # Open trace automatically if running interactively
    if [ -t 1 ]; then
        echo "  [..] Opening Instruments..."
        open "${TRACE_FILE}" || true
    fi
else
    echo "  [FAIL] Trace file not created. Check xctrace output above."
    exit 1
fi

echo ""
echo "=== Metal Debugging Tips ==="
echo ""
echo "  For register pressure / shader analysis:"
echo "    export MTL_SHADER_VALIDATION=1"
echo "    export MTL_DEBUG_LAYER=1"
echo "    ${WARPX_EXE} ..."
echo ""
echo "  For GPU debug frame capture, run WarpX from Xcode:"
echo "    Product > Profile > Metal System Trace"
echo ""
echo "  Per-GPU metric env vars (useful for bandwidth-bound analysis):"
echo "    METAL_CAPTURE_ENABLED=1  — enable programmatic capture"
echo "    MTL_HUD_ENABLED=1        — overlay HUD in windowed apps"
