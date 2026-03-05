#!/usr/bin/env bash
# 08-benchmark.sh — Automated GPU vs CPU benchmark runner for WarpX-on-Metal
#
# Runs each test case on GPU (Metal/SYCL) and CPU (OpenMP) back-to-back,
# captures TinyProfiler output, computes speedups, and writes results to
# benchmarks/RESULTS.md.
#
# Prerequisites:
#   ./scripts/05-build-warpx.sh  (GPU build)
#   ./scripts/07-build-warpx-cpu.sh  (CPU build)
#
# Usage: ./scripts/08-benchmark.sh [--gpu-only] [--cpu-only] [--quick]
#   --gpu-only  Skip CPU runs
#   --cpu-only  Skip GPU runs (useful if GPU build is broken)
#   --quick     Run only 2D small test (fast validation)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

# ── Configuration ──────────────────────────────────────────────────────────────
GPU_BUILD_DIR="${WARPX_SOURCE_DIR}/build-acpp/bin"
CPU_BUILD_DIR="${WARPX_SOURCE_DIR}/build-cpu/bin"
INPUTS_DIR="${WARPX_METAL_ROOT}/benchmarks/inputs"
RESULTS_DIR="${WARPX_METAL_ROOT}/benchmarks/results"
REPORT_FILE="${WARPX_METAL_ROOT}/benchmarks/RESULTS.md"

WARPX_GPU_2D="${GPU_BUILD_DIR}/warpx.2d.NOMPI.SYCL.SP.PSP.EB"
WARPX_GPU_3D="${GPU_BUILD_DIR}/warpx.3d.NOMPI.SYCL.SP.PSP.EB"
WARPX_CPU_2D="${CPU_BUILD_DIR}/warpx.2d.NOMPI.OMP.SP.PSP.EB"
WARPX_CPU_3D="${CPU_BUILD_DIR}/warpx.3d.NOMPI.OMP.SP.PSP.EB"

# Warm-up runs before timed runs (primes JIT cache for GPU)
WARMUP_RUNS=2
# Timed runs — report median
TIMED_RUNS=3
# CPU thread count
OMP_THREADS=12

# ── Parse flags ────────────────────────────────────────────────────────────────
RUN_GPU=true
RUN_CPU=true
QUICK=false
for arg in "$@"; do
    case "${arg}" in
        --gpu-only) RUN_CPU=false ;;
        --cpu-only) RUN_GPU=false ;;
        --quick)    QUICK=true ;;
    esac
done

# ── Validate executables ───────────────────────────────────────────────────────
check_exe() {
    local exe="$1" label="$2"
    if [ ! -x "${exe}" ]; then
        echo "  [WARN] ${label} not found: ${exe}"
        return 1
    fi
    return 0
}

echo ""
echo "=== WarpX-on-Metal Benchmark ==="
echo "  GPU 2D: ${WARPX_GPU_2D}"
echo "  GPU 3D: ${WARPX_GPU_3D}"
echo "  CPU 2D: ${WARPX_CPU_2D}"
echo "  CPU 3D: ${WARPX_CPU_3D}"
echo "================================"

if ${RUN_GPU}; then
    check_exe "${WARPX_GPU_2D}" "GPU 2D" || RUN_GPU=false
    check_exe "${WARPX_GPU_3D}" "GPU 3D" || RUN_GPU=false
fi
if ${RUN_CPU}; then
    check_exe "${WARPX_CPU_2D}" "CPU 2D" || RUN_CPU=false
    check_exe "${WARPX_CPU_3D}" "CPU 3D" || RUN_CPU=false
fi

if ! ${RUN_GPU} && ! ${RUN_CPU}; then
    echo "[FAIL] No executables found. Build WarpX first."
    exit 1
fi

mkdir -p "${RESULTS_DIR}"

# ── Helper: run one WarpX instance and extract wall time ──────────────────────
# Returns: total wall time in seconds (from TinyProfiler "Evolve" line)
run_warpx() {
    local label="$1"
    local exe="$2"
    local input="$3"
    local work_dir="$4"
    shift 4
    local extra_args=("$@")

    mkdir -p "${work_dir}"
    local log="${work_dir}/output.log"

    cd "${work_dir}"
    local exit_code=0
    "${exe}" "${input}" \
        diag1.intervals=99999 \
        warpx.verbose=0 \
        "${extra_args[@]}" \
        > "${log}" 2>&1 || exit_code=$?

    if [ ${exit_code} -ne 0 ]; then
        echo "    [FAIL] ${label} — exit ${exit_code}"
        tail -10 "${log}" | sed 's/^/    /'
        echo "0.0"
        return 0
    fi

    # Extract total Evolve time from TinyProfiler output
    # TinyProfiler prints: WarpX::Evolve()   1   <incl>   <incl>   <incl>   XX.XX%
    local evolve_time
    evolve_time=$(grep -E 'WarpX::Evolve\(\)' "${log}" 2>/dev/null | grep -v 'REG::\|BEGIN\|END' | awk '{v=$(NF-1)+0; if(v>max) max=v} END{print max}' || true)
    if [ -z "${evolve_time}" ]; then
        # Fallback: TinyProfiler total time line
        local total_time
        total_time=$(grep -oE 'TinyProfiler total time.*: ([0-9.]+)' "${log}" 2>/dev/null | grep -oE '[0-9.]+$' || true)
        evolve_time="${total_time:-0.0}"
    fi

    echo "${evolve_time}"
}

# ── Helper: compute median of N values ────────────────────────────────────────
median() {
    printf '%s\n' "$@" | sort -n | awk '
        { vals[NR] = $1 }
        END {
            n = NR
            if (n % 2 == 1) print vals[(n+1)/2]
            else printf "%.4f\n", (vals[n/2] + vals[n/2+1]) / 2.0
        }'
}

# ── Helper: compute speedup (CPU/GPU) ─────────────────────────────────────────
speedup() {
    local cpu_time="$1" gpu_time="$2"
    if [ "${gpu_time}" = "0.0" ] || [ "${gpu_time}" = "0" ]; then
        echo "N/A"
        return
    fi
    awk "BEGIN { printf \"%.2f\", ${cpu_time} / ${gpu_time} }"
}

# ── Helper: time per step ─────────────────────────────────────────────────────
per_step() {
    local total="$1" nstep="$2"
    awk "BEGIN { printf \"%.4f\", ${total} / ${nstep} }"
}

# ── Benchmark matrix ──────────────────────────────────────────────────────────
declare -a TEST_NAMES TEST_DIMS TEST_NX TEST_PPC TEST_STEPS
TEST_NAMES=("langmuir_2d_small" "langmuir_2d_large" "langmuir_3d_small" "langmuir_3d_large")
TEST_DIMS=(2 2 3 3)
TEST_NX=(128 512 64 128)
TEST_PPC=(2 2 1 1)
TEST_STEPS=(40 40 20 20)

if ${QUICK}; then
    TEST_NAMES=("langmuir_2d_small")
    TEST_DIMS=(2)
    TEST_NX=(128)
    TEST_PPC=(2)
    TEST_STEPS=(40)
    echo "  [INFO] Quick mode: running only langmuir_2d_small"
fi

NTESTS=${#TEST_NAMES[@]}

# ── Arrays to hold results ────────────────────────────────────────────────────
declare -a GPU_TIMES CPU_TIMES

# ── Run benchmarks ────────────────────────────────────────────────────────────
echo ""
echo "=== Running Benchmarks ==="

for (( i=0; i<NTESTS; i++ )); do
    name="${TEST_NAMES[$i]}"
    dim="${TEST_DIMS[$i]}"
    nx="${TEST_NX[$i]}"
    ppc="${TEST_PPC[$i]}"
    nstep="${TEST_STEPS[$i]}"

    input="${INPUTS_DIR}/langmuir_${dim}d_bench.txt"

    if [ "${dim}" = "2" ]; then
        gpu_exe="${WARPX_GPU_2D}"
        cpu_exe="${WARPX_CPU_2D}"
    else
        gpu_exe="${WARPX_GPU_3D}"
        cpu_exe="${WARPX_CPU_3D}"
    fi

    extra_args=("my_constants.NX=${nx}" "my_constants.PPC=${ppc}" "my_constants.NSTEP=${nstep}")

    echo ""
    echo "--- ${name} (${dim}D, ${nx}^${dim} grid, ${ppc}^${dim} ppc/species, ${nstep} steps) ---"

    # --- GPU ---
    gpu_median="0.0"
    if ${RUN_GPU}; then
        echo "  [GPU] Warm-up runs (${WARMUP_RUNS}x)..."
        for (( w=0; w<WARMUP_RUNS; w++ )); do
            run_warpx "warmup" "${gpu_exe}" "${input}" \
                "${RESULTS_DIR}/${name}/gpu_warmup_${w}" \
                "${extra_args[@]}" > /dev/null
        done

        echo "  [GPU] Timed runs (${TIMED_RUNS}x)..."
        gpu_run_times=()
        for (( r=0; r<TIMED_RUNS; r++ )); do
            t=$(run_warpx "gpu_run_${r}" "${gpu_exe}" "${input}" \
                "${RESULTS_DIR}/${name}/gpu_run_${r}" \
                "${extra_args[@]}")
            echo "    run $((r+1)): ${t}s"
            gpu_run_times+=("${t}")
        done
        gpu_median=$(median "${gpu_run_times[@]}")
        echo "  [GPU] Median: ${gpu_median}s  ($(per_step "${gpu_median}" "${nstep}")s/step)"
    fi
    GPU_TIMES[$i]="${gpu_median}"

    # --- CPU ---
    cpu_median="0.0"
    if ${RUN_CPU}; then
        echo "  [CPU] Timed runs (${TIMED_RUNS}x, OMP_NUM_THREADS=${OMP_THREADS})..."
        cpu_run_times=()
        for (( r=0; r<TIMED_RUNS; r++ )); do
            t=$(OMP_NUM_THREADS="${OMP_THREADS}" \
                run_warpx "cpu_run_${r}" "${cpu_exe}" "${input}" \
                "${RESULTS_DIR}/${name}/cpu_run_${r}" \
                "${extra_args[@]}")
            echo "    run $((r+1)): ${t}s"
            cpu_run_times+=("${t}")
        done
        cpu_median=$(median "${cpu_run_times[@]}")
        echo "  [CPU] Median: ${cpu_median}s  ($(per_step "${cpu_median}" "${nstep}")s/step)"
    fi
    CPU_TIMES[$i]="${cpu_median}"

    if ${RUN_GPU} && ${RUN_CPU}; then
        spdup=$(speedup "${cpu_median}" "${gpu_median}")
        echo "  [SPEEDUP] GPU vs CPU: ${spdup}x"
    fi
done

# ── Extract kernel breakdown from last GPU large run ─────────────────────────
echo ""
echo "=== Extracting Kernel Breakdown ==="

PROFILE_LOG=""
if [ -f "${RESULTS_DIR}/langmuir_3d_large/gpu_run_2/output.log" ]; then
    PROFILE_LOG="${RESULTS_DIR}/langmuir_3d_large/gpu_run_2/output.log"
elif [ -f "${RESULTS_DIR}/langmuir_3d_small/gpu_run_2/output.log" ]; then
    PROFILE_LOG="${RESULTS_DIR}/langmuir_3d_small/gpu_run_2/output.log"
elif [ -f "${RESULTS_DIR}/langmuir_2d_large/gpu_run_2/output.log" ]; then
    PROFILE_LOG="${RESULTS_DIR}/langmuir_2d_large/gpu_run_2/output.log"
fi

KERNEL_BREAKDOWN=""
if [ -n "${PROFILE_LOG}" ]; then
    # Extract TinyProfiler table
    KERNEL_BREAKDOWN=$(grep -E '^\s*[A-Za-z].*[0-9]+\s+[0-9.]' "${PROFILE_LOG}" 2>/dev/null | head -40 || true)
fi

# ── Generate RESULTS.md ───────────────────────────────────────────────────────
echo ""
echo "=== Writing benchmarks/RESULTS.md ==="

HARDWARE_INFO="Apple M4 Pro (12 CPU cores, 16 GPU CUs, 24 GB unified memory)"
TIMESTAMP="$(date '+%Y-%m-%d %H:%M:%S')"

{
cat << 'HEADER'
# WarpX-on-Metal Benchmark Results

HEADER

echo "**Hardware:** ${HARDWARE_INFO}"
echo "**Date:** ${TIMESTAMP}"
echo ""
echo "## GPU vs CPU Comparison"
echo ""
echo "CPU baseline uses 12 OpenMP threads (Apple Clang, libomp)."
echo "GPU uses Metal via AdaptiveCpp SSCP. GPU times include first-run JIT warm-up cost;"
echo "timed runs use cached JIT. All results are median of ${TIMED_RUNS} runs."
echo ""

echo "| Test | Grid | Particles/cell | Steps | CPU (12T) s/step | GPU (Metal) s/step | Speedup |"
echo "|------|------|---------------|-------|-------------------|---------------------|---------|"

for (( i=0; i<NTESTS; i++ )); do
    name="${TEST_NAMES[$i]}"
    dim="${TEST_DIMS[$i]}"
    nx="${TEST_NX[$i]}"
    ppc="${TEST_PPC[$i]}"
    nstep="${TEST_STEPS[$i]}"
    gpu_t="${GPU_TIMES[$i]}"
    cpu_t="${CPU_TIMES[$i]}"

    if [ "${dim}" = "2" ]; then
        grid="${nx}x${nx}"
        ppc_str="${ppc}^2=${ppc}*${ppc}"
    else
        grid="${nx}^3"
        ppc_str="${ppc}^3"
    fi

    cpu_per=""
    gpu_per=""
    spdup="N/A"

    if ${RUN_CPU} && [ "${cpu_t}" != "0.0" ] && [ "${cpu_t}" != "0" ]; then
        cpu_per=$(per_step "${cpu_t}" "${nstep}")
    else
        cpu_per="(skipped)"
    fi

    if ${RUN_GPU} && [ "${gpu_t}" != "0.0" ] && [ "${gpu_t}" != "0" ]; then
        gpu_per=$(per_step "${gpu_t}" "${nstep}")
    else
        gpu_per="(skipped)"
    fi

    if ${RUN_GPU} && ${RUN_CPU} && [ "${gpu_t}" != "0.0" ] && [ "${cpu_t}" != "0.0" ]; then
        spdup=$(speedup "${cpu_t}" "${gpu_t}")x
    fi

    echo "| ${name} | ${grid} | ${ppc_str} | ${nstep} | ${cpu_per} | ${gpu_per} | ${spdup} |"
done

echo ""
echo "## Kernel Breakdown (GPU)"
echo ""

if [ -n "${PROFILE_LOG}" ]; then
    echo "Source: \`$(basename "${PROFILE_LOG%/output.log}")\`"
    echo ""
    echo "\`\`\`"
    if [ -n "${KERNEL_BREAKDOWN}" ]; then
        echo "${KERNEL_BREAKDOWN}"
    else
        grep -A 60 'TinyProfiler' "${PROFILE_LOG}" | head -60 || echo "(TinyProfiler output not found in log)"
    fi
    echo "\`\`\`"
else
    echo "(No GPU run logs found — run \`./scripts/08-benchmark.sh\` to populate)"
fi

echo ""
echo "## Sort Interval Sensitivity"
echo ""
echo "Run \`./scripts/08-benchmark.sh\` with sort_intervals overrides to populate:"
echo ""
echo "| sort_intervals | Test | GPU s/step |"
echo "|---------------|------|-----------|"
echo "| 4 (default) | langmuir_2d_small | (run to populate) |"
echo "| -1 (disabled) | langmuir_2d_small | (run to populate) |"
echo "| 20 | langmuir_2d_small | (run to populate) |"
echo ""
echo "## Notes"
echo ""
echo "- GPU first-run includes JIT compilation (LLVM IR -> MSL). Subsequent runs use cache at \`~/.acpp/apps/global/jit-cache/\`."
echo "- Single precision (FP32) throughout — Metal has no FP64 support."
echo "- PSATD spectral solver unavailable (no Metal FFT). FDTD only."
echo "- CPU build: Apple Clang, libomp, \`-O3 -DNDEBUG\`."
echo "- GPU build: AdaptiveCpp SSCP + LLVM 20 -> MSL -> Apple GPU."

} > "${REPORT_FILE}"

echo "  [OK] Written: ${REPORT_FILE}"
echo ""
echo "=== Benchmark Complete ==="

if ${RUN_GPU} && ${RUN_CPU}; then
    echo ""
    echo "Summary:"
    for (( i=0; i<NTESTS; i++ )); do
        name="${TEST_NAMES[$i]}"
        nstep="${TEST_STEPS[$i]}"
        spdup=$(speedup "${CPU_TIMES[$i]}" "${GPU_TIMES[$i]}")
        echo "  ${name}: GPU=$(per_step "${GPU_TIMES[$i]}" "${nstep}")s/step  CPU=$(per_step "${CPU_TIMES[$i]}" "${nstep}")s/step  speedup=${spdup}x"
    done
fi
echo ""
echo "Full results: ${REPORT_FILE}"
echo "Raw logs:     ${RESULTS_DIR}/"
echo ""
echo "Next step: ./scripts/09-profile-metal.sh  (Metal GPU timeline via xctrace)"
