#!/usr/bin/env bash

set -Eeuo pipefail

# This is host-platform CI. On macOS it builds WarpX's Apple CPU path, not
# Linux. A second syntax gate undefines __APPLE__ for guarded WarpX translation
# units where feasible; that catches guard leakage but does not reproduce a
# Linux kernel, libc, toolchain, ABI, or runtime.

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
AMREX_REPO="https://github.com/AMReX-Codes/amrex.git"
AMREX_REF="fa795322b44fff24fef3a795c3b00d24e015ee42"
AMREX_RELEASE_TAG="26.06"
WARPX_REPO="https://github.com/ECP-WarpX/WarpX.git"
WARPX_REF="f7db079f9a8ca96d179e02709a29fe6c027ed8ed"
WARPX_RELEASE_TAG="26.06"
BUILD_JOBS="${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
HOST_OS="$(uname -s)"
PATCH_COUNT=0
BINARY_COUNT=0
WORK_DIR=""

finish() {
    local status="$1"
    local outcome="FAIL"

    if [[ "${KEEP_LOCAL_CI_WORKDIR:-0}" != "1" && -n "${WORK_DIR}" ]]; then
        rm -rf -- "${WORK_DIR}" || true
    fi
    if [[ "${status}" -eq 0 ]]; then
        outcome="PASS"
    fi

    printf 'LOCAL_CI_RESULT outcome=%s host=%s patches=%d binaries=%d\n' \
        "${outcome}" "${HOST_OS}" "${PATCH_COUNT}" "${BINARY_COUNT}"
}
trap 'finish "$?"' EXIT

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

for tool in git cmake ninja python3; do
    command -v "${tool}" >/dev/null 2>&1 || fail "required tool not found: ${tool}"
done

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/warpx-metal-local-ci.XXXXXX")"
AMREX_SOURCE_DIR="${WORK_DIR}/amrex"
WARPX_SOURCE_DIR="${WORK_DIR}/warpx"
BUILD_DIR="${WORK_DIR}/build"

clone_pinned() {
    local repository="$1"
    local revision="$2"
    local release_tag="$3"
    local destination="$4"
    local label="$5"
    local tag_commit

    git clone --filter=blob:none "${repository}" "${destination}"
    git -C "${destination}" checkout --detach "${revision}"
    [[ "$(git -C "${destination}" rev-parse HEAD)" == "${revision}" ]] || \
        fail "${label} checkout did not resolve to the pinned revision"

    git -C "${destination}" fetch --force --filter=blob:none origin \
        "refs/tags/${release_tag}:refs/tags/${release_tag}"
    tag_commit="$(git -C "${destination}" rev-parse "${release_tag}^{commit}")"
    git -C "${destination}" merge-base --is-ancestor "${revision}" "${tag_commit}" || \
        fail "${label} pin is not an ancestor of upstream tag ${release_tag}"

    printf '%s pin=%s upstream_tag=%s tag_commit=%s\n' \
        "${label}" "${revision}" "${release_tag}" "${tag_commit}"
}

copy_amrex_replacement() {
    local source_name="$1"
    local target_name="$2"
    local source_file="${ROOT_DIR}/patches/amrex/${source_name}"
    local target_file="${AMREX_SOURCE_DIR}/${target_name}"

    [[ -f "${source_file}" ]] || fail "missing AMReX replacement: ${source_name}"
    [[ -f "${target_file}" ]] || fail "missing AMReX target: ${target_name}"
    cp "${source_file}" "${target_file}"
    printf 'AMReX replacement applied: %s\n' "${target_name}"
}

apply_patch() {
    local repository="$1"
    local patch_file="$2"

    [[ -f "${patch_file}" ]] || fail "missing patch: ${patch_file}"
    git -C "${repository}" apply --check "${patch_file}"
    git -C "${repository}" apply "${patch_file}"
    PATCH_COUNT=$((PATCH_COUNT + 1))
    printf 'Patch applied: %s\n' "${patch_file#"${ROOT_DIR}/"}"
}

clone_pinned "${AMREX_REPO}" "${AMREX_REF}" "${AMREX_RELEASE_TAG}" \
    "${AMREX_SOURCE_DIR}" "AMReX"
clone_pinned "${WARPX_REPO}" "${WARPX_REF}" "${WARPX_RELEASE_TAG}" \
    "${WARPX_SOURCE_DIR}" "WarpX"

copy_amrex_replacement "AMReXSYCL.cmake" "Tools/CMake/AMReXSYCL.cmake"
copy_amrex_replacement "AMReX_RandomEngine.H" "Src/Base/AMReX_RandomEngine.H"
copy_amrex_replacement "AMReX_Random.cpp" "Src/Base/AMReX_Random.cpp"

apply_patch "${AMREX_SOURCE_DIR}" \
    "${ROOT_DIR}/patches/amrex-post/0004-metal-pic-rng-reduction-fixes.patch"
apply_patch "${WARPX_SOURCE_DIR}" \
    "${ROOT_DIR}/patches/warpx/0001-metal-source-identity.patch"
apply_patch "${WARPX_SOURCE_DIR}" \
    "${ROOT_DIR}/patches/warpx/0002-fix-addplasma-metal-parser-momentum-oob.patch"
apply_patch "${WARPX_SOURCE_DIR}" \
    "${ROOT_DIR}/patches/warpx/0003-correct-macos-memory-guard-available-memory.patch"

(( PATCH_COUNT > 0 )) || fail "no patches were applied"
git -C "${AMREX_SOURCE_DIR}" diff --check
git -C "${WARPX_SOURCE_DIR}" diff --check

if [[ "${HOST_OS}" == "Darwin" ]]; then
    CC_BIN="${CC:-$(xcrun --find clang)}"
    CXX_BIN="${CXX:-$(xcrun --find clang++)}"
else
    CC_BIN="${CC:-$(command -v gcc || command -v cc)}"
    CXX_BIN="${CXX:-$(command -v g++ || command -v c++)}"
fi

cmake_args=(
    -G Ninja
    "-DCMAKE_C_COMPILER=${CC_BIN}"
    "-DCMAKE_CXX_COMPILER=${CXX_BIN}"
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    -DAMReX_GPU_BACKEND=NONE
    -DAMReX_MPI=OFF
    -DAMReX_OMP=ON
    -DAMReX_FORTRAN=OFF
    -DWarpX_COMPUTE=OMP
    -DWarpX_PRECISION=SINGLE
    -DWarpX_PARTICLE_PRECISION=SINGLE
    '-DWarpX_DIMS=2;3'
    -DWarpX_MPI=OFF
    -DWarpX_FFT=OFF
    -DWarpX_QED=OFF
    -DWarpX_OPENPMD=OFF
    "-DWarpX_amrex_src=${AMREX_SOURCE_DIR}"
)

if [[ "${HOST_OS}" == "Darwin" ]] && command -v brew >/dev/null 2>&1; then
    LIBOMP_PREFIX="$(brew --prefix libomp)"
    cmake_args+=("-DCMAKE_PREFIX_PATH=${LIBOMP_PREFIX}" "-DOpenMP_ROOT=${LIBOMP_PREFIX}")
fi

printf 'Configuring host CPU build: host=%s cc=%s cxx=%s\n' \
    "${HOST_OS}" "${CC_BIN}" "${CXX_BIN}"
cmake -S "${WARPX_SOURCE_DIR}" -B "${BUILD_DIR}" "${cmake_args[@]}"

if [[ "${HOST_OS}" == "Darwin" ]]; then
    printf '%s\n' \
        'Running best-effort non-Apple syntax gate with -U__APPLE__.' \
        'This checks guarded translation units only; it is not a Linux build.'
    python3 - "${BUILD_DIR}/compile_commands.json" \
        "${WARPX_SOURCE_DIR}/Source/Evolve/WarpXEvolve.cpp" \
        "${WARPX_SOURCE_DIR}/Source/main.cpp" <<'PY'
import json
import os
import shlex
import subprocess
import sys

database_path, *sources = sys.argv[1:]
with open(database_path, encoding="utf-8") as stream:
    database = json.load(stream)

by_file = {}
for entry in database:
    entry_file = entry["file"]
    if not os.path.isabs(entry_file):
        entry_file = os.path.join(entry["directory"], entry_file)
    by_file[os.path.realpath(entry_file)] = entry
for source in sources:
    source = os.path.realpath(source)
    entry = by_file.get(source)
    if entry is None:
        raise SystemExit(f"compile command not found for {source}")

    command = list(entry.get("arguments") or shlex.split(entry["command"]))
    filtered = []
    skip_next = False
    for argument in command:
        if skip_next:
            skip_next = False
            continue
        if argument in {"-o", "-MF", "-MT", "-MQ"}:
            skip_next = True
            continue
        argument_path = argument if os.path.isabs(argument) \
            else os.path.join(entry["directory"], argument)
        if argument in {"-c", "-MD", "-MMD"} or os.path.realpath(argument_path) == source:
            continue
        filtered.append(argument)

    # Apple libc++ also keys availability and thread selection on __APPLE__.
    # Keep the host headers usable while selecting WarpX's non-Apple branches.
    filtered.extend([
        "-U__APPLE__",
        "-D__FreeBSD__",
        "-D_LIBCPP_DISABLE_AVAILABILITY",
        "-fsyntax-only",
        source,
    ])
    print(f"Syntax gate: {os.path.basename(source)}", flush=True)
    subprocess.run(filtered, cwd=entry["directory"], check=True)
PY
fi

cmake --build "${BUILD_DIR}" --parallel "${BUILD_JOBS}"

binaries=()
while IFS= read -r binary; do
    [[ -x "${binary}" ]] && binaries+=("${binary}")
done < <(find "${BUILD_DIR}/bin" -maxdepth 1 -type f -name 'warpx.*' -print | sort)

BINARY_COUNT="${#binaries[@]}"
(( BINARY_COUNT > 0 )) || fail "no linked WarpX executable found"
printf 'Linked executable: %s\n' "${binaries[@]}"

printf '%s\n' \
    "Host CPU build passed on ${HOST_OS}." \
    'A macOS PASS plus the -U__APPLE__ syntax gate does not prove Linux/GCC behavior.'
