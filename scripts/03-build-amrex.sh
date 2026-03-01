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

if [ -d "${AMREX_SOURCE_DIR}/.git" ]; then
    echo "  [OK] AMReX already cloned at ${AMREX_SOURCE_DIR}"
    cd "${AMREX_SOURCE_DIR}"
    # Reset any previously applied patches before re-applying
    git checkout -- .
    git clean -fd
else
    echo "  [..] Cloning AMReX..."
    git clone https://github.com/AMReX-Codes/amrex.git "${AMREX_SOURCE_DIR}"
fi

echo ""
echo "=== Step 2: Apply patches ==="

PATCH_DIR="${PATCHES_DIR}/amrex"

# Apply .patch files via git apply
PATCH_COUNT=$(find "${PATCH_DIR}" -name '*.patch' 2>/dev/null | wc -l | tr -d ' ')
if [ "${PATCH_COUNT}" -gt 0 ]; then
    cd "${AMREX_SOURCE_DIR}"
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
fi

# Replace files with AdaptiveCpp-compatible versions
replace_file() {
    local src="${PATCH_DIR}/$1"
    local dst="${AMREX_SOURCE_DIR}/$2"
    if [ -f "${src}" ]; then
        echo "  [..] Patching $2..."
        cp "${src}" "${dst}"
        echo "  [OK] $2 patched"
    else
        echo "  [WARN] ${src} not found"
    fi
}

replace_file "AMReXSYCL.cmake"      "Tools/CMake/AMReXSYCL.cmake"
replace_file "AMReX_RandomEngine.H"  "Src/Base/AMReX_RandomEngine.H"
replace_file "AMReX_Random.cpp"      "Src/Base/AMReX_Random.cpp"

# Surgical source-level fixes via Python
python3 -c "
import re, sys

# Fix 1: AMReX_INT.H — Disable __int128 when AMREX_NO_INT128 is defined.
#   The Metal emitter cannot translate i128 (maps to uint4 in MSL, casts unsupported).
#   AMREX_NO_INT128 is set via AMReXSYCL.cmake for AdaptiveCpp builds.
#   All i128 codepaths (umulhi, FastDivmodU64) have safe non-128 fallbacks.
f = '${AMREX_SOURCE_DIR}/Src/Base/AMReX_INT.H'
with open(f) as fh: s = fh.read()
old_guard = '#if (defined(__x86_64) || defined (__aarch64__)) && !defined(_WIN32) && (defined(__GNUC__) || defined(__clang__)) && !defined(__NVCOMPILER)'
new_guard = '#if (defined(__x86_64) || defined (__aarch64__)) && !defined(_WIN32) && (defined(__GNUC__) || defined(__clang__)) && !defined(__NVCOMPILER) && !defined(AMREX_NO_INT128)'
s = s.replace(old_guard, new_guard)
with open(f,'w') as fh: fh.write(s)
print('  [OK] AMReX_INT.H patched (disable __int128 when AMREX_NO_INT128)')

# Fix 2: AMReX_GpuAsyncArray.H — host_task not in AdaptiveCpp
f = '${AMREX_SOURCE_DIR}/Src/Base/AMReX_GpuAsyncArray.H'
with open(f) as fh: s = fh.read()
old = '''                    q.submit([&] (sycl::handler& h) {
                        h.host_task([=] () {
                            The_Arena()->free(pd);
                            The_Pinned_Arena()->free(ph);
                        });
                    });
                } catch (sycl::exception const& ex) {
                    amrex::Abort(std::string(\"host_task: \")+ex.what()+\"!!!!!\");'''
new = '''                    q.wait();
                    The_Arena()->free(pd);
                    The_Pinned_Arena()->free(ph);
                } catch (sycl::exception const& ex) {
                    amrex::Abort(std::string(\"async cleanup: \")+ex.what()+\"!!!!!\");'''
s = s.replace(old, new)
with open(f,'w') as fh: fh.write(s)
print('  [OK] AMReX_GpuAsyncArray.H patched (host_task replacement)')

# Fix 3: AMReX_GpuElixir.cpp — host_task not in AdaptiveCpp
f = '${AMREX_SOURCE_DIR}/Src/Base/AMReX_GpuElixir.cpp'
with open(f) as fh: s = fh.read()
old = '''        auto& q = *(Gpu::gpuStream().queue);
        try {
            q.submit([&] (sycl::handler& h) {
                h.host_task([=] () {
                    for (auto const& pa : lpa) {
                        pa.second->free(pa.first);
                    }
                });
            });
        } catch (sycl::exception const& ex) {
            amrex::Abort(std::string(\"host_task: \")+ex.what()+\"!!!!!\");'''
new = '''        auto& q = *(Gpu::gpuStream().queue);
        try {
            q.wait();
            for (auto const& pa : lpa) {
                pa.second->free(pa.first);
            }
        } catch (sycl::exception const& ex) {
            amrex::Abort(std::string(\"async cleanup: \")+ex.what()+\"!!!!!\");'''
s = s.replace(old, new)
with open(f,'w') as fh: fh.write(s)
print('  [OK] AMReX_GpuElixir.cpp patched (host_task replacement)')

# Fix 4: Strip [[sycl::reqd_sub_group_size(...)]] and [[sycl::reqd_work_group_size(...)]]
# These Intel-specific attributes cause:
#   - compile warnings: 'unknown attribute reqd_sub_group_size/reqd_work_group_size ignored'
#   - runtime crash: 'LLVMToMetal: Unsupported cast involving uint4 type'
# Apple GPUs have a fixed 32-thread sub-group size so these are redundant.
for fname in [
    '${AMREX_SOURCE_DIR}/Src/Base/AMReX_GpuLaunchFunctsG.H',
    '${AMREX_SOURCE_DIR}/Src/Base/AMReX_GpuLaunchMacrosG.nolint.H',
    '${AMREX_SOURCE_DIR}/Src/Base/AMReX_TagParallelFor.H',
    '${AMREX_SOURCE_DIR}/Src/Base/AMReX_FBI.H',
]:
    with open(fname) as fh: s = fh.read()
    orig = s
    s = re.sub(r'^\s*\[\[sycl::reqd_sub_group_size\([^)]*\)\]\].*\n', '', s, flags=re.MULTILINE)
    s = re.sub(r'^\s*\[\[sycl::reqd_work_group_size\([^)]*\)\]\].*\n', '', s, flags=re.MULTILINE)
    if s != orig:
        with open(fname, 'w') as fh: fh.write(s)
        n = fname.split('/')[-1]
        print(f'  [OK] {n} patched (removed reqd_sub_group_size/reqd_work_group_size)')
    else:
        n = fname.split('/')[-1]
        print(f'  [SKIP] {n} (no matching attributes)')
"

echo ""
echo "=== Step 3: Configure AMReX with CMake ==="

BUILD_DIR="${AMREX_SOURCE_DIR}/build-acpp"
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# macOS SDK sysroot (needed for Homebrew LLVM)
MACOS_SDK="$(xcrun --sdk macosx --show-sdk-path)"

cmake .. \
    -G Ninja \
    -DCMAKE_INSTALL_PREFIX="${AMREX_INSTALL_PREFIX}" \
    -DCMAKE_CXX_COMPILER="${ACPP}" \
    -DCMAKE_OSX_SYSROOT="${MACOS_SDK}" \
    -DCMAKE_BUILD_TYPE=Release \
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
