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

if [ -d "${WARPX_SOURCE_DIR}/.git" ]; then
    echo "  [OK] WarpX already cloned at ${WARPX_SOURCE_DIR}"
    cd "${WARPX_SOURCE_DIR}"
    git checkout -- .
    git clean -fd
else
    echo "  [..] Cloning WarpX..."
    git clone https://github.com/ECP-WarpX/WarpX.git "${WARPX_SOURCE_DIR}"
fi

echo ""
echo "=== Step 2: Apply AMReX patches ==="
echo "  (WarpX builds AMReX from source — patches must be applied to AMReX tree)"

cd "${AMREX_SOURCE_DIR}"
git checkout -- .
git clean -fd

AMREX_PATCH_DIR="${PATCHES_DIR}/amrex"

# Apply .patch files via git apply (same as 03-build-amrex.sh)
AMREX_PATCH_COUNT=$(find "${AMREX_PATCH_DIR}" -name '*.patch' 2>/dev/null | wc -l | tr -d ' ')
if [ "${AMREX_PATCH_COUNT}" -gt 0 ]; then
    for patch in "${AMREX_PATCH_DIR}"/*.patch; do
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

# Copy replacement files
replace_file() {
    local src="${AMREX_PATCH_DIR}/$1"
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
                    amrex::Abort(std::string(\\\"host_task: \\\")+ex.what()+\\\"!!!!!\\\");'''
new = '''                    (void)q;
                    Gpu::Device::freeAsync(The_Arena(), pd);
                    Gpu::Device::freeAsync(The_Pinned_Arena(), ph);
                } catch (sycl::exception const& ex) {
                    amrex::Abort(std::string(\\\"async cleanup: \\\")+ex.what()+\\\"!!!!!\\\");'''
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
            amrex::Abort(std::string(\\\"host_task: \\\")+ex.what()+\\\"!!!!!\\\");'''
new = '''        try {
            for (auto const& pa : lpa) {
                Gpu::Device::freeAsync(pa.second, pa.first);
            }
        } catch (sycl::exception const& ex) {
            amrex::Abort(std::string(\\\"async cleanup: \\\")+ex.what()+\\\"!!!!!\\\");'''
s = s.replace(old, new)
with open(f,'w') as fh: fh.write(s)
print('  [OK] AMReX_GpuElixir.cpp patched (host_task replacement)')

# Fix 4: Strip [[sycl::reqd_sub_group_size(...)]] and [[sycl::reqd_work_group_size(...)]]
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
            else
                echo "  [SKIP] ${PATCH_NAME} (already applied or conflicts)"
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

# Use our Metal-patched AMReX source tree (WarpX builds it as a subproject).
# WarpX will configure AMReX with the right components (2D/3D, PIC, EB, etc.).
cmake -S "${WARPX_SOURCE_DIR}" -B . \
    -G Ninja \
    -DCMAKE_CXX_COMPILER="${ACPP}" \
    -DCMAKE_OSX_SYSROOT="${MACOS_SDK}" \
    -DCMAKE_BUILD_TYPE=Release \
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
