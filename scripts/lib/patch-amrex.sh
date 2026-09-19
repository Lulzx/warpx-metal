#!/usr/bin/env bash
# patch-amrex.sh — Bring the AMReX checkout to the exact source state the
# Metal port builds from.
#
# Used by 03-build-amrex.sh (AMReX library), 05-build-warpx.sh (WarpX GPU) and
# 07-build-warpx-cpu.sh (WarpX CPU baseline) so that all three builds compile
# the identical AMReX tree. The SYCL-only edits are guarded by SYCL/AdaptiveCpp
# macros and are no-ops for the CPU build.
#
# Every edit is verified: a replacement that does not change the file is a hard
# failure, never a silent "[OK]".
#
# Usage: source scripts/env.sh; scripts/lib/patch-amrex.sh

set -euo pipefail

: "${AMREX_SOURCE_DIR:?source scripts/env.sh first}"
: "${PATCHES_DIR:?source scripts/env.sh first}"

PATCH_DIR="${PATCHES_DIR}/amrex"
POST_PATCH_DIR="${PATCHES_DIR}/amrex-post"

cd "${AMREX_SOURCE_DIR}"
git checkout -- .
git clean -fd

echo "  [..] Replacing AdaptiveCpp-incompatible files"
replace_file() {
    local src="${PATCH_DIR}/$1"
    local dst="${AMREX_SOURCE_DIR}/$2"
    if [ ! -f "${src}" ]; then
        echo "  [FAIL] replacement file missing: ${src}" >&2
        exit 1
    fi
    cp "${src}" "${dst}"
    echo "  [OK] $2 replaced"
}
replace_file "AMReXSYCL.cmake"       "Tools/CMake/AMReXSYCL.cmake"
replace_file "AMReX_RandomEngine.H"  "Src/Base/AMReX_RandomEngine.H"
replace_file "AMReX_Random.cpp"      "Src/Base/AMReX_Random.cpp"

echo "  [..] Applying source-level edits"
AMREX_SOURCE_DIR="${AMREX_SOURCE_DIR}" python3 - <<'PY'
import os, re, sys

root = os.environ["AMREX_SOURCE_DIR"]

def edit(rel, old, new, what):
    path = os.path.join(root, rel)
    with open(path) as fh:
        s = fh.read()
    n = s.count(old)
    if n != 1:
        sys.exit(f"  [FAIL] {rel}: expected exactly one match for {what}, found {n}. "
                 "Upstream source drifted; refresh the edit before building.")
    with open(path, "w") as fh:
        fh.write(s.replace(old, new))
    print(f"  [OK] {rel}: {what}")

# 1. AMReX_INT.H — disable __int128 when AMREX_NO_INT128 is defined. The Metal
#    emitter maps i128 to uint4 and supports only a few casts; every i128 code
#    path (umulhi, FastDivmodU64) has a non-128 fallback.
edit("Src/Base/AMReX_INT.H",
     "#if (defined(__x86_64) || defined (__aarch64__)) && !defined(_WIN32) && (defined(__GNUC__) || defined(__clang__)) && !defined(__NVCOMPILER)",
     "#if (defined(__x86_64) || defined (__aarch64__)) && !defined(_WIN32) && (defined(__GNUC__) || defined(__clang__)) && !defined(__NVCOMPILER) && !defined(AMREX_NO_INT128)",
     "guard __int128 behind AMREX_NO_INT128")

# 2./3. host_task is not implemented by AdaptiveCpp. Defer the frees through
#    Gpu::Device::freeAsync (stream-ordered) instead of blocking on q.wait().
edit("Src/Base/AMReX_GpuAsyncArray.H",
     '''                    q.submit([&] (sycl::handler& h) {
                        h.host_task([=] () {
                            The_Arena()->free(pd);
                            The_Pinned_Arena()->free(ph);
                        });
                    });
                } catch (sycl::exception const& ex) {
                    amrex::Abort(std::string("host_task: ")+ex.what()+"!!!!!");''',
     '''                    (void)q;
                    Gpu::Device::freeAsync(The_Arena(), pd);
                    Gpu::Device::freeAsync(The_Pinned_Arena(), ph);
                } catch (sycl::exception const& ex) {
                    amrex::Abort(std::string("async cleanup: ")+ex.what()+"!!!!!");''',
     "replace host_task with freeAsync")

edit("Src/Base/AMReX_GpuElixir.cpp",
     '''        auto& q = *(Gpu::gpuStream().queue);
        try {
            q.submit([&] (sycl::handler& h) {
                h.host_task([=] () {
                    for (auto const& pa : lpa) {
                        pa.second->free(pa.first);
                    }
                });
            });
        } catch (sycl::exception const& ex) {
            amrex::Abort(std::string("host_task: ")+ex.what()+"!!!!!");''',
     '''        try {
            for (auto const& pa : lpa) {
                Gpu::Device::freeAsync(pa.second, pa.first);
            }
        } catch (sycl::exception const& ex) {
            amrex::Abort(std::string("async cleanup: ")+ex.what()+"!!!!!");''',
     "replace host_task with freeAsync")

# 4. Strip [[sycl::reqd_sub_group_size]] / [[sycl::reqd_work_group_size]].
#    They warn as unknown attributes and crash the Metal emitter ("Unsupported
#    cast involving uint4"). Apple GPUs have a fixed 32-wide SIMD group anyway.
attr = re.compile(r'^\s*\[\[sycl::reqd_(sub_group|work_group)_size\([^)]*\)\]\].*\n', re.MULTILINE)
stripped = 0
for rel in ["Src/Base/AMReX_GpuLaunchFunctsG.H",
            "Src/Base/AMReX_GpuLaunchMacrosG.nolint.H",
            "Src/Base/AMReX_TagParallelFor.H",
            "Src/Base/AMReX_FBI.H"]:
    path = os.path.join(root, rel)
    with open(path) as fh:
        s = fh.read()
    s2, n = attr.subn("", s)
    if n:
        with open(path, "w") as fh:
            fh.write(s2)
        print(f"  [OK] {rel}: removed {n} reqd_*_size attribute(s)")
    stripped += n
if stripped == 0:
    sys.exit("  [FAIL] no reqd_sub_group_size/reqd_work_group_size attributes found; "
             "upstream source drifted")
PY

echo "  [..] Applying post-replacement patches"
shopt -s nullglob
post_patches=("${POST_PATCH_DIR}"/*.patch)
shopt -u nullglob
if [ "${#post_patches[@]}" -eq 0 ]; then
    echo "  [FAIL] no patches found in ${POST_PATCH_DIR}" >&2
    exit 1
fi
for patch in "${post_patches[@]}"; do
    name="$(basename "${patch}")"
    if git apply --check "${patch}" 2>/dev/null; then
        git apply "${patch}"
        echo "  [OK] applied ${name}"
    elif git apply --reverse --check "${patch}" 2>/dev/null; then
        echo "  [OK] ${name} already applied"
    else
        echo "  [FAIL] ${name} does not apply cleanly" >&2
        exit 1
    fi
done

echo "  [OK] AMReX source state ready"
