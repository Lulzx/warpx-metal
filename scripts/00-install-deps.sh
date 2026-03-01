#!/usr/bin/env bash
# 00-install-deps.sh — Install build dependencies for AdaptiveCpp Metal backend
#
# Installs: LLVM 18, Boost, Ninja, CMake via Homebrew
# Downloads: Apple metal-cpp headers
#
# Usage: ./scripts/00-install-deps.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

echo ""
echo "=== Step 1: Install Homebrew packages ==="

PACKAGES=(llvm@20 llvm@18 boost ninja cmake libomp)
for pkg in "${PACKAGES[@]}"; do
    if brew list --formula "$pkg" &>/dev/null; then
        echo "  [OK] $pkg already installed"
    else
        echo "  [..] Installing $pkg..."
        brew install "$pkg"
    fi
done

echo ""
echo "=== Step 2: Verify LLVM 20 ==="

CLANG_VERSION=$("${LLVM_PREFIX}/bin/clang" --version | head -1)
echo "  ${CLANG_VERSION}"
if echo "${CLANG_VERSION}" | grep -q "20\."; then
    echo "  [OK] LLVM 20 confirmed"
else
    echo "  [WARN] Expected LLVM 20, got: ${CLANG_VERSION}"
fi

# LLVM 18 provides ld64.lld (not included in LLVM 20 Homebrew bottle)
LLD_PATH="$(brew --prefix llvm@18)/bin/ld64.lld"
if [ -x "${LLD_PATH}" ]; then
    echo "  [OK] ld64.lld: ${LLD_PATH}"
else
    echo "  [WARN] ld64.lld not found in llvm@18"
fi

echo ""
echo "=== Step 3: Verify Metal compiler toolchain ==="

# Check for the Metal compiler binary (may need Metal Toolchain component)
METAL_PATH="$(xcrun -f metal 2>/dev/null || true)"
if [ -n "${METAL_PATH}" ]; then
    # Verify it actually works (Xcode 26+ may require downloading Metal Toolchain)
    if "${METAL_PATH}" --version &>/dev/null; then
        echo "  [OK] Metal compiler: ${METAL_PATH}"
    else
        echo "  [WARN] Metal compiler found at ${METAL_PATH} but not functional."
        echo "         Run: xcodebuild -downloadComponent MetalToolchain"
        echo "         (AdaptiveCpp uses LLVM JIT, so this may not be blocking.)"
    fi
else
    echo "  [WARN] Metal compiler not found via xcrun."
    echo "         Install Xcode command-line tools: xcode-select --install"
    echo "         Then download Metal Toolchain: xcodebuild -downloadComponent MetalToolchain"
fi

METALLIB_PATH="$(xcrun -f metallib 2>/dev/null || true)"
if [ -n "${METALLIB_PATH}" ]; then
    echo "  [OK] metallib: ${METALLIB_PATH}"
else
    echo "  [WARN] metallib not found. This is expected if Metal Toolchain is not installed."
    echo "         Run: xcodebuild -downloadComponent MetalToolchain"
    echo "         (Not blocking — AdaptiveCpp uses its own LLVM IR → MSL JIT pipeline.)"
fi

echo ""
echo "=== Step 4: Download metal-cpp headers ==="

if [ -f "${METAL_CPP_DIR}/Metal/Metal.hpp" ]; then
    echo "  [OK] metal-cpp already present at ${METAL_CPP_DIR}"
else
    echo "  [..] Downloading metal-cpp from Apple..."
    METAL_CPP_URL="https://developer.apple.com/metal/cpp/files/metal-cpp_macOS15_iOS18.zip"
    TMPDIR_DL="$(mktemp -d)"
    curl -sL "${METAL_CPP_URL}" -o "${TMPDIR_DL}/metal-cpp.zip"
    unzip -qo "${TMPDIR_DL}/metal-cpp.zip" -d "${TMPDIR_DL}/extracted"

    # Find the extracted directory (name varies by release)
    EXTRACTED_DIR="$(find "${TMPDIR_DL}/extracted" -maxdepth 1 -type d -name 'metal-cpp*' | head -1)"
    if [ -z "${EXTRACTED_DIR}" ]; then
        # Fallback: the headers might be directly in extracted/
        EXTRACTED_DIR="${TMPDIR_DL}/extracted"
    fi

    mkdir -p "${METAL_CPP_DIR}"
    cp -R "${EXTRACTED_DIR}/"* "${METAL_CPP_DIR}/"
    rm -rf "${TMPDIR_DL}"

    if [ -f "${METAL_CPP_DIR}/Metal/Metal.hpp" ]; then
        echo "  [OK] metal-cpp installed to ${METAL_CPP_DIR}"
    else
        echo "  [WARN] metal-cpp downloaded but Metal/Metal.hpp not found at expected path."
        echo "         Contents of ${METAL_CPP_DIR}:"
        ls -la "${METAL_CPP_DIR}/"
        echo "         You may need to adjust METAL_CPP_DIR in env.sh"
    fi
fi

echo ""
echo "=== Step 5: Verify Boost ==="

BOOST_PREFIX="$(brew --prefix boost 2>/dev/null || true)"
if [ -d "${BOOST_PREFIX}/include/boost" ]; then
    echo "  [OK] Boost headers: ${BOOST_PREFIX}/include/boost"
else
    echo "  [FAIL] Boost headers not found"
    exit 1
fi

echo ""
echo "=== All dependencies installed ==="
echo ""
echo "Next step: ./scripts/01-build-adaptivecpp.sh"
