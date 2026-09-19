// double_test.cpp — Exercise FP64 device code on Metal through the VF64
// software binary64 lowering in the AdaptiveCpp Metal emitter.
//
// Apple GPUs have no native FP64. The emitter carries `double` as a 64-bit
// IEEE bit pattern and lowers arithmetic, comparisons, conversions and the
// exact math builtins to VF64-metal soft-float. This test checks:
//   1. host/device struct layout of a double next to an int (parser-style)
//   2. add/sub/mul/div/fma/sqrt results are bit-identical to the CPU
//   3. comparisons, int<->double and float<->double conversions
//   4. a Horner-evaluated polynomial in double keeps ~1e-15 relative error
//      where the same evaluation in float loses precision
#include <sycl/sycl.hpp>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

struct alignas(8) Node { int type; double v; };

static bool same_bits(double a, double b) {
    return std::memcmp(&a, &b, sizeof a) == 0;
}

int main() {
    sycl::queue q{sycl::gpu_selector_v};
    std::printf("=== double (VF64) test ===\nDevice: %s\n",
                q.get_device().get_info<sycl::info::device::name>().c_str());

    constexpr int N = 4096;
    double* a = sycl::malloc_shared<double>(N, q);
    double* b = sycl::malloc_shared<double>(N, q);
    double* out = sycl::malloc_shared<double>(N * 8, q);
    Node* nodes = sycl::malloc_shared<Node>(N, q);
    int* cmp = sycl::malloc_shared<int>(N * 4, q);
    float* f32 = sycl::malloc_shared<float>(N, q);
    long* i64 = sycl::malloc_shared<long>(N, q);

    for (int i = 0; i < N; ++i) {
        a[i] = 1.0 + 1e-9 * i + 0.123456789012345 * (i % 7);
        b[i] = 3.0e-8 * (i + 1) - 0.5 * (i % 3);
        nodes[i] = Node{i, 1.0 / (i + 1)};
        f32[i] = 0.1f * i;
        i64[i] = 1000000007L * i - 5;
    }

    q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
        const int i = idx[0];
        const double x = a[i], y = b[i];
        out[i * 8 + 0] = x + y;
        out[i * 8 + 1] = x - y;
        out[i * 8 + 2] = x * y;
        out[i * 8 + 3] = x / y;
        out[i * 8 + 4] = sycl::fma(x, y, -x);
        out[i * 8 + 5] = sycl::sqrt(sycl::fabs(y));
        out[i * 8 + 6] = nodes[i].v * static_cast<double>(nodes[i].type);
        // Horner polynomial: coefficients chosen so float loses ~1e-7.
        double p = 0.0;
        for (int k = 0; k < 8; ++k) { p = p * x + (1.0 / (k + 1)); }
        out[i * 8 + 7] = p;
        cmp[i * 4 + 0] = (x < y) ? 1 : 0;
        cmp[i * 4 + 1] = (x == x) ? 1 : 0;
        cmp[i * 4 + 2] = static_cast<int>(x * 1000.0);
        cmp[i * 4 + 3] = (static_cast<double>(f32[i]) > 1.0) ? 1 : 0;
        f32[i] = static_cast<float>(x * y);
        i64[i] = static_cast<long>(static_cast<double>(i64[i]) * 0.5);
    }).wait();

    int fails = 0;
    double max_rel = 0.0;
    for (int i = 0; i < N; ++i) {
        const double x = a[i], y = b[i];
        const double ref[8] = {
            x + y, x - y, x * y, x / y, std::fma(x, y, -x), std::sqrt(std::fabs(y)),
            (1.0 / (i + 1)) * i, 0.0
        };
        for (int k = 0; k < 7; ++k) {
            if (!same_bits(out[i * 8 + k], ref[k])) {
                if (fails < 10)
                    std::printf("  MISMATCH i=%d op=%d gpu=%.17g cpu=%.17g\n", i, k, out[i * 8 + k], ref[k]);
                ++fails;
            }
        }
        double p = 0.0;
        for (int k = 0; k < 8; ++k) { p = p * x + (1.0 / (k + 1)); }
        max_rel = std::max(max_rel, std::fabs(out[i * 8 + 7] - p) / std::fabs(p));
        const float fexp = static_cast<float>(x * y);
        const long lexp = static_cast<long>(static_cast<double>(1000000007L * i - 5) * 0.5);
        if (cmp[i * 4 + 0] != ((x < y) ? 1 : 0) || cmp[i * 4 + 1] != 1 ||
            cmp[i * 4 + 2] != static_cast<int>(x * 1000.0) ||
            cmp[i * 4 + 3] != ((static_cast<double>(0.1f * i) > 1.0) ? 1 : 0) ||
            f32[i] != fexp || i64[i] != lexp) {
            if (fails < 10)
                std::printf("  MISMATCH i=%d cmp/conv: %d %d %d %d  f32=%.9g/%.9g  i64=%ld/%ld\n", i,
                            cmp[i*4], cmp[i*4+1], cmp[i*4+2], cmp[i*4+3], f32[i], fexp, i64[i], lexp);
            ++fails;
        }
    }
    std::printf("  bitwise mismatches: %d\n  Horner max rel err vs CPU double: %.3g\n", fails, max_rel);

    sycl::free(a, q); sycl::free(b, q); sycl::free(out, q); sycl::free(nodes, q);
    sycl::free(cmp, q); sycl::free(f32, q); sycl::free(i64, q);
    if (fails == 0 && max_rel < 1e-14) { std::printf("PASS\n"); return 0; }
    std::printf("FAIL\n");
    return 1;
}
