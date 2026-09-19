// double_bench.cpp — Throughput of VF64 software binary64 vs native float on
// the Metal GPU. Compute-bound dependent chain: x = x * a + b, 32 steps.
// Reports ns per element and the FP64/FP32 slowdown factor.
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>

template <typename T>
static double bench(sycl::queue& q, T* x, size_t n, int reps) {
    const T a = T(1.0000001), b = T(-0.0000001);
    auto run = [&] {
        q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
            T v = x[i];
            for (int k = 0; k < 32; ++k) { v = v * a + b; }
            x[i] = v;
        }).wait();
    };
    run(); // JIT + warm-up
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) run();
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::nano>(t1 - t0).count() / (double(reps) * n);
}

int main() {
    sycl::queue q{sycl::gpu_selector_v};
    constexpr size_t N = 1 << 22;
    float* xf = sycl::malloc_shared<float>(N, q);
    double* xd = sycl::malloc_shared<double>(N, q);
    for (size_t i = 0; i < N; ++i) { xf[i] = 1.0f; xd[i] = 1.0; }
    const double nf = bench(q, xf, N, 5);
    const double nd = bench(q, xd, N, 5);
    std::printf("=== FP64 (VF64) vs FP32 throughput, %zu elements x 32 mul-add ===\n", N);
    std::printf("  float : %8.3f ns/element  (%6.1f GFLOP/s)\n", nf, 64.0 / nf);
    std::printf("  double: %8.3f ns/element  (%6.1f GFLOP/s)\n", nd, 64.0 / nd);
    std::printf("  slowdown: %.1fx\n", nd / nf);
    sycl::free(xf, q); sycl::free(xd, q);
    return 0;
}
