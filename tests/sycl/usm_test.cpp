// usm_test.cpp — Test sycl::malloc_shared (Unified Shared Memory)
//
// CRITICAL for AMReX: AMReX's SYCL backend relies on USM (malloc_shared,
// malloc_device) for memory management. If this test fails, AMReX will
// need significant memory management workarounds.
//
// Tests:
//   1. sycl::malloc_shared allocation and host/device access
//   2. Kernel writing to shared memory, host reading result
//   3. sycl::free deallocation

#include <sycl/sycl.hpp>
#include <iostream>
#include <cmath>

int main() {
    constexpr size_t N = 1024 * 1024;
    constexpr float TOL = 1e-6f;

    try {
        sycl::queue q{sycl::gpu_selector_v};

        std::string dev_name = q.get_device().get_info<sycl::info::device::name>();
        std::cout << "=== USM Test ===" << std::endl;
        std::cout << "Device: " << dev_name << std::endl;
        std::cout << "N = " << N << std::endl;

        // Test 1: malloc_shared allocation
        std::cout << "  Test 1: sycl::malloc_shared... ";
        float* data = sycl::malloc_shared<float>(N, q);
        if (data == nullptr) {
            std::cout << "FAIL (returned nullptr)" << std::endl;
            std::cout << "FAIL — malloc_shared not supported" << std::endl;
            return 1;
        }
        std::cout << "OK" << std::endl;

        // Test 2: Host writes, device reads and writes
        std::cout << "  Test 2: Host init, device compute... ";
        for (size_t i = 0; i < N; ++i) {
            data[i] = static_cast<float>(i);
        }

        q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
            data[idx] = data[idx] * 2.0f + 1.0f;
        }).wait();

        // Verify on host
        size_t errors = 0;
        for (size_t i = 0; i < N; ++i) {
            float expected = static_cast<float>(i) * 2.0f + 1.0f;
            if (std::fabs(data[i] - expected) > TOL) {
                if (errors < 5) {
                    std::cerr << "    Mismatch at i=" << i
                              << ": got " << data[i]
                              << ", expected " << expected << std::endl;
                }
                ++errors;
            }
        }

        if (errors > 0) {
            std::cout << "FAIL (" << errors << " mismatches)" << std::endl;
            sycl::free(data, q);
            std::cout << "FAIL — USM read/write incorrect" << std::endl;
            return 1;
        }
        std::cout << "OK" << std::endl;

        // Test 3: Multiple allocations (simulates AMReX arena behavior)
        std::cout << "  Test 3: Multiple USM allocations... ";
        float* a = sycl::malloc_shared<float>(N, q);
        float* b = sycl::malloc_shared<float>(N, q);
        float* c = sycl::malloc_shared<float>(N, q);

        if (a == nullptr || b == nullptr || c == nullptr) {
            std::cout << "FAIL (allocation returned nullptr)" << std::endl;
            if (a) sycl::free(a, q);
            if (b) sycl::free(b, q);
            if (c) sycl::free(c, q);
            sycl::free(data, q);
            std::cout << "FAIL — multiple malloc_shared failed" << std::endl;
            return 1;
        }

        // Initialize a and b on host
        for (size_t i = 0; i < N; ++i) {
            a[i] = static_cast<float>(i);
            b[i] = static_cast<float>(N - i);
        }

        // Compute c = a + b on device
        q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
            c[idx] = a[idx] + b[idx];
        }).wait();

        // Verify: c[i] should be N for all i
        errors = 0;
        for (size_t i = 0; i < N; ++i) {
            float expected = static_cast<float>(N);
            if (std::fabs(c[i] - expected) > TOL) {
                ++errors;
            }
        }

        sycl::free(a, q);
        sycl::free(b, q);
        sycl::free(c, q);
        sycl::free(data, q);

        if (errors > 0) {
            std::cout << "FAIL (" << errors << " mismatches)" << std::endl;
            std::cout << "FAIL — multi-buffer USM incorrect" << std::endl;
            return 1;
        }
        std::cout << "OK" << std::endl;

        std::cout << "PASS — " << dev_name << std::endl;

    } catch (const sycl::exception& e) {
        std::cerr << "SYCL exception: " << e.what() << std::endl;
        std::cout << "FAIL — SYCL exception" << std::endl;
        return 1;
    }

    return 0;
}
