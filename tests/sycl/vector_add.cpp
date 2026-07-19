// vector_add.cpp — Basic parallel_for with device USM
//
// Validates: kernel dispatch, device USM, and host-device data transfer

#include <sycl/sycl.hpp>

#include <cmath>
#include <iostream>
#include <vector>

int main() {
    constexpr size_t N = 1024 * 1024;  // 1M elements
    constexpr float TOL = 1e-6f;

    std::vector<float> a(N), b(N), c(N);

    // Initialize host data
    for (size_t i = 0; i < N; ++i) {
        a[i] = static_cast<float>(i) * 1.0f;
        b[i] = static_cast<float>(i) * 2.0f;
    }

    try {
        sycl::queue q{
            sycl::gpu_selector_v,
            sycl::property_list{sycl::property::queue::in_order{}}};

        std::string dev_name = q.get_device().get_info<sycl::info::device::name>();
        std::cout << "=== Vector Add ===" << std::endl;
        std::cout << "Device: " << dev_name << std::endl;
        std::cout << "N = " << N << std::endl;

        float* device_a = sycl::malloc_device<float>(N, q);
        float* device_b = sycl::malloc_device<float>(N, q);
        float* device_c = sycl::malloc_device<float>(N, q);
        auto free_device_allocations = [&]() {
            if (device_a) sycl::free(device_a, q);
            if (device_b) sycl::free(device_b, q);
            if (device_c) sycl::free(device_c, q);
        };

        if (!device_a || !device_b || !device_c) {
            free_device_allocations();
            std::cerr << "FAIL — device USM allocation failed" << std::endl;
            return 1;
        }

        try {
            q.memcpy(device_a, a.data(), N * sizeof(float));
            q.memcpy(device_b, b.data(), N * sizeof(float));
            q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
                device_c[idx] = device_a[idx] + device_b[idx];
            });
            q.memcpy(c.data(), device_c, N * sizeof(float)).wait();
        } catch (...) {
            free_device_allocations();
            throw;
        }
        free_device_allocations();

        // Verify results
        size_t errors = 0;
        for (size_t i = 0; i < N; ++i) {
            float expected = static_cast<float>(i) * 3.0f;
            if (std::fabs(c[i] - expected) > TOL) {
                if (errors < 5) {
                    std::cerr << "  Mismatch at i=" << i
                              << ": got " << c[i]
                              << ", expected " << expected << std::endl;
                }
                ++errors;
            }
        }

        if (errors == 0) {
            std::cout << "PASS — " << dev_name << std::endl;
        } else {
            std::cout << "FAIL — " << errors << " mismatches out of " << N << std::endl;
            return 1;
        }

    } catch (const sycl::exception& e) {
        std::cerr << "SYCL exception: " << e.what() << std::endl;
        std::cout << "FAIL — SYCL exception" << std::endl;
        return 1;
    }

    return 0;
}
