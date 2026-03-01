// vector_add.cpp — Basic parallel_for with buffer/accessor model (no USM)
//
// Validates: kernel dispatch, buffer creation, host-device data transfer
// Uses buffers deliberately — Metal backend may not fully support USM.

#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>

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
        // Select GPU device, fall back to default
        sycl::queue q{sycl::gpu_selector_v};

        std::string dev_name = q.get_device().get_info<sycl::info::device::name>();
        std::cout << "=== Vector Add ===" << std::endl;
        std::cout << "Device: " << dev_name << std::endl;
        std::cout << "N = " << N << std::endl;

        {
            sycl::buffer<float> buf_a(a.data(), sycl::range<1>(N));
            sycl::buffer<float> buf_b(b.data(), sycl::range<1>(N));
            sycl::buffer<float> buf_c(c.data(), sycl::range<1>(N));

            q.submit([&](sycl::handler& h) {
                auto acc_a = buf_a.get_access<sycl::access::mode::read>(h);
                auto acc_b = buf_b.get_access<sycl::access::mode::read>(h);
                auto acc_c = buf_c.get_access<sycl::access::mode::write>(h);

                h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
                    acc_c[idx] = acc_a[idx] + acc_b[idx];
                });
            });

            q.wait();
        }
        // Buffers go out of scope — data copied back to host vectors

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
