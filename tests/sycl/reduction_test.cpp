// reduction_test.cpp — Test atomic float add (critical for WarpX deposition)
//
// WarpX current deposition uses atomic float adds to accumulate particle
// contributions to grid cells. This test validates that atomic_ref<float>
// with fetch_add works correctly on the Metal backend.
//
// Tests:
//   1. atomic_ref<float> fetch_add from many threads
//   2. atomic_ref<int> fetch_add (used for counters/reductions in AMReX)

#include <sycl/sycl.hpp>
#include <iostream>
#include <cmath>
#include <vector>

int main() {
    constexpr size_t N = 1024 * 1024;
    constexpr float TOL = 1.0f;  // Relaxed: FP32 atomics accumulate rounding error

    try {
        sycl::queue q{sycl::gpu_selector_v};

        std::string dev_name = q.get_device().get_info<sycl::info::device::name>();
        std::cout << "=== Reduction Test ===" << std::endl;
        std::cout << "Device: " << dev_name << std::endl;
        std::cout << "N = " << N << std::endl;

        // Test 1: Float atomic reduction
        std::cout << "  Test 1: atomic_ref<float> fetch_add... ";
        {
            float* sum = sycl::malloc_shared<float>(1, q);
            if (sum == nullptr) {
                std::cout << "FAIL (malloc_shared returned nullptr)" << std::endl;
                std::cout << "FAIL — USM not available" << std::endl;
                return 1;
            }
            *sum = 0.0f;

            q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
                sycl::atomic_ref<float,
                    sycl::memory_order::relaxed,
                    sycl::memory_scope::device,
                    sycl::access::address_space::global_space> atomic_sum(*sum);
                atomic_sum.fetch_add(1.0f);
            }).wait();

            float result = *sum;
            float expected = static_cast<float>(N);
            sycl::free(sum, q);

            if (std::fabs(result - expected) > TOL) {
                std::cout << "FAIL (got " << result << ", expected " << expected << ")" << std::endl;
                std::cout << "FAIL — float atomic reduction incorrect" << std::endl;
                return 1;
            }
            std::cout << "OK (sum = " << result << ")" << std::endl;
        }

        // Test 2: Integer atomic reduction
        std::cout << "  Test 2: atomic_ref<int> fetch_add... ";
        {
            int* count = sycl::malloc_shared<int>(1, q);
            if (count == nullptr) {
                std::cout << "FAIL (malloc_shared returned nullptr)" << std::endl;
                std::cout << "FAIL — USM not available" << std::endl;
                return 1;
            }
            *count = 0;

            q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
                sycl::atomic_ref<int,
                    sycl::memory_order::relaxed,
                    sycl::memory_scope::device,
                    sycl::access::address_space::global_space> atomic_count(*count);
                atomic_count.fetch_add(1);
            }).wait();

            int result = *count;
            int expected = static_cast<int>(N);
            sycl::free(count, q);

            if (result != expected) {
                std::cout << "FAIL (got " << result << ", expected " << expected << ")" << std::endl;
                std::cout << "FAIL — int atomic reduction incorrect" << std::endl;
                return 1;
            }
            std::cout << "OK (count = " << result << ")" << std::endl;
        }

        // Test 3: Multi-cell atomic accumulation (simulates deposition)
        std::cout << "  Test 3: Multi-cell atomic deposition... ";
        {
            constexpr size_t NCELLS = 256;
            constexpr size_t NPART = 1024 * 1024;

            float* grid = sycl::malloc_shared<float>(NCELLS, q);
            if (grid == nullptr) {
                std::cout << "FAIL (malloc_shared returned nullptr)" << std::endl;
                std::cout << "FAIL — USM not available" << std::endl;
                return 1;
            }
            for (size_t i = 0; i < NCELLS; ++i) grid[i] = 0.0f;

            // Each "particle" deposits 1.0 to its cell (cell = particle_id % NCELLS)
            q.parallel_for(sycl::range<1>(NPART), [=](sycl::id<1> idx) {
                size_t cell = idx[0] % NCELLS;
                sycl::atomic_ref<float,
                    sycl::memory_order::relaxed,
                    sycl::memory_scope::device,
                    sycl::access::address_space::global_space> atomic_cell(grid[cell]);
                atomic_cell.fetch_add(1.0f);
            }).wait();

            // Each cell should have NPART/NCELLS deposits
            float expected_per_cell = static_cast<float>(NPART / NCELLS);
            size_t errors = 0;
            for (size_t i = 0; i < NCELLS; ++i) {
                if (std::fabs(grid[i] - expected_per_cell) > 1.0f) {
                    if (errors < 5) {
                        std::cerr << "    Cell " << i << ": got " << grid[i]
                                  << ", expected " << expected_per_cell << std::endl;
                    }
                    ++errors;
                }
            }

            sycl::free(grid, q);

            if (errors > 0) {
                std::cout << "FAIL (" << errors << " cells incorrect)" << std::endl;
                std::cout << "FAIL — deposition simulation incorrect" << std::endl;
                return 1;
            }
            std::cout << "OK" << std::endl;
        }

        std::cout << "PASS — " << dev_name << std::endl;

    } catch (const sycl::exception& e) {
        std::cerr << "SYCL exception: " << e.what() << std::endl;
        std::cout << "FAIL — SYCL exception" << std::endl;
        return 1;
    }

    return 0;
}
