// Exercise repeated private-device to host copies on the Metal queue.

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>

int main() {
    constexpr std::size_t size = 4096;
    int iterations = 2000;
    if (const char* value = std::getenv("ACPP_D2H_STRESS_ITERATIONS")) {
        const int requested = std::atoi(value);
        if (requested > 0) {
            iterations = requested;
        }
    }

    try {
        sycl::queue queue{sycl::gpu_selector_v};
        float* device = sycl::malloc_device<float>(size, queue);
        if (!device) {
            std::cerr << "FAIL - malloc_device returned null\n";
            return 1;
        }

        float host[size];
        for (int iteration = 0; iteration < iterations; ++iteration) {
            const float expected = static_cast<float>(iteration) + 0.25f;
            queue.parallel_for(sycl::range<1>{size}, [=](sycl::id<1> index) {
                device[index] = expected;
            });
            queue.memcpy(host, device, sizeof(host)).wait();

            for (std::size_t index = 0; index < size; ++index) {
                if (std::fabs(host[index] - expected) > 1.0e-6f) {
                    std::cerr << "FAIL - iteration " << iteration
                              << ", index " << index << ": got " << host[index]
                              << ", expected " << expected << '\n';
                    sycl::free(device, queue);
                    return 1;
                }
            }
        }

        sycl::free(device, queue);
        std::cout << "PASS - " << iterations
                  << " private-device to host copies completed\n";
        return 0;
    } catch (const sycl::exception& error) {
        std::cerr << "FAIL - SYCL exception: " << error.what() << '\n';
        return 1;
    }
}
