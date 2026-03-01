// device_query.cpp — Enumerate SYCL devices, confirm Metal GPU is visible
//
// Expected output: Lists all SYCL platforms/devices, prints PASS if a GPU
// device (Metal) is found.

#include <sycl/sycl.hpp>
#include <iostream>
#include <string>

int main() {
    std::cout << "=== SYCL Device Query ===" << std::endl;

    auto platforms = sycl::platform::get_platforms();
    std::cout << "Found " << platforms.size() << " platform(s)" << std::endl;

    bool found_gpu = false;

    for (const auto& platform : platforms) {
        std::string plat_name = platform.get_info<sycl::info::platform::name>();
        std::string plat_vendor = platform.get_info<sycl::info::platform::vendor>();
        std::cout << std::endl;
        std::cout << "Platform: " << plat_name << std::endl;
        std::cout << "  Vendor: " << plat_vendor << std::endl;

        auto devices = platform.get_devices();
        std::cout << "  Devices: " << devices.size() << std::endl;

        for (const auto& device : devices) {
            std::string dev_name = device.get_info<sycl::info::device::name>();
            auto dev_type = device.get_info<sycl::info::device::device_type>();
            size_t global_mem = device.get_info<sycl::info::device::global_mem_size>();
            size_t local_mem = device.get_info<sycl::info::device::local_mem_size>();
            unsigned int compute_units = device.get_info<sycl::info::device::max_compute_units>();
            size_t max_wg = device.get_info<sycl::info::device::max_work_group_size>();

            std::string type_str;
            switch (dev_type) {
                case sycl::info::device_type::cpu:  type_str = "CPU"; break;
                case sycl::info::device_type::gpu:  type_str = "GPU"; break;
                case sycl::info::device_type::accelerator: type_str = "Accelerator"; break;
                default: type_str = "Other"; break;
            }

            std::cout << "    Device: " << dev_name << std::endl;
            std::cout << "      Type:          " << type_str << std::endl;
            std::cout << "      Global memory: " << (global_mem / (1024*1024)) << " MB" << std::endl;
            std::cout << "      Local memory:  " << (local_mem / 1024) << " KB" << std::endl;
            std::cout << "      Compute units: " << compute_units << std::endl;
            std::cout << "      Max work-group: " << max_wg << std::endl;

            if (dev_type == sycl::info::device_type::gpu) {
                found_gpu = true;
            }
        }
    }

    std::cout << std::endl;
    if (found_gpu) {
        std::cout << "PASS — GPU device found" << std::endl;
    } else {
        std::cout << "FAIL — No GPU device found" << std::endl;
        return 1;
    }

    return 0;
}
