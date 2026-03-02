// test_as5_noopt.cpp
// Tests whether reading thread-local memory via device pointer works.
// Compiled with -O0 to prevent LLVM from optimizing away the addrspacecast.
// This reproduces the exact WarpX AddPlasma pattern:
//   *((thread float*)t33) = val;
//   t46 = (device void*)((ulong)t33);
//   result = *((device float*)t46);

#include <sycl/sycl.hpp>
#include <cstdio>

// Prevent optimization on this function: returns addr of thread-local var as uintptr
[[clang::optnone]]
static uintptr_t get_thread_addr(float* p, float val) {
    // Store val to the thread-local location *p
    *p = val;
    // Return address as integer (AS5→integer, then use as device pointer)
    return (uintptr_t)p;
}

int main() {
    sycl::queue q;
    float* result = sycl::malloc_shared<float>(3, q);
    result[0] = -999.0f;
    result[1] = -999.0f;
    result[2] = -999.0f;

    q.single_task([=]() {
        float x_lo = 1.0f;
        float x_mid = 1.5f;
        float x_hi = 2.0f;

        // optnone function: stores to thread-local memory, returns addr as uintptr
        uintptr_t addr_lo  = get_thread_addr(&x_lo,  x_lo);
        uintptr_t addr_mid = get_thread_addr(&x_mid, x_mid);
        uintptr_t addr_hi  = get_thread_addr(&x_hi,  x_hi);

        // Dereference via device pointer (the WarpX AddPlasma pattern)
        float* dev_lo  = (float*)(void*)addr_lo;
        float* dev_mid = (float*)(void*)addr_mid;
        float* dev_hi  = (float*)(void*)addr_hi;

        result[0] = *dev_lo;
        result[1] = *dev_mid;
        result[2] = *dev_hi;
    });
    q.wait();

    printf("Expected: x_lo=1.0 x_mid=1.5 x_hi=2.0\n");
    printf("Got:      x_lo=%.1f x_mid=%.1f x_hi=%.1f\n",
           result[0], result[1], result[2]);

    bool pass = (result[0] == 1.0f && result[1] == 1.5f && result[2] == 2.0f);
    printf("%s\n", pass
        ? "PASS: thread-local memory IS accessible via device pointer"
        : "FAIL: thread-local memory NOT accessible via device pointer");

    sycl::free(result, q);
    return pass ? 0 : 1;
}
