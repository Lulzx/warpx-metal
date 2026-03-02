// test_as5_escape.cpp
// Definitively tests whether AS5 (thread) pointers can be accessed via AS0/device
// after the addrspacecast AS5->AS0 pattern.
//
// Strategy: use [[clang::noinline]] on the receiving function so LLVM cannot
// eliminate the addrspacecast. This reproduces the exact WarpX AddPlasma pattern
// where nested lambda captures (thread-local) are passed as device void*.

#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>

// Struct mimicking AddPlasma lambda captures (position bounds etc.)
struct Captures {
    float x_lo, x_hi;
    float y_lo, y_hi;
    int   count;
};

// NOINLINE: forces LLVM to emit actual addrspacecast in callee + caller boundary.
// Receives a generic (AS0) pointer to captures and a device output buffer.
[[clang::noinline]]
static void process_captures(void* caps_ptr, float* result) {
    Captures* c = (Captures*)caps_ptr;
    result[0] = c->x_lo;
    result[1] = c->x_hi;
    result[2] = c->y_lo;
    result[3] = c->y_hi;
    result[4] = (float)c->count;
}

int main() {
    sycl::queue q;
    float* result = sycl::malloc_shared<float>(5, q);
    for (int i = 0; i < 5; i++) result[i] = -999.0f;

    q.single_task([=]() {
        // Thread-local struct — this is the AS5 alloca pattern
        Captures caps;
        caps.x_lo  = 1.0f;
        caps.x_hi  = 2.0f;
        caps.y_lo  = 3.0f;
        caps.y_hi  = 4.0f;
        caps.count = 42;

        // Take address of thread-local struct → addrspacecast AS5→AS0
        // The noinline call forces LLVM to keep this cast.
        process_captures(&caps, result);
    });
    q.wait();

    printf("Expected: x_lo=1 x_hi=2 y_lo=3 y_hi=4 count=42\n");
    printf("Got:      x_lo=%.0f x_hi=%.0f y_lo=%.0f y_hi=%.0f count=%.0f\n",
           result[0], result[1], result[2], result[3], result[4]);

    bool pass = (result[0] == 1.0f && result[1] == 2.0f &&
                 result[2] == 3.0f && result[3] == 4.0f && result[4] == 42.0f);
    printf("%s\n", pass ? "PASS" : "FAIL - thread-local memory NOT accessible via device pointer");

    sycl::free(result, q);
    return pass ? 0 : 1;
}
