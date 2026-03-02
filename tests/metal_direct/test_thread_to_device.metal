// Direct Metal Shading Language test for AS5->AS0 (thread -> device pointer) pattern.
// Tests whether reading thread-private memory via a device pointer works on Apple Silicon.
// This pattern appears in the WarpX AddPlasma kernel.

#include <metal_stdlib>
using namespace metal;

struct TestResult {
    float thread_stored;   // value stored via thread pointer
    float device_read;     // value read back via device pointer cast
    uint  test_pass;       // 1 if device_read == thread_stored
    uint  pad;
};

kernel void test_thread_to_device_cast(
    device TestResult* results [[buffer(0)]],
    device const float* input  [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid > 0) { return; }  // only first thread

    // --- Test 1: Simple scalar ---
    thread float local_scalar;
    local_scalar = input[0];  // = 42.0f (from host)

    // Cast thread pointer to device pointer via ulong (the AS5->AS0 pattern)
    device float* dev_ptr = (device float*)((ulong)(&local_scalar));

    // Read back via device pointer
    float via_device = *dev_ptr;

    results[0].thread_stored = local_scalar;
    results[0].device_read   = via_device;
    results[0].test_pass     = (local_scalar == via_device) ? 1u : 0u;
}

kernel void test_struct_to_device_cast(
    device TestResult* results [[buffer(0)]],
    device const float* input  [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid > 0) { return; }

    // --- Test 2: Struct (like amrex::GpuArray) ---
    struct Bounds {
        float lo;
        float mid;
        float hi;
    };

    thread Bounds b;
    b.lo  = input[0];   // 1.0f
    b.mid = input[1];   // 1.5f
    b.hi  = input[2];   // 2.0f

    // Cast struct thread pointer to device pointer (the exact WarpX pattern)
    device void* dev_vptr = (device void*)((ulong)(&b));

    // Read the struct fields via device pointer
    device float* df = (device float*)dev_vptr;
    float lo_via_dev  = df[0];
    float mid_via_dev = df[1];
    float hi_via_dev  = df[2];

    results[1].thread_stored = b.lo;
    results[1].device_read   = lo_via_dev;
    results[1].test_pass     = ((b.lo == lo_via_dev) && (b.mid == mid_via_dev) && (b.hi == hi_via_dev)) ? 1u : 0u;

    // Also write mid/hi for inspection
    results[2].thread_stored = b.mid;
    results[2].device_read   = mid_via_dev;
    results[2].test_pass     = 0xdeadbeefu;
    results[3].thread_stored = b.hi;
    results[3].device_read   = hi_via_dev;
    results[3].test_pass     = 0xdeadbeefu;
}

kernel void test_device_ptr_via_cast(
    device TestResult* results [[buffer(0)]],
    device const float* input  [[buffer(1)]],
    device float*       aux    [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid > 0) { return; }

    // --- Test 3: Device pointer stored in thread memory, read back via device cast ---
    // This is the WarpX $2clE capture pattern: store device ptr in thread struct,
    // then access the struct via device pointer cast.
    thread struct {
        uint  cell_lo_x;   // offset 0
        uint  cell_lo_z;   // offset 4
        float dx;          // offset 8
        float dz;          // offset 12
        device float* dev_arr;  // offset 16 (8 bytes)
    } captures;

    captures.cell_lo_x = (uint)input[0];  // 10
    captures.cell_lo_z = (uint)input[1];  // 20
    captures.dx        = input[2];        // 0.001
    captures.dz        = input[3];        // 0.001
    captures.dev_arr   = aux;             // pointer to aux buffer

    // Write known value to aux
    aux[0] = input[4];  // 99.0f

    // Cast captures struct thread ptr to device ptr
    device void* dev_caps = (device void*)((ulong)(&captures));
    device uint*  u_caps  = (device uint*)dev_caps;
    device float* f_caps  = (device float*)dev_caps;

    // Read fields via device pointer
    uint  cell_lo_x_via_dev = u_caps[0];           // offset 0
    uint  cell_lo_z_via_dev = u_caps[1];           // offset 4
    float dx_via_dev        = f_caps[2];           // offset 8
    // Read the device pointer stored at offset 16
    device float** ptr_field = (device float**)((ulong)((device uchar*)dev_caps + 16));
    device float*  dev_arr_via_dev = *ptr_field;

    // Use the device pointer read from thread struct
    float via_dev_arr = dev_arr_via_dev[0];

    bool all_ok = (cell_lo_x_via_dev == 10u) &&
                  (cell_lo_z_via_dev == 20u) &&
                  (dx_via_dev == 0.001f) &&
                  (dev_arr_via_dev == aux) &&
                  (via_dev_arr == input[4]);

    results[4].thread_stored = (float)captures.cell_lo_x;
    results[4].device_read   = (float)cell_lo_x_via_dev;
    results[4].test_pass     = all_ok ? 1u : 0u;
}
