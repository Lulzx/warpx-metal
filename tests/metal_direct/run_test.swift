// Swift driver for the Metal AS5->AS0 test
// Compiles and runs the test_thread_to_device.metal shader
import Metal
import Foundation

// Compile shader
let device = MTLCreateSystemDefaultDevice()!
print("GPU: \(device.name)")

let metalURL = URL(fileURLWithPath: CommandLine.arguments[1])
let source = try! String(contentsOf: metalURL, encoding: .utf8)
let library = try! device.makeLibrary(source: source, options: nil)

struct TestResult {
    var thread_stored: Float
    var device_read: Float
    var test_pass: UInt32
    var pad: UInt32
}

let queue = device.makeCommandQueue()!

// Prepare buffers
let resultsSize = 5 * MemoryLayout<TestResult>.size
let resultsBuffer = device.makeBuffer(length: resultsSize, options: .storageModeShared)!
let results = resultsBuffer.contents().bindMemory(to: TestResult.self, capacity: 5)

let inputData: [Float] = [42.0, 1.0, 1.5, 2.0, 10.0, 20.0, 0.001, 0.001, 99.0]
let inputBuffer = device.makeBuffer(bytes: inputData, length: inputData.count * 4, options: .storageModeShared)!

let auxData: [Float] = [0.0, 0.0, 0.0, 0.0]
let auxBuffer = device.makeBuffer(bytes: auxData, length: auxData.count * 4, options: .storageModeShared)!

func runKernel(_ name: String, buffers: [MTLBuffer]) {
    let fn = library.makeFunction(name: name)!
    let pipeline = try! device.makeComputePipelineState(function: fn)
    let cmd = queue.makeCommandBuffer()!
    let enc = cmd.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipeline)
    for (i, buf) in buffers.enumerated() {
        enc.setBuffer(buf, offset: 0, index: i)
    }
    enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    if let err = cmd.error {
        print("GPU error: \(err)")
    }
}

// Run Test 1: scalar cast
runKernel("test_thread_to_device_cast", buffers: [resultsBuffer, inputBuffer])
let r0 = results[0]
print("\n=== Test 1: Scalar thread->device cast ===")
print("  Stored (thread*): \(r0.thread_stored)")
print("  Read   (device*): \(r0.device_read)")
print("  Pass: \(r0.test_pass == 1 ? "PASS" : "FAIL")")

// Reset results for test 2
memset(resultsBuffer.contents(), 0, resultsSize)

// Run Test 2: struct cast
let inputBuffer2 = device.makeBuffer(bytes: [Float](inputData[1...3]), length: 3 * 4, options: .storageModeShared)!
runKernel("test_struct_to_device_cast", buffers: [resultsBuffer, inputBuffer2])
let r1 = results[1], r2 = results[2], r3 = results[3]
print("\n=== Test 2: Struct thread->device cast ===")
print("  Stored lo=\(r1.thread_stored) mid=\(r2.thread_stored) hi=\(r3.thread_stored)")
print("  Read   lo=\(r1.device_read)  mid=\(r2.device_read)  hi=\(r3.device_read)")
print("  Pass: \(r1.test_pass == 1 ? "PASS" : "FAIL")")

// Reset
memset(resultsBuffer.contents(), 0, resultsSize)

// Run Test 3: device ptr in thread struct
runKernel("test_device_ptr_via_cast", buffers: [resultsBuffer, inputBuffer, auxBuffer])
let r4 = results[4]
print("\n=== Test 3: Device ptr stored in thread struct, read via device cast ===")
print("  Stored cell_lo_x: \(r4.thread_stored)")
print("  Read   cell_lo_x: \(r4.device_read)")
print("  Pass: \(r4.test_pass == 1 ? "PASS" : "FAIL")")

let allPass = r0.test_pass == 1 && r1.test_pass == 1 && r4.test_pass == 1
print("\n=== Overall: \(allPass ? "ALL PASS" : "SOME FAILED") ===")
exit(allPass ? 0 : 1)
