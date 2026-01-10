/**
 * Demo of the torch namespace API
 *
 * This demonstrates the PyTorch-like interface for tensor operations.
 * Note: This won't run without the native library, but shows the TypeScript types work.
 */

import { torch, Device, float32, float64 } from '../index.js'
import type { Tensor } from '../tensor/tensor.js'

// ==================== Basic Tensor Creation ====================

// Create tensors with type inference
const x = torch.zeros([2, 3] as const) // Tensor<[2, 3], DType<"float32">>
const y = torch.ones([2, 3] as const) // Tensor<[2, 3], DType<"float32">>
const z = torch.randn([100, 50] as const) // Random normal

// Specify dtype
const f64Tensor = torch.zeros([10, 20] as const, float64)
const f32Tensor = torch.ones([5, 5] as const, float32)

// Create from data
const dataMatrix = torch.tensor([1, 2, 3, 4, 5, 6], [2, 3] as const, float32)

// Auto-infer shape from nested arrays
const nested = torch.from(
  [
    [1, 2],
    [3, 4],
  ],
  float32,
)

// Arange
const range = torch.arange(0, 10) // [0, 1, 2, ..., 9]
const floatRange = torch.arange(0, 1, 0.1) // [0.0, 0.1, ..., 0.9]

// ==================== Memory Scopes ====================

// Automatic cleanup with scopes
const result = torch.run(() => {
  const a = torch.randn([100, 100] as const)
  const b = torch.randn([100, 100] as const)
  const c = a.matmul(b)
  return c.escape() // Keep c alive after scope
})
// a and b are freed here, c persists

// Async scopes
async function asyncExample() {
  const result = await torch.runAsync(async () => {
    const x = torch.zeros([10, 10] as const)
    const y = torch.ones([10, 10] as const)
    const sum = x.add(y)
    return sum.escape()
  })
  return result
}

// ==================== Tensor Operations ====================

function tensorOperationsExample() {
  const a = torch.zeros([2, 3] as const)
  const b = torch.ones([2, 3] as const)

  // Element-wise ops
  const sum = a.add(b)
  const diff = a.sub(b)
  const prod = a.mul(b)
  const quot = a.div(b)

  // Matrix multiplication with type inference
  const m1 = torch.randn([10, 20] as const)
  const m2 = torch.randn([20, 5] as const)
  const matmulResult = m1.matmul(m2) // Type: Tensor<[10, 5], ...>

  // Reshape
  const reshaped = a.reshape([6] as const)

  // Transpose
  const transposed = m1.transpose(0, 1) // [20, 10]

  // Reductions
  const totalSum = a.sum()
  const mean = a.mean()

  // Activations
  const relu = a.relu()
  const sigmoid = a.sigmoid()
  const softmax = a.softmax(1)

  return {
    sum,
    diff,
    prod,
    quot,
    matmulResult,
    reshaped,
    transposed,
    totalSum,
    mean,
    relu,
    sigmoid,
    softmax,
  }
}

// ==================== CUDA Support ====================

function cudaExample() {
  // Check CUDA availability
  if (torch.cuda.isAvailable()) {
    console.log(`Found ${torch.cuda.deviceCount()} CUDA devices`)

    // Create tensor on CPU then move to CUDA
    const cpuTensor = torch.zeros([100, 100] as const)
    const gpuTensor = cpuTensor.cuda(0)

    // Operations on GPU
    const result = gpuTensor.add(gpuTensor)

    // Move back to CPU
    const backToCpu = result.cpu()
    void backToCpu // Demonstrates moving tensor back to CPU
  } else {
    console.log('CUDA not available, using CPU')
  }
}

// ==================== Device Management ====================

function deviceExample() {
  // Create devices
  const cpu = Device.cpu()
  const gpu = Device.cuda(0)
  const mps = Device.mps()

  console.log(cpu.toString()) // "cpu"
  console.log(gpu.toString()) // "cuda:0"
  console.log(mps.toString()) // "mps"

  // Move tensor to device
  const t = torch.zeros([10, 10] as const)
  const gpuT = t.to(gpu.type)
  void gpuT // Demonstrates moving tensor to GPU device
}

// ==================== Type Safety Examples ====================

function typeSafetyExamples() {
  // These demonstrate compile-time type checking

  // Valid: shapes match for element-wise ops
  const a = torch.zeros([2, 3] as const)
  const b = torch.ones([2, 3] as const)
  a.add(b) // OK

  // Valid: matmul with compatible shapes
  const m1 = torch.randn([10, 20] as const)
  const m2 = torch.randn([20, 5] as const)
  const product = m1.matmul(m2) // OK, result is [10, 5]

  // Valid: reshape preserving element count
  const x = torch.zeros([2, 3, 4] as const) // 24 elements
  x.reshape([4, 6] as const) // 24 elements - OK

  // Type inference works (used in type level)
  type ProductShape = typeof product.shape // [10, 5]
  type ProductDType = typeof product.dtype // DType<"float32">
  // Satisfy type checker
  type _Check = [ProductShape, ProductDType]
}

// ==================== Advanced Usage ====================

function advancedExample() {
  // Chaining operations
  const result = torch
    .randn([100, 784] as const)
    .matmul(torch.randn([784, 128] as const))
    .relu()
    .matmul(torch.randn([128, 10] as const))
    .softmax(1)

  // Using with modules (future: when nn module is implemented)
  // const x = torch.zeros([1, 28, 28] as const);
  // const y = x.pipe(conv1).pipe(relu).pipe(pool).pipe(linear);

  return result
}

// ==================== Version Info ====================

function versionInfo() {
  const version = torch.version()
  console.log(`ts-torch v${version.major}.${version.minor}.${version.patch}`)
}

// ==================== Export Examples ====================

export {
  // Example functions
  tensorOperationsExample,
  cudaExample,
  deviceExample,
  typeSafetyExamples,
  advancedExample,
  versionInfo,
  asyncExample,
  // Demonstration tensors
  x,
  y,
  z,
  f64Tensor,
  f32Tensor,
  dataMatrix,
  nested,
  range,
  floatRange,
  result,
}

// Type-level tests to ensure the API works correctly
type _TestZeros = ReturnType<typeof torch.zeros<[2, 3], typeof float32>>
type _TestOnes = ReturnType<typeof torch.ones<[10, 20]>>
type _TestRandn = ReturnType<typeof torch.randn<[100, 50]>>

// Verify type inference
const _typeTest1: Tensor<[2, 3], typeof float32> = torch.zeros([2, 3] as const)
const _typeTest2: Tensor<[10, 20], typeof float32> = torch.ones([10, 20] as const)

// Verify matmul shape inference
const _m1 = torch.zeros([10, 20] as const)
const _m2 = torch.zeros([20, 5] as const)
const _matmul = _m1.matmul(_m2)
// _matmul should have shape [10, 5]
