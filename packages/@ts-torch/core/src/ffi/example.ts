/**
 * Comprehensive FFI usage example
 * Demonstrates the complete workflow from library loading to tensor operations
 *
 * NOTE: This example requires the native library to be built and available
 * Build instructions: cd native && cargo build --release
 */

import {
  getLib,
  createError,
  checkError,
  withError,
  checkNull,
  validateShape,
  validateDtype,
  ptr,
  type TensorHandle,
} from "./index.js";

/**
 * Example 1: Basic tensor creation and cleanup
 */
function example1_basic_tensor() {
  console.log("\n=== Example 1: Basic Tensor Creation ===");

  const lib = getLib();

  // Validate inputs
  const shape = [2, 3];
  validateShape(shape);
  const dtype = 0; // f32
  validateDtype(dtype);

  // Create shape buffer
  const shapeBuffer = new BigInt64Array(shape.map(BigInt));

  // Create tensor with manual error handling
  const err = createError();
  const handle = lib.symbols.ts_tensor_zeros(
    ptr(shapeBuffer),
    shape.length,
    dtype,
    false, // requires_grad
    err,
  );

  checkError(err);
  checkNull(handle, "Failed to create tensor");

  console.log("Created tensor successfully:", handle);

  // Get tensor properties
  const ndim = lib.symbols.ts_tensor_ndim(handle, err);
  checkError(err);
  console.log("Tensor ndim:", ndim);

  const numel = lib.symbols.ts_tensor_numel(handle, err);
  checkError(err);
  console.log("Tensor numel:", numel);

  // Cleanup
  lib.symbols.ts_tensor_delete(handle);
  console.log("Tensor deleted");
}

/**
 * Example 2: Using withError for cleaner code
 */
function example2_with_error() {
  console.log("\n=== Example 2: Using withError() ===");

  const lib = getLib();
  const shape = new BigInt64Array([3, 4]);

  // Create tensor with automatic error handling
  const handle = withError((err) =>
    lib.symbols.ts_tensor_ones(ptr(shape), shape.length, 0, false, err),
  );

  console.log("Created ones tensor:", handle);

  // Get properties
  const dtype = withError((err) => lib.symbols.ts_tensor_dtype(handle, err));
  const requiresGrad = withError((err) => lib.symbols.ts_tensor_requires_grad(handle, err));

  console.log("dtype:", dtype, "requires_grad:", requiresGrad);

  // Cleanup
  lib.symbols.ts_tensor_delete(handle);
}

/**
 * Example 3: Tensor operations
 */
function example3_operations() {
  console.log("\n=== Example 3: Tensor Operations ===");

  const lib = getLib();
  const shape = new BigInt64Array([2, 2]);

  // Create two tensors
  const a = withError((err) => lib.symbols.ts_tensor_ones(ptr(shape), 2, 0, false, err));

  const b = withError((err) => lib.symbols.ts_tensor_ones(ptr(shape), 2, 0, false, err));

  console.log("Created tensors a and b");

  // Add tensors
  const sum = withError((err) => lib.symbols.ts_tensor_add(a, b, err));
  console.log("sum = a + b:", sum);

  // Multiply tensors
  const product = withError((err) => lib.symbols.ts_tensor_mul(a, b, err));
  console.log("product = a * b:", product);

  // Matrix multiplication
  const matmul = withError((err) => lib.symbols.ts_tensor_matmul(a, b, err));
  console.log("matmul = a @ b:", matmul);

  // Cleanup
  [a, b, sum, product, matmul].forEach((handle) => {
    lib.symbols.ts_tensor_delete(handle);
  });
}

/**
 * Example 4: Reading tensor data
 */
function example4_read_data() {
  console.log("\n=== Example 4: Reading Tensor Data ===");

  const lib = getLib();
  const shape = new BigInt64Array([2, 3]);

  // Create tensor
  const handle = withError((err) => lib.symbols.ts_tensor_randn(ptr(shape), 2, 0, false, err));

  // Get number of elements
  const numel = withError((err) => lib.symbols.ts_tensor_numel(handle, err));
  console.log("Tensor has", numel, "elements");

  // Allocate buffer and copy data
  const buffer = new Float32Array(Number(numel));
  withError((err) =>
    lib.symbols.ts_tensor_copy_to_buffer(handle, ptr(buffer), BigInt(buffer.byteLength), err),
  );

  console.log("Tensor data:", Array.from(buffer));

  // Get data pointer (alternative method)
  const dataPtr = withError((err) => lib.symbols.ts_tensor_data_ptr(handle, err));
  console.log("Data pointer:", dataPtr);

  // Cleanup
  lib.symbols.ts_tensor_delete(handle);
}

/**
 * Example 5: Scope management for automatic cleanup
 */
function example5_scopes() {
  console.log("\n=== Example 5: Scope Management ===");

  const lib = getLib();
  const shape = new BigInt64Array([2, 2]);

  // Begin scope
  const scopeId = lib.symbols.ts_scope_begin();
  console.log("Scope created:", scopeId);

  try {
    // Create tensors and register with scope
    const a = withError((err) => lib.symbols.ts_tensor_ones(ptr(shape), 2, 0, false, err));
    withError((err) => lib.symbols.ts_scope_register_tensor(a, err));

    const b = withError((err) => lib.symbols.ts_tensor_ones(ptr(shape), 2, 0, false, err));
    withError((err) => lib.symbols.ts_scope_register_tensor(b, err));

    const c = withError((err) => lib.symbols.ts_tensor_add(a, b, err));
    withError((err) => lib.symbols.ts_scope_register_tensor(c, err));

    console.log("Created and registered 3 tensors");

    // Escape tensor c from automatic cleanup
    withError((err) => lib.symbols.ts_scope_escape_tensor(c, err));
    console.log("Escaped tensor c from scope");

    // End scope - only a and b will be deleted
    withError((err) => lib.symbols.ts_scope_end(scopeId, err));
    console.log("Scope ended - a and b automatically deleted");

    // Manually delete escaped tensor
    lib.symbols.ts_tensor_delete(c);
    console.log("Manually deleted tensor c");
  } catch (err) {
    // Cleanup on error
    withError((e) => lib.symbols.ts_scope_end(scopeId, e));
    throw err;
  }
}

/**
 * Example 6: Activation functions
 */
function example6_activations() {
  console.log("\n=== Example 6: Activation Functions ===");

  const lib = getLib();
  const shape = new BigInt64Array([3, 3]);

  const x = withError((err) => lib.symbols.ts_tensor_randn(ptr(shape), 2, 0, false, err));

  // Apply various activations
  const relu = withError((err) => lib.symbols.ts_tensor_relu(x, err));
  console.log("ReLU applied:", relu);

  const sigmoid = withError((err) => lib.symbols.ts_tensor_sigmoid(x, err));
  console.log("Sigmoid applied:", sigmoid);

  const tanh = withError((err) => lib.symbols.ts_tensor_tanh(x, err));
  console.log("Tanh applied:", tanh);

  const softmax = withError((err) => lib.symbols.ts_tensor_softmax(x, -1, err));
  console.log("Softmax applied:", softmax);

  // Cleanup
  [x, relu, sigmoid, tanh, softmax].forEach((h) => {
    lib.symbols.ts_tensor_delete(h);
  });
}

/**
 * Example 7: Autograd
 */
function example7_autograd() {
  console.log("\n=== Example 7: Autograd ===");

  const lib = getLib();
  const shape = new BigInt64Array([2, 2]);

  // Create tensor with gradients enabled
  const x = withError(
    (err) => lib.symbols.ts_tensor_ones(ptr(shape), 2, 0, true, err), // requires_grad = true
  );

  const y = withError((err) => lib.symbols.ts_tensor_ones(ptr(shape), 2, 0, true, err));

  // Perform operations
  const z = withError((err) => lib.symbols.ts_tensor_mul(x, y, err));
  const loss = withError((err) => lib.symbols.ts_tensor_sum(z, -1, false, err));

  console.log("Created computation graph: loss = sum(x * y)");

  // Backward pass
  withError((err) => lib.symbols.ts_tensor_backward(loss, err));
  console.log("Backward pass completed");

  // Get gradients
  const xGrad = withError((err) => lib.symbols.ts_tensor_grad(x, err));
  const yGrad = withError((err) => lib.symbols.ts_tensor_grad(y, err));

  console.log("x.grad:", xGrad);
  console.log("y.grad:", yGrad);

  // Zero gradients
  withError((err) => lib.symbols.ts_tensor_zero_grad(x, err));
  withError((err) => lib.symbols.ts_tensor_zero_grad(y, err));
  console.log("Gradients zeroed");

  // Cleanup
  [x, y, z, loss].forEach((h) => lib.symbols.ts_tensor_delete(h));

  if (xGrad) lib.symbols.ts_tensor_delete(xGrad);
  if (yGrad) lib.symbols.ts_tensor_delete(yGrad);
}

/**
 * Example 8: CUDA operations
 */
function example8_cuda() {
  console.log("\n=== Example 8: CUDA Operations ===");

  const lib = getLib();

  // Check CUDA availability
  const cudaAvailable = lib.symbols.ts_cuda_is_available();
  console.log("CUDA available:", cudaAvailable);

  if (cudaAvailable) {
    const deviceCount = lib.symbols.ts_cuda_device_count();
    console.log("CUDA device count:", deviceCount);

    if (deviceCount > 0) {
      const shape = new BigInt64Array([100, 100]);

      // Create tensor on CPU
      const cpuTensor = withError((err) =>
        lib.symbols.ts_tensor_randn(ptr(shape), 2, 0, false, err),
      );

      console.log("Created CPU tensor:", cpuTensor);

      // Move to GPU
      const gpuTensor = withError((err) =>
        lib.symbols.ts_tensor_to_device(
          cpuTensor,
          1, // device_type: 1=CUDA
          0, // device_id: first GPU
          err,
        ),
      );

      console.log("Moved to GPU:", gpuTensor);

      // Perform GPU operation
      const result = withError((err) => lib.symbols.ts_tensor_relu(gpuTensor, err));
      console.log("Applied ReLU on GPU:", result);

      // Move back to CPU
      const backToCpu = withError((err) =>
        lib.symbols.ts_tensor_to_device(
          result,
          0, // device_type: 0=CPU
          0,
          err,
        ),
      );

      console.log("Moved back to CPU:", backToCpu);

      // Cleanup
      [cpuTensor, gpuTensor, result, backToCpu].forEach((h) => {
        lib.symbols.ts_tensor_delete(h);
      });
    }
  } else {
    console.log("CUDA not available - skipping GPU operations");
  }
}

/**
 * Example 9: Version info
 */
function example9_version() {
  console.log("\n=== Example 9: Version Info ===");

  const lib = getLib();

  // Get version string
  const versionPtr = lib.symbols.ts_version();
  // Note: In a real implementation, you'd need to read the C string
  console.log("Version pointer:", versionPtr);
}

/**
 * Main function - run all examples
 */
export function runAllExamples() {
  console.log("=".repeat(60));
  console.log("ts-torch FFI Examples");
  console.log("=".repeat(60));

  try {
    example1_basic_tensor();
    example2_with_error();
    example3_operations();
    example4_read_data();
    example5_scopes();
    example6_activations();
    example7_autograd();
    example8_cuda();
    example9_version();

    console.log("\n" + "=".repeat(60));
    console.log("All examples completed successfully!");
    console.log("=".repeat(60));
  } catch (err) {
    console.error("\n" + "=".repeat(60));
    console.error("Error running examples:");
    console.error(err);
    console.error("=".repeat(60));

    if (err instanceof Error && err.message.includes("Could not find")) {
      console.error("\nThe native library is not available.");
      console.error("Please build it first:");
      console.error("  cd native && cargo build --release");
    }
  }
}

// Run if executed directly
if (import.meta.main) {
  runAllExamples();
}
