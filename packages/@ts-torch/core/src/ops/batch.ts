/**
 * Batched Operations API (Phase 2)
 *
 * Provides two ways to reduce FFI overhead:
 *
 * 1. Recording Mode (batch()): Records operations and executes them
 *    in a single FFI call. Best for complex computation graphs.
 *
 * 2. Direct Batched Ops (chainMatmul, mlpForward): Execute common
 *    patterns in a single FFI call without recording overhead.
 *
 * @module ops/batch
 */

import { getLib } from '../ffi/loader.js'
import { withErrorFast, type Pointer } from '../ffi/error.js'
import { Tensor, type AnyTensor } from '../tensor/tensor.js'
import type { DType, DTypeName } from '../types/index.js'

// ============================================================================
// Recording Mode API
// ============================================================================

/**
 * Execute a batch of tensor operations in a single FFI round-trip.
 *
 * All tensor operations within the callback are recorded and executed
 * together when the callback returns. Operations return placeholder
 * tensors that can be chained normally.
 *
 * NOTE: This is experimental and currently has limited op support.
 *
 * @template T - Tensor type returned
 * @param fn - Function that performs tensor operations
 * @returns The result tensor from the last operation
 *
 * @example
 * ```ts
 * import { batch } from '@ts-torch/core/ops';
 *
 * // Without batching: 5 FFI calls
 * const result1 = input.matmul(w1).relu().matmul(w2).relu().matmul(w3);
 *
 * // With batching: 1 FFI call for all operations
 * const result2 = batch(() => {
 *   return input.matmul(w1).relu().matmul(w2).relu().matmul(w3);
 * });
 * ```
 */
export function batch<T extends AnyTensor>(fn: () => T): T {
  const lib = getLib()

  const batchHandle = withErrorFast((err) => lib.ts_batch_begin(err))

  try {
    // Execute callback - ops will record and return placeholders
    const result = fn()

    // End batch and execute all recorded ops
    const resultHandle = withErrorFast((err) => lib.ts_batch_end(batchHandle, err))

    // Create result tensor with same metadata as placeholder result
    // The actual data comes from batch_end execution
    return new Tensor(
      resultHandle as Pointer,
      result.shape as readonly number[],
      result.dtype,
      result.device,
    ) as T
  } catch (e) {
    // Abort batch on error
    lib.ts_batch_abort(batchHandle)
    throw e
  }
}

/**
 * Check if currently in batch recording mode.
 *
 * @returns true if inside a batch() call, false otherwise
 */
export function isBatchRecording(): boolean {
  const lib = getLib()
  return lib.ts_batch_is_recording() !== 0
}

// ============================================================================
// Direct Batched Operations
// ============================================================================

/**
 * Chain matrix multiplication: A @ B @ C @ D @ ...
 *
 * Executes all matrix multiplications in a single FFI call,
 * reducing overhead compared to chaining .matmul() calls.
 *
 * @param tensors - Array of tensors to multiply (minimum 2)
 * @returns Result of A @ B @ C @ ...
 *
 * @example
 * ```ts
 * import { chainMatmul } from '@ts-torch/core/ops';
 *
 * const result = chainMatmul([A, B, C, D, E]);
 * // Equivalent to: A.matmul(B).matmul(C).matmul(D).matmul(E)
 * // But with only 1 FFI call instead of 4
 * ```
 */
export function chainMatmul<D extends DTypeName = 'float32'>(
  tensors: Tensor<readonly number[], DType<D>>[],
): Tensor<readonly number[], DType<D>> {
  if (tensors.length < 2) {
    throw new Error('chainMatmul requires at least 2 tensors')
  }

  const lib = getLib()

  // Create array of handles for FFI
  const handles = tensors.map((t) => t.handle)
  const handleArray = new BigUint64Array(handles.length)
  for (let i = 0; i < handles.length; i++) {
    // Convert handle to BigInt for the pointer array
    handleArray[i] = BigInt(handles[i] as unknown as number)
  }

  const resultHandle = withErrorFast((err) =>
    lib.ts_tensor_chain_matmul(handleArray, tensors.length, err),
  )

  // Compute result shape: first tensor's batch dims + last tensor's final dim
  const first = tensors[0]!
  const last = tensors[tensors.length - 1]!
  const resultShape = [...first.shape.slice(0, -1), last.shape[last.shape.length - 1]] as readonly number[]

  return new Tensor(resultHandle as Pointer, resultShape, first.dtype, first.device)
}

/**
 * MLP forward pass: input @ W1 + b1 -> relu -> @ W2 + b2 -> ...
 *
 * Executes a complete multi-layer perceptron forward pass in a single
 * FFI call, eliminating the overhead of multiple layer calls.
 *
 * @param input - Input tensor [batch, in_features]
 * @param weights - Array of weight tensors [out, in] for each layer
 * @param biases - Array of bias tensors [out] (can be null/undefined for no bias)
 * @param applyReluExceptLast - Apply ReLU activation between layers (default: true)
 * @returns Output tensor [batch, final_out_features]
 *
 * @example
 * ```ts
 * import { mlpForward } from '@ts-torch/core/ops';
 *
 * // 3-layer MLP: 784 -> 256 -> 128 -> 10
 * const output = mlpForward(
 *   input,                    // [batch, 784]
 *   [w1, w2, w3],            // [256,784], [128,256], [10,128]
 *   [b1, b2, b3],            // [256], [128], [10]
 *   true                      // ReLU between layers
 * );
 * // Result: [batch, 10]
 * ```
 */
export function mlpForward<D extends DTypeName = 'float32'>(
  input: Tensor<readonly number[], DType<D>>,
  weights: Tensor<readonly number[], DType<D>>[],
  biases?: (Tensor<readonly number[], DType<D>> | null | undefined)[],
  applyReluExceptLast: boolean = true,
): Tensor<readonly number[], DType<D>> {
  if (weights.length === 0) {
    throw new Error('mlpForward requires at least 1 layer')
  }

  const lib = getLib()

  // Create arrays of handles
  const weightHandles = new BigUint64Array(weights.length)
  for (let i = 0; i < weights.length; i++) {
    weightHandles[i] = BigInt(weights[i]!.handle as unknown as number)
  }

  const biasHandles = new BigUint64Array(weights.length)
  for (let i = 0; i < weights.length; i++) {
    const bias = biases?.[i]
    biasHandles[i] = bias ? BigInt(bias.handle as unknown as number) : BigInt(0)
  }

  const resultHandle = withErrorFast((err) =>
    lib.ts_tensor_mlp_forward(
      input.handle,
      weightHandles,
      biasHandles,
      weights.length,
      applyReluExceptLast ? 1 : 0,
      err,
    ),
  )

  // Compute result shape: [batch, last_layer_out_features]
  const lastWeight = weights[weights.length - 1]!
  const batchDims = input.shape.slice(0, -1)
  const outFeatures = lastWeight.shape[0]!
  const resultShape = [...batchDims, outFeatures] as readonly number[]

  return new Tensor(resultHandle as Pointer, resultShape, input.dtype, input.device)
}

// ============================================================================
// Thread Controls (Phase 6)
// ============================================================================

/**
 * Set the number of threads used by LibTorch for inter-op parallelism.
 *
 * WARNING: This is a GLOBAL setting that affects all tensor operations.
 * Call this early in your program, before any tensor operations.
 *
 * @param numThreads - Number of threads (0 or undefined = auto/default)
 *
 * @example
 * ```ts
 * import { setNumThreads, getNumThreads } from '@ts-torch/core/ops';
 *
 * // Use single thread for deterministic benchmarks
 * setNumThreads(1);
 *
 * // Restore auto mode
 * setNumThreads(0);
 *
 * console.log(`Using ${getNumThreads()} threads`);
 * ```
 */
export function setNumThreads(numThreads: number = 0): void {
  const lib = getLib()
  lib.ts_set_num_threads(numThreads)
}

/**
 * Get the current number of threads used by LibTorch.
 *
 * @returns Current thread count
 */
export function getNumThreads(): number {
  const lib = getLib()
  return lib.ts_get_num_threads() as number
}
