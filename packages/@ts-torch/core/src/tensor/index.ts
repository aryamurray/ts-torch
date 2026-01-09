/**
 * Tensor operations and core tensor class
 *
 * This module provides the main Tensor class and factory functions for creating tensors.
 *
 * @example
 * ```ts
 * import { Tensor, zeros, ones, fromArray } from '@ts-torch/core/tensor';
 * import { DType } from '@ts-torch/core/types';
 *
 * // Create tensors
 * const a = zeros([2, 3], DType.float32);
 * const b = ones([2, 3], DType.float32);
 * const c = fromArray([[1, 2], [3, 4]], [2, 2], DType.float32);
 *
 * // Perform operations
 * const sum = a.add(b);
 * const product = a.matmul(b.transpose(0, 1));
 * const activated = sum.relu();
 * ```
 *
 * @module @ts-torch/core/tensor
 */

// Core Tensor class
export { Tensor } from "./tensor.js";

// Factory functions
export {
  zeros,
  ones,
  empty,
  randn,
  fromArray,
  createArange,
  createTensorFromData,
} from "./factory.js";

// Re-export types for convenience
export type { Shape } from "../types/shape.js";
export type { DType, DTypeName, DTypeToTypedArray } from "../types/dtype.js";
export type { Device, TensorOptions } from "../types/index.js";
