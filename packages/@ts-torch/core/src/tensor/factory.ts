/**
 * Tensor factory functions for creating new tensors
 *
 * Provides type-safe tensor creation with compile-time shape checking.
 * All functions integrate with the memory scope system for automatic cleanup.
 */

import { Tensor } from './tensor.js'
import type { Shape } from '../types/shape.js'
import type { DType } from '../types/dtype.js'
import { DType as DTypeConstants } from '../types/dtype.js'
import { getLib } from '../ffi/loader.js'
import { withError, checkNull } from '../ffi/error.js'
import {
  ValidationError,
  validatePositiveInt,
  validateNonZero,
  validateFinite,
} from '../validation/index.js'

/**
 * Validate that a shape array contains only positive integers
 * @internal
 */
function validateShapeArray(shape: readonly number[], paramName: string = 'shape'): void {
  if (shape.length === 0) {
    throw new ValidationError(paramName, shape, 'a non-empty shape array')
  }
  for (let i = 0; i < shape.length; i++) {
    validatePositiveInt(shape[i]!, `${paramName}[${i}]`)
  }
}

/**
 * Create a tensor filled with zeros
 *
 * @template S - Shape type as readonly tuple
 * @template D - DType type
 * @param shape - Tensor shape
 * @param dtype - Data type (default: float32)
 * @param requiresGrad - Enable autograd tracking (default: false)
 * @returns New tensor filled with zeros
 *
 * @example
 * ```ts
 * const t = zeros([2, 3] as const, DType.float32);
 * // Type: Tensor<[2, 3], DType<"float32">>
 * ```
 */
export function zeros<S extends Shape, D extends DType<string> = DType<'float32'>>(
  shape: S,
  dtype: D = DTypeConstants.float32 as D,
  requiresGrad = false,
): Tensor<S, D> {
  validateShapeArray(shape)
  const lib = getLib()

  // Convert shape to BigInt64Array for FFI (koffi accepts ArrayBuffer directly)
  const shapeArray = new BigInt64Array(shape.map((dim) => BigInt(dim)))

  // Device: CPU (0), device_index: 0
  const handle = withError((err) => lib.ts_tensor_zeros(shapeArray.buffer, shape.length, dtype.value, 0, 0, err))

  checkNull(handle, 'Failed to create zeros tensor')

  const tensor = new Tensor<S, D>(handle!, shape, dtype)

  // Set requires_grad after creation
  if (requiresGrad) {
    tensor.requiresGrad = true
  }

  return tensor
}

/**
 * Create a tensor filled with ones
 *
 * @template S - Shape type as readonly tuple
 * @template D - DType type
 * @param shape - Tensor shape
 * @param dtype - Data type (default: float32)
 * @param requiresGrad - Enable autograd tracking (default: false)
 * @returns New tensor filled with ones
 *
 * @example
 * ```ts
 * const t = ones([2, 3] as const, DType.float32);
 * // Type: Tensor<[2, 3], DType<"float32">>
 * ```
 */
export function ones<S extends Shape, D extends DType<string> = DType<'float32'>>(
  shape: S,
  dtype: D = DTypeConstants.float32 as D,
  requiresGrad = false,
): Tensor<S, D> {
  validateShapeArray(shape)
  const lib = getLib()

  const shapeArray = new BigInt64Array(shape.map((dim) => BigInt(dim)))

  // Device: CPU (0), device_index: 0
  const handle = withError((err) => lib.ts_tensor_ones(shapeArray.buffer, shape.length, dtype.value, 0, 0, err))

  checkNull(handle, 'Failed to create ones tensor')

  const tensor = new Tensor<S, D>(handle!, shape, dtype)

  if (requiresGrad) {
    tensor.requiresGrad = true
  }

  return tensor
}

/**
 * Create an uninitialized tensor
 *
 * Memory is allocated but not initialized - contains arbitrary values.
 * Useful for performance when you'll immediately overwrite all values.
 *
 * @template S - Shape type as readonly tuple
 * @template D - DType type
 * @param shape - Tensor shape
 * @param dtype - Data type (default: float32)
 * @param requiresGrad - Enable autograd tracking (default: false)
 * @returns New uninitialized tensor
 *
 * @example
 * ```ts
 * const t = empty([1000, 1000] as const);
 * // Fast allocation, fill it yourself
 * ```
 */
export function empty<S extends Shape, D extends DType<string> = DType<'float32'>>(
  shape: S,
  dtype: D = DTypeConstants.float32 as D,
  requiresGrad = false,
): Tensor<S, D> {
  validateShapeArray(shape)
  const lib = getLib()

  const shapeArray = new BigInt64Array(shape.map((dim) => BigInt(dim)))

  // Device: CPU (0), device_index: 0
  const handle = withError((err) => lib.ts_tensor_empty(shapeArray.buffer, shape.length, dtype.value, 0, 0, err))

  checkNull(handle, 'Failed to create empty tensor')

  const tensor = new Tensor<S, D>(handle!, shape, dtype)

  if (requiresGrad) {
    tensor.requiresGrad = true
  }

  return tensor
}

/**
 * Create a tensor with random normal distribution (mean=0, std=1)
 *
 * @template S - Shape type as readonly tuple
 * @template D - DType type
 * @param shape - Tensor shape
 * @param dtype - Data type (default: float32)
 * @param requiresGrad - Enable autograd tracking (default: false)
 * @returns New tensor with random normal values
 *
 * @example
 * ```ts
 * const t = randn([100, 50] as const);
 * // Random initialization for neural networks
 * ```
 */
export function randn<S extends Shape, D extends DType<string> = DType<'float32'>>(
  shape: S,
  dtype: D = DTypeConstants.float32 as D,
  requiresGrad = false,
): Tensor<S, D> {
  validateShapeArray(shape)
  const lib = getLib()

  const shapeArray = new BigInt64Array(shape.map((dim) => BigInt(dim)))

  // Device: CPU (0), device_index: 0
  const handle = withError((err) => lib.ts_tensor_randn(shapeArray.buffer, shape.length, dtype.value, 0, 0, err))

  checkNull(handle, 'Failed to create randn tensor')

  const tensor = new Tensor<S, D>(handle!, shape, dtype)

  if (requiresGrad) {
    tensor.requiresGrad = true
  }

  return tensor
}

/**
 * Create tensor from JavaScript array or TypedArray
 *
 * @template S - Shape type as readonly tuple
 * @template D - DType type
 * @param data - Flat array of data (length must match shape product)
 * @param shape - Tensor shape
 * @param dtype - Data type (default: float32)
 * @param requiresGrad - Enable autograd tracking (default: false)
 * @returns New tensor with data copied from array
 *
 * @example
 * ```ts
 * const t = fromArray(
 *   [1, 2, 3, 4, 5, 6],
 *   [2, 3] as const,
 *   DType.float32
 * );
 * // [[1, 2, 3], [4, 5, 6]]
 * ```
 */
export function fromArray<S extends Shape, D extends DType<string> = DType<'float32'>>(
  data: number[] | Float32Array | Float64Array | Int32Array | BigInt64Array,
  shape: S,
  dtype: D = DTypeConstants.float32 as D,
  requiresGrad = false,
): Tensor<S, D> {
  validateShapeArray(shape)
  const lib = getLib()

  // Validate size
  const expectedSize = shape.reduce((acc, dim) => acc * dim, 1)
  if (data.length !== expectedSize) {
    throw new ValidationError(
      'data',
      `array of length ${data.length}`,
      `array of length ${expectedSize} to match shape [${shape.join(', ')}]`,
    )
  }

  // Convert to TypedArray if needed
  let typedData: Float32Array | Float64Array | Int32Array | BigInt64Array
  if (Array.isArray(data)) {
    switch (dtype.name) {
      case 'float32':
        typedData = new Float32Array(data)
        break
      case 'float64':
        typedData = new Float64Array(data)
        break
      case 'int32':
        typedData = new Int32Array(data)
        break
      case 'int64':
        typedData = new BigInt64Array(data.map((x) => BigInt(x)))
        break
      default:
        throw new Error(`Unsupported dtype: ${dtype.name}`)
    }
  } else {
    typedData = data as Float32Array | Float64Array | Int32Array | BigInt64Array
  }

  const shapeArray = new BigInt64Array(shape.map((dim) => BigInt(dim)))

  // Device: CPU (0), device_index: 0 (koffi accepts ArrayBuffer directly)
  const handle = withError((err) =>
    lib.ts_tensor_from_buffer(typedData.buffer, shapeArray.buffer, shape.length, dtype.value, 0, 0, err),
  )

  checkNull(handle, 'Failed to create tensor from array')

  const tensor = new Tensor<S, D>(handle!, shape, dtype)

  if (requiresGrad) {
    tensor.requiresGrad = true
  }

  return tensor
}

/**
 * Create 1D tensor with evenly spaced values
 *
 * @template D - DType type
 * @param start - Starting value (inclusive)
 * @param end - Ending value (exclusive)
 * @param step - Step size (default: 1)
 * @param dtype - Data type (default: float32)
 * @returns New 1D tensor
 *
 * @example
 * ```ts
 * const t = createArange(0, 10, 1); // [0, 1, 2, ..., 9]
 * const t2 = createArange(0, 1, 0.1); // [0.0, 0.1, 0.2, ..., 0.9]
 * ```
 */
export function createArange<D extends DType<string> = DType<'float32'>>(
  start: number,
  end: number,
  step = 1,
  dtype: D = DTypeConstants.float32 as D,
): Tensor<readonly [number], D> {
  validateFinite(start, 'start')
  validateFinite(end, 'end')
  validateNonZero(step, 'step')

  const size = Math.ceil((end - start) / step)
  if (size <= 0) {
    throw new ValidationError(
      'range',
      `start=${start}, end=${end}, step=${step}`,
      'a valid range where (end - start) / step > 0',
    )
  }

  // Generate data
  const data: number[] = []
  for (let i = 0; i < size; i++) {
    data.push(start + i * step)
  }

  return fromArray(data, [size] as const, dtype)
}

/**
 * Create tensor from nested array data (convenience wrapper)
 *
 * Automatically infers shape from nested arrays.
 *
 * @template D - DType type
 * @param data - Nested array data
 * @param dtype - Data type (default: float32)
 * @param requiresGrad - Enable autograd tracking (default: false)
 * @returns New tensor
 *
 * @example
 * ```ts
 * const t = createTensorFromData(
 *   [[1, 2, 3], [4, 5, 6]],
 *   DType.float32
 * );
 * // Type: Tensor<readonly [number, number], DType<"float32">>
 * ```
 */
export function createTensorFromData<D extends DType<string> = DType<'float32'>>(
  data: number | number[] | number[][] | number[][][] | number[][][][],
  dtype: D = DTypeConstants.float32 as D,
  requiresGrad = false,
): Tensor<readonly number[], D> {
  // Infer shape
  const shape = inferShape(data)

  // Flatten data
  const flatData = flattenArray(data)

  return fromArray(flatData, shape as readonly number[], dtype, requiresGrad)
}

/**
 * Infer shape from nested array
 * @internal
 */
function inferShape(data: any): number[] {
  const shape: number[] = []
  let current = data

  while (Array.isArray(current)) {
    shape.push(current.length)
    current = current[0]
  }

  return shape
}

/**
 * Flatten nested array to 1D
 * @internal
 */
function flattenArray(data: any): number[] {
  if (!Array.isArray(data)) {
    return [data]
  }

  const result: number[] = []

  function flatten(arr: any): void {
    for (const item of arr) {
      if (Array.isArray(item)) {
        flatten(item)
      } else {
        result.push(item)
      }
    }
  }

  flatten(data)
  return result
}
