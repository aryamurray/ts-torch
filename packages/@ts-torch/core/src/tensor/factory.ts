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
import type { DeviceType } from '../types/tensor.js'
import { getLib, koffi } from '../ffi/index.js'
import { withError, checkNull } from '../ffi/error.js'
import { shapeCache } from '../ffi/buffer-pool.js'
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

  // Use pooled shape buffer to reduce allocation overhead
  const shapeBuffer = shapeCache.fillShape(shape)
  try {
    // Device: CPU (0), device_index: 0
    const handle = withError((err) => lib.ts_tensor_zeros(shapeBuffer, shape.length, dtype.value, 0, 0, err))

    checkNull(handle, 'Failed to create zeros tensor')

    const tensor = new Tensor<S, D>(handle!, shape, dtype)

    // Set requires_grad after creation
    if (requiresGrad) {
      tensor.requiresGrad = true
    }

    return tensor
  } finally {
    shapeCache.release(shapeBuffer)
  }
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

  // Use pooled shape buffer to reduce allocation overhead
  const shapeBuffer = shapeCache.fillShape(shape)
  try {
    // Device: CPU (0), device_index: 0
    const handle = withError((err) => lib.ts_tensor_ones(shapeBuffer, shape.length, dtype.value, 0, 0, err))

    checkNull(handle, 'Failed to create ones tensor')

    const tensor = new Tensor<S, D>(handle!, shape, dtype)

    if (requiresGrad) {
      tensor.requiresGrad = true
    }

    return tensor
  } finally {
    shapeCache.release(shapeBuffer)
  }
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

  // Use pooled shape buffer to reduce allocation overhead
  const shapeBuffer = shapeCache.fillShape(shape)
  try {
    // Device: CPU (0), device_index: 0
    const handle = withError((err) => lib.ts_tensor_empty(shapeBuffer, shape.length, dtype.value, 0, 0, err))

    checkNull(handle, 'Failed to create empty tensor')

    const tensor = new Tensor<S, D>(handle!, shape, dtype)

    if (requiresGrad) {
      tensor.requiresGrad = true
    }

    return tensor
  } finally {
    shapeCache.release(shapeBuffer)
  }
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

  // Use pooled shape buffer to reduce allocation overhead
  const shapeBuffer = shapeCache.fillShape(shape)
  try {
    // Device: CPU (0), device_index: 0
    const handle = withError((err) => lib.ts_tensor_randn(shapeBuffer, shape.length, dtype.value, 0, 0, err))

    checkNull(handle, 'Failed to create randn tensor')

    const tensor = new Tensor<S, D>(handle!, shape, dtype)

    if (requiresGrad) {
      tensor.requiresGrad = true
    }

    return tensor
  } finally {
    shapeCache.release(shapeBuffer)
  }
}

/**
 * Create a tensor with random uniform distribution in [0, 1)
 *
 * @template S - Shape type as readonly tuple
 * @template D - DType type
 * @param shape - Tensor shape
 * @param dtype - Data type (default: float32)
 * @param requiresGrad - Enable autograd tracking (default: false)
 * @returns New tensor with random uniform values
 *
 * @example
 * ```ts
 * const t = rand([2, 2] as const);
 * // Random values in [0, 1)
 * ```
 */
export function rand<S extends Shape, D extends DType<string> = DType<'float32'>>(
  shape: S,
  dtype: D = DTypeConstants.float32 as D,
  requiresGrad = false,
): Tensor<S, D> {
  validateShapeArray(shape)
  const lib = getLib()

  // Use pooled shape buffer to reduce allocation overhead
  const shapeBuffer = shapeCache.fillShape(shape)
  try {
    // Device: CPU (0), device_index: 0
    const handle = withError((err) => lib.ts_tensor_rand(shapeBuffer, shape.length, dtype.value, 0, 0, err))

    checkNull(handle, 'Failed to create rand tensor')

    const tensor = new Tensor<S, D>(handle!, shape, dtype)

    if (requiresGrad) {
      tensor.requiresGrad = true
    }

    return tensor
  } finally {
    shapeCache.release(shapeBuffer)
  }
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

  // Use pooled shape buffer to reduce allocation overhead
  const shapeBuffer = shapeCache.fillShape(shape)
  try {
    // Device: CPU (0), device_index: 0
    const handle = withError((err) =>
      lib.ts_tensor_from_buffer(typedData.buffer, shapeBuffer, shape.length, dtype.value, 0, 0, err),
    )

    checkNull(handle, 'Failed to create tensor from array')

    const tensor = new Tensor<S, D>(handle!, shape, dtype)

    if (requiresGrad) {
      tensor.requiresGrad = true
    }

    return tensor
  } finally {
    shapeCache.release(shapeBuffer)
  }
}

/**
 * Create tensor from raw byte buffer
 *
 * For loading tensor data from binary formats (safetensors, checkpoints).
 * Passes raw bytes directly to the FFI layer.
 *
 * @template S - Shape type as readonly tuple
 * @template D - DType type
 * @param buffer - Raw bytes (must be correctly sized for shape * dtype.bytes)
 * @param shape - Tensor shape
 * @param dtype - Data type
 * @param requiresGrad - Enable autograd tracking (default: false)
 * @returns New tensor with data copied from buffer
 */
export function fromBuffer<S extends Shape, D extends DType<string> = DType<'float32'>>(
  buffer: Uint8Array,
  shape: S,
  dtype: D = DTypeConstants.float32 as D,
  requiresGrad = false,
): Tensor<S, D> {
  validateShapeArray(shape)
  const lib = getLib()

  const expectedSize = shape.reduce((acc, dim) => acc * dim, 1)
  const expectedBytes = expectedSize * dtype.bytes
  if (buffer.byteLength !== expectedBytes) {
    throw new ValidationError(
      'buffer',
      `buffer of ${buffer.byteLength} bytes`,
      `buffer of ${expectedBytes} bytes to match shape [${shape.join(', ')}] with dtype ${dtype.name} (${dtype.bytes} bytes per element)`,
    )
  }

  // Ensure we pass a properly aligned ArrayBuffer
  const aligned = new ArrayBuffer(buffer.byteLength)
  new Uint8Array(aligned).set(buffer)

  const shapeBuffer = shapeCache.fillShape(shape)
  try {
    const handle = withError((err) =>
      lib.ts_tensor_from_buffer(aligned, shapeBuffer, shape.length, dtype.value, 0, 0, err),
    )

    checkNull(handle, 'Failed to create tensor from buffer')

    const tensor = new Tensor<S, D>(handle!, shape, dtype)

    if (requiresGrad) {
      tensor.requiresGrad = true
    }

    return tensor
  } finally {
    shapeCache.release(shapeBuffer)
  }
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

/**
 * Concatenate tensors along a dimension
 *
 * Uses native FFI to concatenate tensors, preserving device placement.
 * All input tensors must be on the same device.
 *
 * @param tensors - Array of tensors to concatenate (must have same shape except along dim)
 * @param dim - Dimension along which to concatenate (default: 0)
 * @returns New tensor with concatenated data on the same device as inputs
 *
 * @example
 * ```ts
 * const a = fromArray([1, 2, 3], [1, 3] as const);
 * const b = fromArray([4, 5, 6], [1, 3] as const);
 * const c = cat([a, b], 0); // shape: [2, 3], data: [[1,2,3], [4,5,6]]
 * ```
 */
export function cat<D extends DType<string> = DType<'float32'>, Dev extends DeviceType = DeviceType>(
  tensors: Tensor<Shape, D, Dev>[],
  dim = 0,
): Tensor<Shape, D, Dev> {
  if (tensors.length === 0) {
    throw new ValidationError('tensors', 'empty array', 'at least one tensor')
  }

  if (tensors.length === 1) {
    // Single tensor - just clone it
    return tensors[0]!.clone() as Tensor<Shape, D, Dev>
  }

  const first = tensors[0]!
  const dtype = first.dtype
  const device = first.device as Dev
  const ndim = first.shape.length

  // Validate dimension
  if (dim < 0) dim = ndim + dim
  if (dim < 0 || dim >= ndim) {
    throw new ValidationError('dim', dim, `a value between 0 and ${ndim - 1}`)
  }

  // Validate all tensors have compatible shapes and same device
  const resultShape = [...first.shape] as number[]
  for (let i = 1; i < tensors.length; i++) {
    const t = tensors[i]!
    if (t.device !== device) {
      throw new ValidationError(
        `tensors[${i}].device`,
        t.device,
        `same device as tensors[0] (${device})`,
      )
    }
    if (t.shape.length !== ndim) {
      throw new ValidationError(
        `tensors[${i}].shape`,
        t.shape,
        `same number of dimensions as tensors[0] (${ndim})`,
      )
    }
    for (let d = 0; d < ndim; d++) {
      if (d === dim) {
        resultShape[d]! += t.shape[d]!
      } else if (t.shape[d] !== first.shape[d]) {
        throw new ValidationError(
          `tensors[${i}].shape[${d}]`,
          t.shape[d],
          `${first.shape[d]} (same as tensors[0])`,
        )
      }
    }
  }

  const lib = getLib()

  // Get raw pointer addresses from tensor handles using koffi.address()
  // Pack them into a BigUint64Array buffer for passing to FFI
  const handleAddresses = new BigUint64Array(tensors.length)
  for (let i = 0; i < tensors.length; i++) {
    const handle = tensors[i]!.handle
    // koffi.address() returns the raw address of an opaque pointer as BigInt
    handleAddresses[i] = koffi.address(handle)
  }

  // Call native ts_tensor_cat
  const handle = withError((err) =>
    lib.ts_tensor_cat(handleAddresses.buffer, tensors.length, dim, err),
  )

  checkNull(handle, 'Failed to concatenate tensors')

  return new Tensor<Shape, D, Dev>(handle!, resultShape as Shape, dtype, device)
}

/**
 * Stack tensors along a new dimension
 *
 * All tensors must have the same shape, dtype, and device.
 *
 * @param tensors - Array of tensors to stack
 * @param dim - New dimension index (default: 0)
 * @returns New tensor with stacked data
 *
 * @example
 * ```ts
 * const a = fromArray([1, 2], [2] as const)
 * const b = fromArray([3, 4], [2] as const)
 * const c = stack([a, b], 0) // shape: [2, 2]
 * ```
 */
export function stack<D extends DType<string> = DType<'float32'>, Dev extends DeviceType = DeviceType>(
  tensors: Tensor<Shape, D, Dev>[],
  dim: number = 0,
): Tensor<Shape, D, Dev> {
  if (tensors.length === 0) {
    throw new ValidationError('tensors', 'empty array', 'at least one tensor')
  }

  const first = tensors[0]!
  const dtype = first.dtype
  const device = first.device as Dev
  const shape = first.shape as readonly number[]

  const ndim = shape.length
  const normalizedDim = dim < 0 ? dim + ndim + 1 : dim
  if (!Number.isInteger(dim) || normalizedDim < 0 || normalizedDim > ndim) {
    throw new ValidationError('dim', dim, `a value between ${-(ndim + 1)} and ${ndim}`)
  }

  for (let i = 1; i < tensors.length; i++) {
    const t = tensors[i]!
    if (t.device !== device) {
      throw new ValidationError(
        `tensors[${i}].device`,
        t.device,
        `same device as tensors[0] (${device})`,
      )
    }
    if (t.dtype.name !== dtype.name) {
      throw new ValidationError(
        `tensors[${i}].dtype`,
        t.dtype.name,
        `same dtype as tensors[0] (${dtype.name})`,
      )
    }
    if (t.shape.length !== shape.length || !t.shape.every((dimSize, idx) => dimSize === shape[idx])) {
      throw new ValidationError(`tensors[${i}].shape`, t.shape, `same shape as tensors[0] (${shape})`)
    }
  }

  const expanded = tensors.map((tensor) => tensor.unsqueeze(normalizedDim))
  return cat(expanded as unknown as Tensor<Shape, D, Dev>[], normalizedDim)
}

/**
 * Einstein summation notation for tensor operations
 *
 * Provides a powerful way to express tensor contractions and operations using
 * Einstein notation. Widely used in attention mechanisms and ML operations.
 *
 * @param equation - Einstein summation notation string (e.g., 'ij,jk->ik')
 * @param tensors - Array of input tensors
 * @returns Result tensor
 *
 * @example
 * ```ts
 * // Matrix multiplication: C = A @ B
 * const result = einsum('ij,jk->ik', [a, b])
 *
 * // Batched matrix multiplication
 * const batched = einsum('bij,bjk->bik', [a, b])
 *
 * // Matrix trace
 * const trace = einsum('ii->', [matrix])
 *
 * // Transpose
 * const transposed = einsum('ij->ji', [matrix])
 *
 * // Dot product
 * const dot = einsum('i,i->', [a, b])
 *
 * // Outer product
 * const outer = einsum('i,j->ij', [a, b])
 * ```
 */
export function einsum<D extends DType<string> = DType<'float32'>, Dev extends DeviceType = DeviceType>(
  equation: string,
  tensors: Tensor<Shape, D, Dev>[],
): Tensor<Shape, D, Dev> {
  if (tensors.length === 0) {
    throw new ValidationError('tensors', 'empty array', 'at least one tensor')
  }

  if (!equation || typeof equation !== 'string') {
    throw new ValidationError('equation', equation, 'a valid einsum equation string')
  }

  const first = tensors[0]!
  const dtype = first.dtype
  const device = first.device as Dev

  // Validate all tensors have same dtype and device
  for (let i = 1; i < tensors.length; i++) {
    const t = tensors[i]!
    if (t.device !== device) {
      throw new ValidationError(
        `tensors[${i}].device`,
        t.device,
        `same device as tensors[0] (${device})`,
      )
    }
    if (t.dtype.name !== dtype.name) {
      throw new ValidationError(
        `tensors[${i}].dtype`,
        t.dtype.name,
        `same dtype as tensors[0] (${dtype.name})`,
      )
    }
  }

  const lib = getLib()

  // Encode the equation string as a null-terminated buffer
  const encoder = new TextEncoder()
  const equationBytes = encoder.encode(equation + '\0')
  const equationBuffer = equationBytes.buffer

  // Get raw pointer addresses from tensor handles using koffi.address()
  // Pack them into a BigUint64Array buffer for passing to FFI
  const handleAddresses = new BigUint64Array(tensors.length)
  for (let i = 0; i < tensors.length; i++) {
    const handle = tensors[i]!.handle
    // koffi.address() returns the raw address of an opaque pointer as BigInt
    handleAddresses[i] = koffi.address(handle)
  }

  // Call native ts_tensor_einsum
  const handle = withError((err) =>
    lib.ts_tensor_einsum(equationBuffer, handleAddresses.buffer, tensors.length, err),
  )

  checkNull(handle, 'Failed to perform einsum operation')

  // Query the result tensor's shape since einsum's output shape
  // depends on the equation which is only known at runtime
  const ndim = withError((err) => lib.ts_tensor_ndim(handle!, err)) as number
  const shapeArray: number[] = []
  for (let i = 0; i < ndim; i++) {
    const size = withError((err) => lib.ts_tensor_size(handle!, i, err)) as number
    shapeArray.push(size)
  }

  return new Tensor<Shape, D, Dev>(handle!, shapeArray as unknown as Shape, dtype, device)
}
