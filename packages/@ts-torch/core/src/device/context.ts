/**
 * DeviceContext - Declarative device-bound tensor creation
 *
 * Instead of creating tensors and then moving them, DeviceContext
 * creates tensors directly on the target device.
 *
 * @example
 * ```ts
 * const cuda = device.cuda(0)
 *
 * // Tensors are created directly on GPU - no transfer
 * const x = cuda.zeros([784, 128])
 * const y = cuda.randn([128, 10])
 * ```
 */

import { Tensor } from '../tensor/tensor.js'
import type { Shape } from '../types/shape.js'
import type { DType } from '../types/dtype.js'
import { DType as DTypeConstants } from '../types/dtype.js'
import { getLib } from '../ffi/loader.js'
import { withError, checkNull } from '../ffi/error.js'
import { ValidationError, validatePositiveInt } from '../validation/index.js'

/** Device type enum values for FFI */
const DeviceType = {
  cpu: 0,
  cuda: 1,
  mps: 2,
} as const

type DeviceTypeName = keyof typeof DeviceType

/**
 * Validate shape array
 * @internal
 */
function validateShapeArray(shape: readonly number[], paramName = 'shape'): void {
  if (shape.length === 0) {
    throw new ValidationError(paramName, shape, 'a non-empty shape array')
  }
  for (let i = 0; i < shape.length; i++) {
    validatePositiveInt(shape[i]!, `${paramName}[${i}]`)
  }
}

/**
 * DeviceContext - A device-bound context for creating tensors
 *
 * All tensor factory methods create tensors directly on the device,
 * avoiding expensive CPU-to-GPU transfers.
 */
export class DeviceContext {
  /** Device type: 'cpu', 'cuda', or 'mps' */
  readonly type: DeviceTypeName

  /** Device index (e.g., GPU 0, GPU 1) */
  readonly index: number

  private constructor(type: DeviceTypeName, index: number) {
    this.type = type
    this.index = index
  }

  /** Create a CPU device context */
  static cpu(): DeviceContext {
    return new DeviceContext('cpu', 0)
  }

  /** Create a CUDA device context */
  static cuda(index = 0): DeviceContext {
    return new DeviceContext('cuda', index)
  }

  /** Create an MPS device context (Apple Silicon) */
  static mps(): DeviceContext {
    return new DeviceContext('mps', 0)
  }

  /** Get the FFI device type value */
  private get deviceTypeValue(): number {
    return DeviceType[this.type]
  }

  /**
   * Create a tensor filled with zeros on this device
   *
   * @example
   * ```ts
   * const cuda = device.cuda(0)
   * const x = cuda.zeros([2, 3])
   * ```
   */
  zeros<S extends Shape, D extends DType<string> = DType<'float32'>>(
    shape: S,
    dtype: D = DTypeConstants.float32 as D,
    requiresGrad = false,
  ): Tensor<S, D> {
    validateShapeArray(shape)
    const lib = getLib()
    const shapeArray = new BigInt64Array(shape.map((dim) => BigInt(dim)))

    const handle = withError((err) =>
      lib.ts_tensor_zeros(shapeArray.buffer, shape.length, dtype.value, this.deviceTypeValue, this.index, err),
    )

    checkNull(handle, `Failed to create zeros tensor on ${this}`)
    const tensor = new Tensor<S, D>(handle!, shape, dtype)
    if (requiresGrad) tensor.requiresGrad = true
    return tensor
  }

  /**
   * Create a tensor filled with ones on this device
   *
   * @example
   * ```ts
   * const cuda = device.cuda(0)
   * const x = cuda.ones([2, 3])
   * ```
   */
  ones<S extends Shape, D extends DType<string> = DType<'float32'>>(
    shape: S,
    dtype: D = DTypeConstants.float32 as D,
    requiresGrad = false,
  ): Tensor<S, D> {
    validateShapeArray(shape)
    const lib = getLib()
    const shapeArray = new BigInt64Array(shape.map((dim) => BigInt(dim)))

    const handle = withError((err) =>
      lib.ts_tensor_ones(shapeArray.buffer, shape.length, dtype.value, this.deviceTypeValue, this.index, err),
    )

    checkNull(handle, `Failed to create ones tensor on ${this}`)
    const tensor = new Tensor<S, D>(handle!, shape, dtype)
    if (requiresGrad) tensor.requiresGrad = true
    return tensor
  }

  /**
   * Create a tensor with random normal values on this device
   *
   * @example
   * ```ts
   * const cuda = device.cuda(0)
   * const weights = cuda.randn([784, 128])
   * ```
   */
  randn<S extends Shape, D extends DType<string> = DType<'float32'>>(
    shape: S,
    dtype: D = DTypeConstants.float32 as D,
    requiresGrad = false,
  ): Tensor<S, D> {
    validateShapeArray(shape)
    const lib = getLib()
    const shapeArray = new BigInt64Array(shape.map((dim) => BigInt(dim)))

    const handle = withError((err) =>
      lib.ts_tensor_randn(shapeArray.buffer, shape.length, dtype.value, this.deviceTypeValue, this.index, err),
    )

    checkNull(handle, `Failed to create randn tensor on ${this}`)
    const tensor = new Tensor<S, D>(handle!, shape, dtype)
    if (requiresGrad) tensor.requiresGrad = true
    return tensor
  }

  /**
   * Create an uninitialized tensor on this device
   *
   * @example
   * ```ts
   * const cuda = device.cuda(0)
   * const buffer = cuda.empty([1000, 1000])
   * ```
   */
  empty<S extends Shape, D extends DType<string> = DType<'float32'>>(
    shape: S,
    dtype: D = DTypeConstants.float32 as D,
    requiresGrad = false,
  ): Tensor<S, D> {
    validateShapeArray(shape)
    const lib = getLib()
    const shapeArray = new BigInt64Array(shape.map((dim) => BigInt(dim)))

    const handle = withError((err) =>
      lib.ts_tensor_empty(shapeArray.buffer, shape.length, dtype.value, this.deviceTypeValue, this.index, err),
    )

    checkNull(handle, `Failed to create empty tensor on ${this}`)
    const tensor = new Tensor<S, D>(handle!, shape, dtype)
    if (requiresGrad) tensor.requiresGrad = true
    return tensor
  }

  /**
   * Create a tensor from array data on this device
   *
   * Note: Data is first created on CPU then transferred to device.
   * For large datasets, prefer loading directly to device via Data.pipeline().
   *
   * @example
   * ```ts
   * const cuda = device.cuda(0)
   * const x = cuda.tensor([1, 2, 3, 4], [2, 2])
   * ```
   */
  tensor<S extends Shape, D extends DType<string> = DType<'float32'>>(
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

    const handle = withError((err) =>
      lib.ts_tensor_from_buffer(
        typedData.buffer,
        shapeArray.buffer,
        shape.length,
        dtype.value,
        this.deviceTypeValue,
        this.index,
        err,
      ),
    )

    checkNull(handle, `Failed to create tensor from array on ${this}`)
    const tensor = new Tensor<S, D>(handle!, shape, dtype)
    if (requiresGrad) tensor.requiresGrad = true
    return tensor
  }

  /** String representation */
  toString(): string {
    return this.type === 'cpu' ? 'cpu' : `${this.type}:${this.index}`
  }

  /** Check if two device contexts refer to the same device */
  equals(other: DeviceContext): boolean {
    return this.type === other.type && this.index === other.index
  }
}
