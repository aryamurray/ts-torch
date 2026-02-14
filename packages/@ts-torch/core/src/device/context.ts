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
import { checkNull } from '../ffi/error.js'
import { shapeCache } from '../ffi/buffer-pool.js'
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
 *
 * @template Dev - The device type ('cpu' | 'cuda' | 'mps')
 */
export class DeviceContext<Dev extends DeviceTypeName = DeviceTypeName> {
  /** Device type: 'cpu', 'cuda', or 'mps' */
  readonly type: Dev

  /** Device index (e.g., GPU 0, GPU 1) */
  readonly index: number

  private constructor(type: Dev, index: number) {
    this.type = type
    this.index = index
  }

  /** Create a CPU device context */
  static cpu(): DeviceContext<'cpu'> {
    return new DeviceContext('cpu', 0)
  }

  /** Create a CUDA device context */
  static cuda(index = 0): DeviceContext<'cuda'> {
    return new DeviceContext('cuda', index)
  }

  /** Create an MPS device context (Apple Silicon) */
  static mps(): DeviceContext<'mps'> {
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
   * const x = cuda.zeros([2, 3]) // Tensor<[2,3], float32, 'cuda'>
   * ```
   */
  zeros<S extends Shape, D extends DType<string> = DType<'float32'>>(
    shape: S,
    dtype: D = DTypeConstants.float32 as D,
    requiresGrad = false,
  ): Tensor<S, D, Dev> {
    validateShapeArray(shape)
    const lib = getLib()
    const shapeBuffer = shapeCache.fillShape(shape)

    try {
      const handle = lib.ts_tensor_zeros(shapeBuffer, dtype.value, this.deviceTypeValue, this.index)

      checkNull(handle, `Failed to create zeros tensor on ${this}`)
      const tensor = new Tensor<S, D, Dev>(handle!, shape, dtype, this.type)
      if (requiresGrad) tensor.requiresGrad = true
      return tensor
    } finally {
      shapeCache.release(shapeBuffer)
    }
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
  ): Tensor<S, D, Dev> {
    validateShapeArray(shape)
    const lib = getLib()
    const shapeBuffer = shapeCache.fillShape(shape)

    try {
      const handle = lib.ts_tensor_ones(shapeBuffer, dtype.value, this.deviceTypeValue, this.index)

      checkNull(handle, `Failed to create ones tensor on ${this}`)
      const tensor = new Tensor<S, D, Dev>(handle!, shape, dtype, this.type)
      if (requiresGrad) tensor.requiresGrad = true
      return tensor
    } finally {
      shapeCache.release(shapeBuffer)
    }
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
  ): Tensor<S, D, Dev> {
    validateShapeArray(shape)
    const lib = getLib()
    const shapeBuffer = shapeCache.fillShape(shape)

    try {
      const handle = lib.ts_tensor_randn(shapeBuffer, dtype.value, this.deviceTypeValue, this.index)

      checkNull(handle, `Failed to create randn tensor on ${this}`)
      const tensor = new Tensor<S, D, Dev>(handle!, shape, dtype, this.type)
      if (requiresGrad) tensor.requiresGrad = true
      return tensor
    } finally {
      shapeCache.release(shapeBuffer)
    }
  }

  /**
   * Create a tensor with random uniform values on this device
   *
   * @example
   * ```ts
   * const cuda = device.cuda(0)
   * const noise = cuda.rand([256, 256])
   * ```
   */
  rand<S extends Shape, D extends DType<string> = DType<'float32'>>(
    shape: S,
    dtype: D = DTypeConstants.float32 as D,
    requiresGrad = false,
  ): Tensor<S, D, Dev> {
    validateShapeArray(shape)
    const lib = getLib()
    const shapeBuffer = shapeCache.fillShape(shape)

    try {
      const handle = lib.ts_tensor_rand(shapeBuffer, dtype.value, this.deviceTypeValue, this.index)

      checkNull(handle, `Failed to create rand tensor on ${this}`)
      const tensor = new Tensor<S, D, Dev>(handle!, shape, dtype, this.type)
      if (requiresGrad) tensor.requiresGrad = true
      return tensor
    } finally {
      shapeCache.release(shapeBuffer)
    }
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
  ): Tensor<S, D, Dev> {
    validateShapeArray(shape)
    const lib = getLib()
    const shapeBuffer = shapeCache.fillShape(shape)

    try {
      const handle = lib.ts_tensor_empty(shapeBuffer, dtype.value, this.deviceTypeValue, this.index)

      checkNull(handle, `Failed to create empty tensor on ${this}`)
      const tensor = new Tensor<S, D, Dev>(handle!, shape, dtype, this.type)
      if (requiresGrad) tensor.requiresGrad = true
      return tensor
    } finally {
      shapeCache.release(shapeBuffer)
    }
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
    data: number[] | boolean[] | Float32Array | Float64Array | Int32Array | BigInt64Array | Uint8Array,
    shape: S,
    dtype?: D,
    requiresGrad = false,
  ): Tensor<S, D, Dev> {
    // Auto-detect boolean arrays and set dtype to bool
    if (Array.isArray(data) && data.length > 0 && typeof data[0] === 'boolean') {
      dtype = DTypeConstants.bool as D
    }
    // Default to float32 if not specified
    if (dtype === undefined) {
      dtype = DTypeConstants.float32 as D
    }
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
    let typedData: Float32Array | Float64Array | Int32Array | BigInt64Array | Uint8Array
    if (Array.isArray(data)) {
      switch (dtype.name) {
        case 'float32':
          typedData = new Float32Array(data as number[])
          break
        case 'float64':
          typedData = new Float64Array(data as number[])
          break
        case 'int32':
          typedData = new Int32Array(data as number[])
          break
        case 'int64':
          typedData = new BigInt64Array((data as number[]).map((x) => BigInt(x)))
          break
        case 'bool':
          typedData = new Uint8Array((data as boolean[]).map((x) => (x ? 1 : 0)))
          break
        default:
          throw new Error(`Unsupported dtype: ${dtype.name}`)
      }
    } else {
      typedData = data as Float32Array | Float64Array | Int32Array | BigInt64Array | Uint8Array
    }

    const shapeBuffer = shapeCache.fillShape(shape)

    try {
      const handle = lib.ts_tensor_from_buffer(
        typedData.buffer,
        shapeBuffer,
        dtype.value,
        this.deviceTypeValue,
        this.index,
      )

      checkNull(handle, `Failed to create tensor from array on ${this}`)
      const tensor = new Tensor<S, D, Dev>(handle!, shape, dtype, this.type)
      if (requiresGrad) tensor.requiresGrad = true
      return tensor
    } finally {
      shapeCache.release(shapeBuffer)
    }
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
