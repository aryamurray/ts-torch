/**
 * Core Tensor class for ts-torch
 *
 * Wraps native tensor handles and provides type-safe operations with
 * compile-time shape checking.
 */

import type { Shape } from '../types/shape.js'
import type { DType } from '../types/dtype.js'
import type {
  MatMulShape,
  TransposeShape,
  DeviceType,
  SqueezeShape,
  UnsqueezeShape,
  FlattenShape,
  PermuteShape,
} from '../types/tensor.js'
import { getLib } from '../ffi/loader.js'
import { checkNull, TorchError, ErrorCode, type Pointer } from '../ffi/error.js'
import { getDTypeFromValue } from '../types/dtype.js'
import { escapeTensor as scopeEscapeTensor, inScope, registerTensor } from '../memory/scope.js'
import {
  validateMatmulShapes,
  validateDimension,
  validateReshape,
  validateScalar,
  validateNonZero,
  validateProbability,
  validatePositive,
  validateRange,
  validatePoolingParams,
  validatePositiveInt,
} from '../validation/index.js'

/**
 * Core Tensor class representing a multi-dimensional array with type-safe operations
 *
 * @template S - Shape type as readonly tuple of dimensions
 * @template D - Data type
 * @template Dev - Device type ('cpu' | 'cuda' | 'mps')
 *
 * @example
 * ```ts
 * const cpu = device.cpu()
 * const t = cpu.zeros([2, 3]); // Tensor<[2,3], float32, 'cpu'>
 * const gpuT = t.cuda(); // Tensor<[2,3], float32, 'cuda'>
 * ```
 */
export class Tensor<S extends Shape = Shape, D extends DType<string> = DType<'float32'>, Dev extends DeviceType = 'cpu'> {
  /**
   * Native tensor handle (opaque pointer)
   * @internal
   */
  readonly _handle: Pointer

  /**
   * Tensor shape - readonly tuple of dimensions
   */
  readonly shape: S

  /**
   * Data type of tensor elements
   */
  readonly dtype: D

  /**
   * Device where tensor is stored
   */
  readonly device: Dev

  /**
   * Internal flag tracking if tensor has been freed
   * @internal
   */
  private _freed = false

  /**
   * Internal flag tracking if tensor escaped from scope
   * @internal
   */
  private _escaped = false

  /**
   * Cached gradient tensor
   * @internal
   */
  private _gradCache: Tensor<S, D> | null | undefined = undefined

  /**
   * Create a new Tensor instance
   *
   * @param handle - Native tensor handle
   * @param shape - Tensor shape
   * @param dtype - Data type
   * @param device - Device type
   *
   * @internal
   * Use factory functions (zeros, ones, etc.) instead of calling constructor directly
   */
  constructor(handle: Pointer, shape: S, dtype: D, device: Dev = 'cpu' as Dev) {
    this._handle = handle
    this.shape = shape
    this.dtype = dtype
    this.device = device

    // Register with current scope if exists
    this._registerWithScope()
  }

  /**
   * Register tensor with current memory scope for automatic cleanup
   * @internal
   */
  private _registerWithScope(): void {
    // Register with current scope if one exists
    // The scope module will track this tensor and free it when the scope exits
    // (unless the tensor is escaped via .escape())
    registerTensor(this as unknown as { handle: Pointer; escaped: boolean; markEscaped(): void })
  }

  /**
   * Check if tensor is still valid
   * @internal
   */
  private _checkValid(): void {
    if (this._freed) {
      throw new TorchError(ErrorCode.SCOPE_ERROR, 'Tensor has been freed and can no longer be used')
    }
  }

  // ==================== Properties ====================

  /**
   * Get number of dimensions (rank)
   *
   * @example
   * ```ts
   * const t = zeros([2, 3, 4], DType.float32);
   * console.log(t.ndim); // 3
   * ```
   */
  get ndim(): number {
    return this.shape.length
  }

  /**
   * Get total number of elements
   *
   * @example
   * ```ts
   * const t = zeros([2, 3, 4], DType.float32);
   * console.log(t.numel); // 24
   * ```
   */
  get numel(): number {
    return this.shape.reduce((acc, dim) => acc * dim, 1)
  }

  /**
   * Check if tensor requires gradient tracking
   *
   * @example
   * ```ts
   * const t = zeros([2, 3], DType.float32);
   * console.log(t.requiresGrad); // false
   * t.requiresGrad = true;
   * console.log(t.requiresGrad); // true
   * ```
   */
  get requiresGrad(): boolean {
    this._checkValid()
    const lib = getLib()

    // Napi wrapper throws on error, returns boolean directly
    return lib.ts_tensor_requires_grad(this._handle)
  }

  /**
   * Enable or disable gradient tracking
   */
  set requiresGrad(value: boolean) {
    this._checkValid()
    const lib = getLib()

    // Napi wrapper throws on error
    lib.ts_tensor_set_requires_grad(this._handle, value)

    // Free and clear gradient cache when requiresGrad changes
    this._clearGradCache()
  }

  // ==================== Memory Management ====================

  /**
   * Escape tensor from current memory scope
   *
   * Prevents automatic cleanup when scope ends.
   * You are responsible for calling free() on escaped tensors.
   *
   * @returns this for chaining
   *
   * @example
   * ```ts
   * function createTensor() {
   *   const t = zeros([2, 3], DType.float32);
   *   return t.escape(); // Won't be freed when function returns
   * }
   * ```
   */
  escape(): this {
    this._checkValid()

    if (!this._escaped && inScope()) {
      // Use the scope module's escapeTensor which has access to the current scope handle
      scopeEscapeTensor(this as unknown as { handle: Pointer; escaped: boolean; markEscaped(): void })
    }

    return this
  }

  /**
   * Get native handle for FFI operations
   * @internal
   */
  get handle(): Pointer {
    return this._handle
  }

  /**
   * Check if tensor has been escaped from scope
   */
  get escaped(): boolean {
    return this._escaped
  }

  /**
   * Mark tensor as escaped (called by scope module)
   * @internal
   */
  markEscaped(): void {
    this._escaped = true
  }

  /**
   * Manually free tensor memory
   *
   * After calling free(), tensor can no longer be used.
   * Only needed for escaped tensors or manual memory management.
   *
   * @example
   * ```ts
   * const t = zeros([2, 3], DType.float32);
   * t.escape();
   * // ... use tensor ...
   * t.free(); // Cleanup when done
   * ```
   */
  free(): void {
    if (this._freed) {
      return // Already freed
    }

    // Free cached gradient tensor first to prevent memory leaks
    this._clearGradCache()

    const lib = getLib()
    lib.ts_tensor_delete(this._handle)
    this._freed = true
  }

  /**
   * Clear the gradient cache
   * @internal
   *
   * Note: We don't free the gradient tensor because its underlying data is
   * managed by PyTorch's autograd. The ts_Tensor wrapper will be cleaned up
   * by the garbage collector, and its destructor will just decrement the
   * reference count on the underlying torch::Tensor.
   */
  private _clearGradCache(): void {
    // Just drop our reference to the cached gradient
    // Don't call free() - the gradient tensor's underlying data is managed by autograd
    // and may still be needed by the computation graph
    this._gradCache = undefined
  }

  /**
   * Create a deep copy of the tensor
   *
   * @returns New tensor with copied data
   *
   * @example
   * ```ts
   * const t1 = zeros([2, 3], DType.float32);
   * const t2 = t1.clone(); // Independent copy
   * ```
   */
  clone(): Tensor<S, D> {
    this._checkValid()
    const lib = getLib()

    const handle = lib.ts_tensor_clone(this._handle)

    checkNull(handle, 'Failed to clone tensor')

    return new Tensor<S, D>(handle!, this.shape, this.dtype)
  }

  /**
   * Create a detached copy (no gradient tracking)
   *
   * @returns New tensor without gradient tracking
   *
   * @example
   * ```ts
   * const t1 = zeros([2, 3], DType.float32, true);
   * const t2 = t1.detach(); // No gradients
   * ```
   */
  detach(): Tensor<S, D> {
    this._checkValid()
    const lib = getLib()

    const handle = lib.ts_tensor_detach(this._handle)

    checkNull(handle, 'Failed to detach tensor')

    return new Tensor<S, D>(handle!, this.shape, this.dtype)
  }

  // ==================== Data Access ====================

  /**
   * Copy tensor data to TypedArray
   *
   * @returns TypedArray containing tensor data
   *
   * @example
   * ```ts
   * const t = fromArray([[1, 2], [3, 4]], [2, 2], DType.float32);
   * const data = t.toArray(); // Float32Array([1, 2, 3, 4])
   * ```
   */
  toArray() {
    this._checkValid()
    const lib = getLib()

    let buffer: ArrayBuffer
    let result: Float32Array | Float64Array | Int32Array | BigInt64Array | Uint8Array

    switch (this.dtype.name) {
      case 'float32':
        result = new Float32Array(this.numel)
        buffer = result.buffer as ArrayBuffer
        break

      case 'float64':
        result = new Float64Array(this.numel)
        buffer = result.buffer as ArrayBuffer
        break

      case 'int32':
        result = new Int32Array(this.numel)
        buffer = result.buffer as ArrayBuffer
        break

      case 'int64':
        result = new BigInt64Array(this.numel)
        buffer = result.buffer as ArrayBuffer
        break

      case 'bool':
        result = new Uint8Array(this.numel)
        buffer = result.buffer as ArrayBuffer
        break

      default:
        throw new Error(`Unsupported dtype: ${this.dtype.name}`)
    }

    // Copy data from native memory
    // Note: Napi wrapper handles buffer size automatically via ArrayBuffer.byteLength
    lib.ts_tensor_copy_to_buffer(this._handle, buffer)

    return result
  }

  /**
   * Get scalar value (for 0-dimensional tensors)
   *
   * @returns Scalar number value
   * @throws Error if tensor is not scalar
   *
   * @example
   * ```ts
   * const t = fromArray([42], [1], DType.float32);
   * const value = t.item(); // 42
   * ```
   */
  item(): number {
    if (this.numel !== 1) {
      throw new Error(`item() only works on scalar tensors (numel=1), got numel=${this.numel}`)
    }

    const data = this.toArray()
    return Number(data[0])
  }

  // ==================== Element-wise Operations ====================

  /**
   * Element-wise addition
   *
   * @param other - Tensor to add (must have same shape)
   * @param options - Optional: { out: preallocated tensor to write result into }
   * @returns New tensor with result (or the `out` tensor if provided)
   *
   * @example
   * ```ts
   * const a = ones([2, 3], DType.float32);
   * const b = ones([2, 3], DType.float32);
   * const c = a.add(b); // [[2, 2, 2], [2, 2, 2]]
   *
   * // With pre-allocated output (avoids allocation):
   * const out = empty([2, 3], DType.float32);
   * a.add(b, { out }); // Writes to out, returns out
   * ```
   */
  add(other: Tensor<S, D>, options?: { out?: Tensor<S, D> }): Tensor<S, D> {
    this._checkValid()
    other._checkValid()
    const lib = getLib()

    if (options?.out) {
      options.out._checkValid()
      lib.ts_tensor_add_out(this._handle, other._handle, options.out!._handle)
      return options.out
    }

    const handle = lib.ts_tensor_add(this._handle, other._handle)

    checkNull(handle, 'Failed to add tensors')

    return new Tensor<S, D>(handle!, this.shape, this.dtype)
  }

  /**
   * Element-wise subtraction
   *
   * @param other - Tensor to subtract (must have same shape)
   * @param options - Optional: { out: preallocated tensor to write result into }
   * @returns New tensor with result (or the `out` tensor if provided)
   *
   * @example
   * ```ts
   * const a = ones([2, 3], DType.float32);
   * const b = ones([2, 3], DType.float32);
   * const c = a.sub(b); // [[0, 0, 0], [0, 0, 0]]
   * ```
   */
  sub(other: Tensor<S, D>, options?: { out?: Tensor<S, D> }): Tensor<S, D> {
    this._checkValid()
    other._checkValid()
    const lib = getLib()

    if (options?.out) {
      options.out._checkValid()
      lib.ts_tensor_sub_out(this._handle, other._handle, options.out!._handle)
      return options.out
    }

    const handle = lib.ts_tensor_sub(this._handle, other._handle)

    checkNull(handle, 'Failed to subtract tensors')

    return new Tensor<S, D>(handle!, this.shape, this.dtype)
  }

  /**
   * Element-wise multiplication
   *
   * @param other - Tensor to multiply (must have same shape)
   * @param options - Optional: { out: preallocated tensor to write result into }
   * @returns New tensor with result (or the `out` tensor if provided)
   *
   * @example
   * ```ts
   * const a = fromArray([[1, 2], [3, 4]], [2, 2], DType.float32);
   * const b = fromArray([[2, 2], [2, 2]], [2, 2], DType.float32);
   * const c = a.mul(b); // [[2, 4], [6, 8]]
   * ```
   */
  mul(other: Tensor<S, D>, options?: { out?: Tensor<S, D> }): Tensor<S, D> {
    this._checkValid()
    other._checkValid()
    const lib = getLib()

    if (options?.out) {
      options.out._checkValid()
      lib.ts_tensor_mul_out(this._handle, other._handle, options.out!._handle)
      return options.out
    }

    const handle = lib.ts_tensor_mul(this._handle, other._handle)

    checkNull(handle, 'Failed to multiply tensors')

    return new Tensor<S, D>(handle!, this.shape, this.dtype)
  }

  /**
   * Element-wise division
   *
   * @param other - Tensor to divide by (must have same shape)
   * @param options - Optional: { out: preallocated tensor to write result into }
   * @returns New tensor with result (or the `out` tensor if provided)
   *
   * @example
   * ```ts
   * const a = fromArray([[2, 4], [6, 8]], [2, 2], DType.float32);
   * const b = fromArray([[2, 2], [2, 2]], [2, 2], DType.float32);
   * const c = a.div(b); // [[1, 2], [3, 4]]
   * ```
   */
  div(other: Tensor<S, D>, options?: { out?: Tensor<S, D> }): Tensor<S, D> {
    this._checkValid()
    other._checkValid()
    const lib = getLib()

    if (options?.out) {
      options.out._checkValid()
      lib.ts_tensor_div_out(this._handle, other._handle, options.out!._handle)
      return options.out
    }

    const handle = lib.ts_tensor_div(this._handle, other._handle)

    checkNull(handle, 'Failed to divide tensors')

    return new Tensor<S, D>(handle!, this.shape, this.dtype)
  }

  // ==================== Matrix Operations ====================

  /**
   * Matrix multiplication with type-safe shape inference
   *
   * @template S2 - Shape of other tensor
   * @param other - Tensor to multiply with
   * @param options - Optional: { out: preallocated tensor to write result into }
   * @returns New tensor with result shape computed at compile time
   *
   * @example
   * ```ts
   * const a = zeros([2, 3], DType.float32);
   * const b = zeros([3, 4], DType.float32);
   * const c = a.matmul(b); // Type: Tensor<[2, 4], DType<"float32">>
   *
   * // With pre-allocated output:
   * const out = empty([2, 4], DType.float32);
   * a.matmul(b, { out }); // Writes to out
   * ```
   */
  matmul<S2 extends Shape>(other: Tensor<S2, D>, options?: { out?: Tensor<MatMulShape<S, S2>, D> }): Tensor<MatMulShape<S, S2>, D> {
    this._checkValid()
    other._checkValid()
    validateMatmulShapes(this.shape, other.shape)
    const lib = getLib()

    if (options?.out) {
      options.out._checkValid()
      lib.ts_tensor_matmul_out(this._handle, other._handle, options.out!._handle)
      return options.out
    }

    const handle = lib.ts_tensor_matmul(this._handle, other._handle)

    checkNull(handle, 'Failed to perform matrix multiplication')

    // Compute result shape
    // This is simplified - real implementation would compute proper shape
    const resultShape = this._computeMatMulShape(this.shape, other.shape) as MatMulShape<S, S2>

    return new Tensor<MatMulShape<S, S2>, D>(handle!, resultShape, this.dtype)
  }

  /**
   * Compute matrix multiplication result shape
   * @internal
   */
  private _computeMatMulShape(s1: readonly number[], s2: readonly number[]): readonly number[] {
    // Extract batch and matrix dimensions
    const batch1 = s1.slice(0, -2)
    const m = s1[s1.length - 2]!

    const batch2 = s2.slice(0, -2)
    const n = s2[s2.length - 1]!

    // Broadcast batch dimensions
    const batchResult = this._broadcastShapes(batch1, batch2)

    return [...batchResult, m, n]
  }

  /**
   * Simple shape broadcasting helper
   * @internal
   */
  private _broadcastShapes(s1: readonly number[], s2: readonly number[]): readonly number[] {
    const maxLen = Math.max(s1.length, s2.length)
    const result: number[] = []

    for (let i = 0; i < maxLen; i++) {
      const dim1 = s1[s1.length - 1 - i] ?? 1
      const dim2 = s2[s2.length - 1 - i] ?? 1

      if (dim1 === dim2 || dim1 === 1 || dim2 === 1) {
        result.unshift(Math.max(dim1, dim2))
      } else {
        throw new Error(`Cannot broadcast shapes: ${s1} and ${s2}`)
      }
    }

    return result
  }

  /**
   * Normalize a dimension index (supports negative dims)
   * @internal
   */
  private _normalizeDim(dim: number, ndim: number, parameter: string = 'dim'): number {
    validateDimension(dim, ndim, parameter)
    return dim < 0 ? dim + ndim : dim
  }

  /**
   * Transpose two dimensions
   *
   * @template D0 - First dimension index
   * @template D1 - Second dimension index
   * @param dim0 - First dimension to swap
   * @param dim1 - Second dimension to swap
   * @returns New tensor with transposed dimensions
   *
   * @example
   * ```ts
   * const a = zeros([2, 3, 4], DType.float32);
   * const b = a.transpose(0, 2); // Type: Tensor<[4, 3, 2], DType<"float32">>
   * ```
   */
  transpose<D0 extends number, D1 extends number>(dim0: D0, dim1: D1): Tensor<TransposeShape<S, D0, D1>, D> {
    this._checkValid()
    validateDimension(dim0, this.ndim, 'dim0')
    validateDimension(dim1, this.ndim, 'dim1')
    const lib = getLib()

    const handle = lib.ts_tensor_transpose(this._handle, dim0, dim1)

    checkNull(handle, 'Failed to transpose tensor')

    // Compute result shape
    const resultShape = [...this.shape]
    ;[resultShape[dim0], resultShape[dim1]] = [resultShape[dim1]!, resultShape[dim0]!]

    return new Tensor<TransposeShape<S, D0, D1>, D>(
      handle!,
      resultShape as unknown as TransposeShape<S, D0, D1>,
      this.dtype,
    )
  }

  /**
   * Reshape tensor to new shape
   *
   * @template NewS - New shape type
   * @param shape - New shape (must preserve element count)
   * @returns New tensor with new shape
   *
   * @example
   * ```ts
   * const a = zeros([2, 3], DType.float32);
   * const b = a.reshape([3, 2] as const); // Type: Tensor<[3, 2], DType<"float32">>
   * ```
   */
  reshape<NewS extends Shape>(shape: NewS): Tensor<NewS, D> {
    this._checkValid()
    validateReshape(this.shape, shape)
    const lib = getLib()

    // Convert shape to BigInt64Array for FFI
    const shapeArray = new BigInt64Array(shape.map((dim) => BigInt(dim)))

    const handle = lib.ts_tensor_reshape(this._handle, shapeArray)

    checkNull(handle, 'Failed to reshape tensor')

    return new Tensor<NewS, D>(handle!, shape, this.dtype)
  }

  // ==================== Shape Operations ====================

  /**
   * Return a view with dimensions of size 1 removed
   */
  squeeze<Dim extends number | undefined = undefined>(dim?: Dim): Tensor<SqueezeShape<S, Dim> & Shape, D> {
    this._checkValid()
    const shape = [...this.shape] as number[]
    if (shape.length === 0) {
      return this as unknown as Tensor<SqueezeShape<S, Dim> & Shape, D>
    }

    if (dim === undefined) {
      const newShape = shape.filter((size) => size !== 1)
      if (newShape.length === shape.length) {
        return this as unknown as Tensor<SqueezeShape<S, Dim> & Shape, D>
      }
      return this.reshape((newShape.length === 0 ? ([] as number[]) : newShape) as unknown as Shape) as Tensor<
        SqueezeShape<S, Dim> & Shape,
        D
      >
    }

    const normalized = this._normalizeDim(dim, shape.length, 'dim')
    if (shape[normalized] !== 1) {
      return this as unknown as Tensor<SqueezeShape<S, Dim> & Shape, D>
    }
    shape.splice(normalized, 1)
    return this.reshape((shape.length === 0 ? ([] as number[]) : shape) as unknown as Shape) as Tensor<
      SqueezeShape<S, Dim> & Shape,
      D
    >
  }

  /**
   * Return a view with a dimension of size 1 inserted
   */
  unsqueeze<Dim extends number>(dim: Dim): Tensor<UnsqueezeShape<S, Dim>, D> {
    this._checkValid()
    const ndim = this.ndim
    const normalized = dim < 0 ? dim + ndim + 1 : dim
    if (!Number.isInteger(dim) || normalized < 0 || normalized > ndim) {
      throw new Error(`dim must be in range [${-(ndim + 1)}, ${ndim}] for ${ndim}D tensor`)
    }

    const shape = [...this.shape] as number[]
    shape.splice(normalized, 0, 1)
    return this.reshape(shape as unknown as Shape) as Tensor<UnsqueezeShape<S, Dim>, D>
  }

  /**
   * Flatten dimensions from startDim to endDim (inclusive)
   */
  flatten<Start extends number = 0, End extends number = number>(
    startDim: Start = 0 as Start,
    endDim: End = -1 as End,
  ): Tensor<FlattenShape<S, Start, End>, D> {
    this._checkValid()
    const ndim = this.ndim
    let start = startDim < 0 ? (startDim as number) + ndim : (startDim as number)
    let end = endDim < 0 ? (endDim as number) + ndim : (endDim as number)

    if (!Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end < 0 || start >= ndim || end >= ndim) {
      throw new Error(`startDim/endDim must be valid dimensions for ${ndim}D tensor`)
    }
    if (end < start) {
      throw new Error(`endDim (${endDim}) must be >= startDim (${startDim})`)
    }

    const shape = [...this.shape] as number[]
    const prefix = shape.slice(0, start)
    const middle = shape.slice(start, end + 1)
    const suffix = shape.slice(end + 1)
    const flattened = middle.reduce((acc, dim) => acc * dim, 1)
    const newShape = [...prefix, flattened, ...suffix]
    return this.reshape(newShape as unknown as Shape) as Tensor<FlattenShape<S, Start, End>, D>
  }

  /**
   * Alias for reshape
   */
  view<NewS extends Shape>(shape: NewS): Tensor<NewS, D> {
    return this.reshape(shape)
  }

  /**
   * Split tensor along a dimension into chunks
   */
  split(splitSizeOrSections: number | number[], dim: number = 0): Tensor<Shape, D>[] {
    this._checkValid()
    const normalizedDim = this._normalizeDim(dim, this.ndim, 'dim')
    const dimSize = (this.shape as readonly number[])[normalizedDim] ?? 0

    if (Array.isArray(splitSizeOrSections)) {
      const sizes = splitSizeOrSections
      let total = 0
      for (const size of sizes) {
        validatePositiveInt(size, 'splitSize')
        total += size
      }
      if (total !== dimSize) {
        throw new Error(`Sum of split sizes (${total}) does not match dimension size (${dimSize})`)
      }
      const outputs: Tensor<Shape, D>[] = []
      let offset = 0
      for (const size of sizes) {
        outputs.push(this.narrow(normalizedDim, offset, size))
        offset += size
      }
      return outputs
    }

    validatePositiveInt(splitSizeOrSections, 'splitSize')
    const splitSize = splitSizeOrSections
    const outputs: Tensor<Shape, D>[] = []
    let offset = 0
    while (offset < dimSize) {
      const length = Math.min(splitSize, dimSize - offset)
      outputs.push(this.narrow(normalizedDim, offset, length))
      offset += length
    }
    return outputs
  }

  /**
   * Permute tensor dimensions
   */
  permute<Perm extends readonly number[]>(dims: Perm): Tensor<PermuteShape<S, Perm>, D> {
    this._checkValid()
    const ndim = this.ndim
    if (dims.length !== ndim) {
      throw new Error(`permute expects ${ndim} dimensions, got ${dims.length}`)
    }

    const normalized = dims.map((dim) => (dim < 0 ? dim + ndim : dim))
    const seen = new Set<number>()
    for (let i = 0; i < normalized.length; i++) {
      const dim = normalized[i]!
      if (!Number.isInteger(dim) || dim < 0 || dim >= ndim) {
        throw new Error(`Invalid permute dimension: ${dims[i]}`)
      }
      if (seen.has(dim)) {
        throw new Error(`Duplicate dimension in permute: ${dim}`)
      }
      seen.add(dim)
    }

    let result: Tensor<Shape, D> = this as unknown as Tensor<Shape, D>
    const order = Array.from({ length: ndim }, (_, i) => i)
    for (let i = 0; i < ndim; i++) {
      const targetDim = normalized[i]!
      const currentIndex = order.indexOf(targetDim)
      if (currentIndex !== i) {
        result = result.transpose(i, currentIndex) as Tensor<Shape, D>
        ;[order[i], order[currentIndex]] = [order[currentIndex]!, order[i]!]
      }
    }

    return result as Tensor<PermuteShape<S, Perm>, D>
  }

  // ==================== Reductions ====================

  /**
   * Sum all elements to scalar
   *
   * @returns Scalar tensor with sum of all elements
   *
   * @remarks
   * The result dtype may differ from the input dtype (e.g., bool sum returns int64).
   * This method queries the actual dtype from the native tensor.
   *
   * @example
   * ```ts
   * const a = fromArray([[1, 2], [3, 4]], [2, 2], DType.float32);
   * const sum = a.sum(); // Tensor(10)
   * ```
   */
  sum(): Tensor<readonly [], DType<string>> {
    this._checkValid()
    const lib = getLib()

    const handle = lib.ts_tensor_sum(this._handle)

    checkNull(handle, 'Failed to compute sum')

    // Query actual dtype from result tensor (may differ, e.g., bool sum -> int64)
    const dtypeValue = lib.ts_tensor_dtype(handle!) as number
    const resultDtype = getDTypeFromValue(dtypeValue)

    return new Tensor<readonly [], DType<string>>(handle!, [] as const, resultDtype)
  }

  /**
   * Mean of all elements to scalar
   *
   * @returns Scalar tensor with mean of all elements
   *
   * @example
   * ```ts
   * const a = fromArray([[1, 2], [3, 4]], [2, 2], DType.float32);
   * const mean = a.mean(); // Tensor(2.5)
   * ```
   */
mean(): Tensor<readonly [], D> {
    this._checkValid()
    const lib = getLib()

    const handle = lib.ts_tensor_mean(this._handle)

    checkNull(handle, 'Failed to compute mean')

    return new Tensor<readonly [], D>(handle!, [] as const, this.dtype)
  }

  /**
   * Sum along a specific dimension
   *
   * @param dim - Dimension to reduce
   * @param keepdim - Whether to keep the reduced dimension (default: false)
   * @returns Tensor with sum along the specified dimension
   *
   * @example
   * ```ts
   * const a = fromArray([[1, 2, 3], [4, 5, 6]], [2, 3], DType.float32);
   * const b = a.sumDim(1); // [6, 15] - sum along columns
   * const c = a.sumDim(0); // [5, 7, 9] - sum along rows
   * const d = a.sumDim(1, true); // [[6], [15]] - keep dimension
   * ```
   */
  sumDim(dim: number, keepdim: boolean = false): Tensor<Shape, DType<string>> {
    this._checkValid()
    validateDimension(dim, this.ndim, 'dim')
    const lib = getLib()

    const handle = lib.ts_tensor_sum_dim(this._handle, dim, keepdim)

    checkNull(handle, 'Failed to compute sum along dimension')

    // Compute output shape
    const newShape = [...this.shape] as number[]
    if (keepdim) {
      newShape[dim] = 1
    } else {
      newShape.splice(dim, 1)
    }

    // Query actual dtype from result tensor
    const dtypeValue = lib.ts_tensor_dtype(handle!) as number
    const resultDtype = getDTypeFromValue(dtypeValue)

    return new Tensor<Shape, DType<string>>(handle!, newShape as unknown as Shape, resultDtype)
  }

  /**
   * Mean along a specific dimension
   *
   * @param dim - Dimension to reduce
   * @param keepdim - Whether to keep the reduced dimension (default: false)
   * @returns Tensor with mean along the specified dimension
   *
   * @example
   * ```ts
   * const a = fromArray([[1, 2, 3], [4, 5, 6]], [2, 3], DType.float32);
   * const b = a.meanDim(1); // [2, 5] - mean along columns
   * const c = a.meanDim(0); // [2.5, 3.5, 4.5] - mean along rows
   * ```
   */
  meanDim(dim: number, keepdim: boolean = false): Tensor<Shape, D> {
    this._checkValid()
    validateDimension(dim, this.ndim, 'dim')
    const lib = getLib()

    const handle = lib.ts_tensor_mean_dim(this._handle, dim, keepdim)

    checkNull(handle, 'Failed to compute mean along dimension')

    // Compute output shape
    const newShape = [...this.shape] as number[]
    if (keepdim) {
      newShape[dim] = 1
    } else {
      newShape.splice(dim, 1)
    }

    return new Tensor<Shape, D>(handle!, newShape as unknown as Shape, this.dtype)
  }

  // ==================== Activations ====================

  /**
   * ReLU activation: max(0, x)
   *
   * @returns New tensor with ReLU applied element-wise
   *
   * @example
   * ```ts
   * const a = fromArray([[-1, 2], [3, -4]], [2, 2], DType.float32);
   * const b = a.relu(); // [[0, 2], [3, 0]]
   * ```
   */
  relu(): Tensor<S, D> {
    this._checkValid()
    const lib = getLib()

    const handle = lib.ts_tensor_relu(this._handle)

    checkNull(handle, 'Failed to apply ReLU')

    return new Tensor<S, D>(handle!, this.shape, this.dtype)
  }

  /**
   * Sigmoid activation: 1 / (1 + exp(-x))
   *
   * @returns New tensor with sigmoid applied element-wise
   *
   * @example
   * ```ts
   * const a = fromArray([[0, 1], [2, 3]], [2, 2], DType.float32);
   * const b = a.sigmoid(); // [[0.5, 0.73], [0.88, 0.95]]
   * ```
   */
  sigmoid(): Tensor<S, D> {
    this._checkValid()
    const lib = getLib()

    const handle = lib.ts_tensor_sigmoid(this._handle)

    checkNull(handle, 'Failed to apply sigmoid')

    return new Tensor<S, D>(handle!, this.shape, this.dtype)
  }

  /**
   * Softmax activation along dimension
   *
   * @param dim - Dimension to apply softmax
   * @returns New tensor with softmax applied
   *
   * @example
   * ```ts
   * const a = fromArray([[1, 2], [3, 4]], [2, 2], DType.float32);
   * const b = a.softmax(1); // Softmax along dim 1
   * ```
   */
  softmax(dim: number): Tensor<S, D> {
    this._checkValid()
    validateDimension(dim, this.ndim, 'dim')
    const lib = getLib()

    const handle = lib.ts_tensor_softmax(this._handle, dim)

    checkNull(handle, 'Failed to apply softmax')

    return new Tensor<S, D>(handle!, this.shape, this.dtype)
  }

  /**
   * Log-softmax activation along dimension (numerically stable)
   *
   * @param dim - Dimension to apply log-softmax
   * @returns New tensor with log-softmax applied
   *
   * @example
   * ```ts
   * const a = fromArray([[1, 2], [3, 4]], [2, 2], DType.float32);
   * const b = a.logSoftmax(1); // Log-softmax along dim 1
   * ```
   */
  logSoftmax(dim: number): Tensor<S, D> {
    this._checkValid()
    validateDimension(dim, this.ndim, 'dim')
    const lib = getLib()

    const handle = lib.ts_tensor_log_softmax(this._handle, dim)

    checkNull(handle, 'Failed to apply log_softmax')

    return new Tensor<S, D>(handle!, this.shape, this.dtype)
  }

  /**
   * Element-wise natural logarithm
   *
   * @returns New tensor with log applied element-wise
   *
   * @example
   * ```ts
   * const a = fromArray([[1, 2.718], [7.389, 20.086]], [2, 2], DType.float32);
   * const b = a.log(); // [[0, 1], [2, 3]] approximately
   * ```
   */
  log(): Tensor<S, D> {
    this._checkValid()
    const lib = getLib()

    const handle = lib.ts_tensor_log(this._handle)

    checkNull(handle, 'Failed to apply log')

    return new Tensor<S, D>(handle!, this.shape, this.dtype)
  }

  /**
   * Element-wise exponential
   *
   * @returns New tensor with exp applied element-wise
   *
   * @example
   * ```ts
   * const a = fromArray([[0, 1], [2, 3]], [2, 2], DType.float32);
   * const b = a.exp(); // [[1, 2.718], [7.389, 20.086]] approximately
   * ```
   */
  exp(): Tensor<S, D> {
    this._checkValid()
    const lib = getLib()

    const handle = lib.ts_tensor_exp(this._handle)

    checkNull(handle, 'Failed to apply exp')

    return new Tensor<S, D>(handle!, this.shape, this.dtype)
  }

  /**
   * Element-wise negation
   *
   * @returns New tensor with negation applied element-wise
   *
   * @example
   * ```ts
   * const a = fromArray([[1, -2], [3, -4]], [2, 2], DType.float32);
   * const b = a.neg(); // [[-1, 2], [-3, 4]]
   * ```
   */
  neg(): Tensor<S, D> {
    this._checkValid()
    const lib = getLib()

    const handle = lib.ts_tensor_neg(this._handle)

    checkNull(handle, 'Failed to apply neg')

    return new Tensor<S, D>(handle!, this.shape, this.dtype)
  }

  /**
   * Element-wise square root
   *
   * @returns New tensor with sqrt of each element
   *
   * @example
   * ```ts
   * const a = fromArray([[1, 4], [9, 16]], [2, 2], DType.float32);
   * const b = a.sqrt(); // [[1, 2], [3, 4]]
   * ```
   */
  sqrt(): Tensor<S, D> {
    this._checkValid()
    const lib = getLib()

    const handle = lib.ts_tensor_sqrt(this._handle)

    checkNull(handle, 'Failed to apply sqrt')

    return new Tensor<S, D>(handle!, this.shape, this.dtype)
  }

  /**
   * Element-wise tanh activation
   *
   * @returns New tensor with tanh applied to each element
   *
   * @example
   * ```ts
   * const a = fromArray([[0, 1], [-1, 2]], [2, 2], DType.float32);
   * const b = a.tanh(); // [[0, 0.7616], [-0.7616, 0.9640]]
   * ```
   */
  tanh(): Tensor<S, D> {
    this._checkValid()
    const lib = getLib()

    const handle = lib.ts_tensor_tanh(this._handle)

    checkNull(handle, 'Failed to apply tanh')

    return new Tensor<S, D>(handle!, this.shape, this.dtype)
  }

  /**
   * Element-wise minimum of two tensors
   *
   * @param other - Tensor to compare with (must be same shape)
   * @returns New tensor with element-wise minimum
   *
   * @example
   * ```ts
   * const a = fromArray([[1, 5], [3, 2]], [2, 2], DType.float32);
   * const b = fromArray([[2, 4], [1, 6]], [2, 2], DType.float32);
   * const c = a.minimum(b); // [[1, 4], [1, 2]]
   * ```
   */
  minimum(other: Tensor<S, D>): Tensor<S, D> {
    this._checkValid()
    const lib = getLib()

    const handle = lib.ts_tensor_minimum(this._handle, other._handle)

    checkNull(handle, 'Failed to compute minimum')

    return new Tensor<S, D>(handle!, this.shape, this.dtype)
  }

  /**
   * Element-wise maximum of two tensors
   *
   * @param other - Tensor to compare with (must be same shape)
   * @returns New tensor with element-wise maximum
   *
   * @example
   * ```ts
   * const a = fromArray([[1, 5], [3, 2]], [2, 2], DType.float32);
   * const b = fromArray([[2, 4], [1, 6]], [2, 2], DType.float32);
   * const c = a.maximum(b); // [[2, 5], [3, 6]]
   * ```
   */
  maximum(other: Tensor<S, D>): Tensor<S, D> {
    this._checkValid()
    const lib = getLib()

    const handle = lib.ts_tensor_maximum(this._handle, other._handle)

    checkNull(handle, 'Failed to compute maximum')

    return new Tensor<S, D>(handle!, this.shape, this.dtype)
  }

  /**
   * Clamp tensor values to a range [min, max]
   *
   * @param min - Minimum value
   * @param max - Maximum value
   * @returns New tensor with values clamped to range
   *
   * @example
   * ```ts
   * const a = fromArray([[0.1, 0.9], [1.5, -0.5]], [2, 2], DType.float32);
   * const b = a.clamp(0, 1); // [[0.1, 0.9], [1.0, 0.0]]
   * ```
   */
  clamp(min: number, max: number): Tensor<S, D> {
    this._checkValid()
    const lib = getLib()

    const handle = lib.ts_tensor_clamp(this._handle, min, max)

    checkNull(handle, 'Failed to clamp tensor')

    return new Tensor<S, D>(handle!, this.shape, this.dtype)
  }

  /**
   * Clamp tensor values to a minimum
   *
   * @param min - Minimum value
   * @returns New tensor with values clamped to minimum
   *
   * @example
   * ```ts
   * const a = fromArray([[-1, 0.5], [2, -3]], [2, 2], DType.float32);
   * const b = a.clampMin(0); // [[0, 0.5], [2, 0]]
   * ```
   */
  clampMin(min: number): Tensor<S, D> {
    this._checkValid()
    const lib = getLib()

    const handle = lib.ts_tensor_clamp_min(this._handle, min)

    checkNull(handle, 'Failed to clamp tensor to minimum')

    return new Tensor<S, D>(handle!, this.shape, this.dtype)
  }

  /**
   * Clamp tensor values to a maximum
   *
   * @param max - Maximum value
   * @returns New tensor with values clamped to maximum
   *
   * @example
   * ```ts
   * const a = fromArray([[1, 3], [2, 5]], [2, 2], DType.float32);
   * const b = a.clampMax(3); // [[1, 3], [2, 3]]
   * ```
   */
  clampMax(max: number): Tensor<S, D> {
    this._checkValid()
    const lib = getLib()

    const handle = lib.ts_tensor_clamp_max(this._handle, max)

    checkNull(handle, 'Failed to clamp tensor to maximum')

    return new Tensor<S, D>(handle!, this.shape, this.dtype)
  }

  // ==================== Scalar Operations ====================

  /**
   * Add scalar to all elements
   *
   * @param scalar - Scalar value to add
   * @returns New tensor with scalar added
   *
   * @example
   * ```ts
   * const a = fromArray([[1, 2], [3, 4]], [2, 2], DType.float32);
   * const b = a.addScalar(10); // [[11, 12], [13, 14]]
   * ```
   */
  addScalar(scalar: number): Tensor<S, D> {
    this._checkValid()
    validateScalar(scalar, 'scalar')
    const lib = getLib()

    const handle = lib.ts_tensor_add_scalar(this._handle, scalar)

    checkNull(handle, 'Failed to add scalar')

    return new Tensor<S, D>(handle!, this.shape, this.dtype)
  }

  /**
   * Subtract scalar from all elements
   *
   * @param scalar - Scalar value to subtract
   * @returns New tensor with scalar subtracted
   *
   * @example
   * ```ts
   * const a = fromArray([[10, 20], [30, 40]], [2, 2], DType.float32);
   * const b = a.subScalar(5); // [[5, 15], [25, 35]]
   * ```
   */
  subScalar(scalar: number): Tensor<S, D> {
    this._checkValid()
    validateScalar(scalar, 'scalar')
    const lib = getLib()

    const handle = lib.ts_tensor_sub_scalar(this._handle, scalar)

    checkNull(handle, 'Failed to subtract scalar')

    return new Tensor<S, D>(handle!, this.shape, this.dtype)
  }

  /**
   * Multiply all elements by scalar
   *
   * @param scalar - Scalar value to multiply by
   * @returns New tensor with scalar multiplication
   *
   * @example
   * ```ts
   * const a = fromArray([[1, 2], [3, 4]], [2, 2], DType.float32);
   * const b = a.mulScalar(2); // [[2, 4], [6, 8]]
   * ```
   */
  mulScalar(scalar: number): Tensor<S, D> {
    this._checkValid()
    validateScalar(scalar, 'scalar')
    const lib = getLib()

    const handle = lib.ts_tensor_mul_scalar(this._handle, scalar)

    checkNull(handle, 'Failed to multiply by scalar')

    return new Tensor<S, D>(handle!, this.shape, this.dtype)
  }

  /**
   * Divide all elements by scalar
   *
   * @param scalar - Scalar value to divide by
   * @returns New tensor with scalar division
   *
   * @example
   * ```ts
   * const a = fromArray([[2, 4], [6, 8]], [2, 2], DType.float32);
   * const b = a.divScalar(2); // [[1, 2], [3, 4]]
   * ```
   */
  divScalar(scalar: number): Tensor<S, D> {
    this._checkValid()
    validateNonZero(scalar, 'scalar')
    const lib = getLib()

    const handle = lib.ts_tensor_div_scalar(this._handle, scalar)

    checkNull(handle, 'Failed to divide by scalar')

    return new Tensor<S, D>(handle!, this.shape, this.dtype)
  }

  // ==================== Autograd ====================

  /**
   * Compute gradients via backpropagation
   *
   * Tensor must be scalar (numel=1) and have requires_grad=true
   *
   * @throws Error if tensor is not scalar
   *
   * @example
   * ```ts
   * const x = fromArray([2], [1], DType.float32, true);
   * const y = x.mul(x); // y = x^2
   * y.backward();
   * console.log(x.grad); // dy/dx = 2x = 4
   * ```
   */
  backward(): void {
    this._checkValid()
    const lib = getLib()

    // Napi wrapper handles errors internally
    lib.ts_tensor_backward(this._handle)
  }

  /**
   * Zero out gradients
   *
   * Resets the gradient tensor to zero. Call before each backward pass
   * during training to prevent gradient accumulation.
   *
   * @example
   * ```ts
   * const x = fromArray([2], [1], DType.float32, true);
   * x.zeroGrad(); // Clear any existing gradients
   * const y = x.mul(x);
   * y.backward();
   * ```
   */
  zeroGrad(): void {
    this._checkValid()
    const lib = getLib()

    lib.ts_tensor_zero_grad(this._handle)

    // Invalidate gradient cache since native gradient was zeroed
    this._clearGradCache()
  }

  /**
   * Get gradient tensor
   *
   * Returns null if no gradient has been computed
   *
   * @example
   * ```ts
   * const x = fromArray([2], [1], DType.float32, true);
   * const y = x.mul(x);
   * y.backward();
   * const grad = x.grad; // Tensor with gradient
   * ```
   */
  get grad(): Tensor<S, D> | null {
    this._checkValid()

    // Return cached gradient if available
    if (this._gradCache !== undefined) {
      // console.log('grad getter: returning cached', this._gradCache === null ? 'null' : 'tensor')
      return this._gradCache
    }

    const lib = getLib()

    const handle = lib.ts_tensor_grad(this._handle)
    // console.log('grad getter: native returned handle', handle)

    // Null handle means no gradient
    if (handle === null || handle === 0) {
      this._gradCache = null
      return null
    }

    const grad = new Tensor<S, D>(handle, this.shape, this.dtype)
    this._gradCache = grad
    return grad
  }

  // ==================== Device Operations ====================

  /**
   * Move tensor to device
   *
   * @param targetDevice - Target device ('cpu', 'cuda', 'mps')
   * @returns New tensor on target device with updated type
   *
   * @example
   * ```ts
   * const a = cpu.zeros([2, 3]); // Tensor<[2,3], float32, 'cpu'>
   * const b = a.to('cuda'); // Tensor<[2,3], float32, 'cuda'>
   * ```
   */
  to<TargetDev extends DeviceType>(targetDevice: TargetDev): Tensor<S, D, TargetDev> {
    this._checkValid()
    const lib = getLib()

    // Map device string to enum
    let deviceType: number
    let deviceId = 0

    switch (targetDevice) {
      case 'cpu':
        deviceType = 0
        break
      case 'cuda':
        deviceType = 1
        break
      case 'mps':
        deviceType = 2
        break
      default:
        throw new Error(`Unknown device: ${targetDevice}`)
    }

    const handle = lib.ts_tensor_to_device(this._handle, deviceType, deviceId)

    checkNull(handle, 'Failed to move tensor to device')

    return new Tensor<S, D, TargetDev>(handle!, this.shape, this.dtype, targetDevice)
  }

  /**
   * Move tensor to CPU
   *
   * @returns New tensor on CPU
   *
   * @example
   * ```ts
   * const a = cuda.zeros([2, 3]); // Tensor<[2,3], float32, 'cuda'>
   * const b = a.cpu(); // Tensor<[2,3], float32, 'cpu'>
   * ```
   */
  cpu(): Tensor<S, D, 'cpu'> {
    return this.to('cpu')
  }

  /**
   * Move tensor to CUDA device
   *
   * @param deviceIndex - CUDA device index (default: 0)
   * @returns New tensor on CUDA device
   *
   * @example
   * ```ts
   * const a = cpu.zeros([2, 3]); // Tensor<[2,3], float32, 'cpu'>
   * const b = a.cuda(); // Tensor<[2,3], float32, 'cuda'>
   * const c = a.cuda(1); // Tensor<[2,3], float32, 'cuda'>
   * ```
   */
  cuda(deviceIndex = 0): Tensor<S, D, 'cuda'> {
    this._checkValid()
    const lib = getLib()

    const handle = lib.ts_tensor_to_device(this._handle, 1, deviceIndex)

    checkNull(handle, 'Failed to move tensor to CUDA')

    return new Tensor<S, D, 'cuda'>(handle!, this.shape, this.dtype, 'cuda')
  }

  // ==================== Move Operations (Transfer + Free Source) ====================

  /**
   * Move tensor to device and FREE the source (Rust-like move semantics)
   *
   * Unlike `.to()` which copies, `.move()` transfers ownership and frees
   * the original tensor. More memory efficient for one-way transfers.
   *
   * WARNING: After calling .move(), the original tensor is INVALID.
   * Any subsequent operations on it will throw an error.
   *
   * @param targetDevice - Target device ('cpu', 'cuda', 'mps')
   * @returns New tensor on target device with updated type
   *
   * @example
   * ```ts
   * const cpuTensor = cpu.zeros([100, 100]); // Tensor<..., 'cpu'>
   * const gpuTensor = cpuTensor.move('cuda'); // Tensor<..., 'cuda'>
   * // cpuTensor is now INVALID - do not use!
   * ```
   */
  move<TargetDev extends DeviceType>(targetDevice: TargetDev): Tensor<S, D, TargetDev> {
    this._checkValid()

    // Same device = no-op, return self with updated type
    // Cast to string for comparison since Dev and TargetDev are different generic params
    if ((this.device as string) === (targetDevice as string)) {
      return this as unknown as Tensor<S, D, TargetDev>
    }

    // Create new tensor on target device
    const newTensor = this.to(targetDevice)

    // Free the source tensor
    this.free()

    return newTensor
  }

  /**
   * Move tensor to CPU and FREE the source
   *
   * Shorthand for `.move('cpu')`.
   *
   * @returns New tensor on CPU
   *
   * @example
   * ```ts
   * const gpuTensor = cuda.zeros([100, 100]);
   * const cpuTensor = gpuTensor.moveCpu();
   * // gpuTensor is now INVALID
   * ```
   */
  moveCpu(): Tensor<S, D, 'cpu'> {
    return this.move('cpu')
  }

  /**
   * Move tensor to CUDA and FREE the source
   *
   * Shorthand for `.move('cuda')`.
   *
   * @param deviceIndex - CUDA device index (default: 0)
   * @returns New tensor on CUDA device
   *
   * @example
   * ```ts
   * const cpuTensor = cpu.zeros([100, 100]);
   * const gpuTensor = cpuTensor.moveCuda();
   * // cpuTensor is now INVALID
   * ```
   */
  moveCuda(deviceIndex = 0): Tensor<S, D, 'cuda'> {
    this._checkValid()

    // Same device = no-op
    if ((this.device as string) === 'cuda') {
      return this as unknown as Tensor<S, D, 'cuda'>
    }

    // Create new tensor on CUDA
    const newTensor = this.cuda(deviceIndex)

    // Free the source tensor
    this.free()

    return newTensor
  }

  // ==================== Loss Functions ====================

  /**
   * Compute cross entropy loss between logits and targets
   *
   * @param targets - Target class indices tensor
   * @returns Scalar tensor with mean cross entropy loss
   *
   * @example
   * ```ts
   * const logits = fromArray([[1, 2, 3], [1, 2, 3]], [2, 3], DType.float32, true);
   * const targets = fromArray([2, 0], [2], DType.int64);
   * const loss = logits.crossEntropyLoss(targets);
   * loss.backward();
   * ```
   */
  crossEntropyLoss<TargetD extends DType<string>>(
    targets: Tensor<readonly [S[0]], TargetD>,
  ): Tensor<readonly [], D> {
    this._checkValid()
    const lib = getLib()

    const handle = lib.ts_tensor_cross_entropy_loss(this._handle, targets._handle)

    checkNull(handle, 'Failed to compute cross entropy loss')

    return new Tensor<readonly [], D>(handle!, [] as const, this.dtype)
  }

  /**
   * Compute negative log likelihood loss
   *
   * @param targets - Target class indices tensor
   * @returns Scalar tensor with mean NLL loss
   *
   * @example
   * ```ts
   * const logProbs = fromArray([[...]], [2, 3], DType.float32, true).logSoftmax(1);
   * const targets = fromArray([2, 0], [2], DType.int64);
   * const loss = logProbs.nllLoss(targets);
   * ```
   */
  nllLoss<TargetD extends DType<string>>(targets: Tensor<readonly [S[0]], TargetD>): Tensor<readonly [], D> {
    this._checkValid()
    const lib = getLib()

    const handle = lib.ts_tensor_nll_loss(this._handle, targets._handle)

    checkNull(handle, 'Failed to compute NLL loss')

    return new Tensor<readonly [], D>(handle!, [] as const, this.dtype)
  }

  /**
   * Compute mean squared error loss
   *
   * @param target - Target tensor (same shape as this)
   * @returns Scalar tensor with mean MSE loss
   *
   * @example
   * ```ts
   * const pred = fromArray([1, 2, 3], [3], DType.float32, true);
   * const target = fromArray([1.5, 2.5, 3.5], [3], DType.float32);
   * const loss = pred.mseLoss(target);
   * loss.backward();
   * ```
   */
  mseLoss(target: Tensor<S, D>): Tensor<readonly [], D> {
    this._checkValid()
    const lib = getLib()

    const handle = lib.ts_tensor_mse_loss(this._handle, target._handle)

    checkNull(handle, 'Failed to compute MSE loss')

    return new Tensor<readonly [], D>(handle!, [] as const, this.dtype)
  }

  // ==================== In-place Operations ====================

  /**
   * In-place subtraction: this.data -= other
   *
   * Modifies tensor data directly, bypassing autograd.
   * Used for optimizer parameter updates.
   *
   * @param other - Tensor to subtract
   *
   * @example
   * ```ts
   * param.subInplace(gradient);  // param -= gradient
   * ```
   */
  subInplace(other: Tensor<S, D>): void {
    this._checkValid()
    other._checkValid()
    const lib = getLib()

    lib.ts_tensor_sub_inplace(this._handle, other._handle)
  }

  /**
   * In-place scaled addition: this.data += scalar * other
   *
   * Modifies tensor data directly, bypassing autograd.
   * Used for optimizer parameter updates (e.g., param -= lr * grad).
   *
   * @param other - Tensor to add (scaled)
   * @param scalar - Scalar multiplier
   *
   * @example
   * ```ts
   * param.addScaledInplace(gradient, -learningRate);  // param -= lr * grad
   * ```
   */
  addScaledInplace(other: Tensor<S, D>, scalar: number): void {
    this._checkValid()
    other._checkValid()
    const lib = getLib()

    lib.ts_tensor_add_scaled_inplace(this._handle, other._handle, scalar)
  }

  /**
   * In-place addition: this += other
   *
   * WARNING: Will error if this tensor is a leaf with requires_grad=true
   *
   * @param other - Tensor to add
   */
  addInplace(other: Tensor<S, D>): void {
    this._checkValid()
    other._checkValid()
    const lib = getLib()

    lib.ts_tensor_add_(this._handle, other._handle)
  }

  /**
   * In-place multiplication: this *= other
   *
   * @param other - Tensor to multiply with
   */
  mulInplace(other: Tensor<S, D>): void {
    this._checkValid()
    other._checkValid()
    const lib = getLib()

    lib.ts_tensor_mul_(this._handle, other._handle)
  }

  /**
   * In-place scalar multiplication: this *= scalar
   *
   * @param scalar - Scalar to multiply with
   */
  mulScalarInplace(scalar: number): void {
    this._checkValid()
    const lib = getLib()

    lib.ts_tensor_mul_scalar_(this._handle, scalar)
  }

  /**
   * In-place division: this /= other
   *
   * @param other - Tensor to divide by
   */
  divInplace(other: Tensor<S, D>): void {
    this._checkValid()
    other._checkValid()
    const lib = getLib()

    lib.ts_tensor_div_(this._handle, other._handle)
  }

  /**
   * In-place scalar division: this /= scalar
   *
   * @param scalar - Scalar to divide by
   */
  divScalarInplace(scalar: number): void {
    this._checkValid()
    const lib = getLib()

    lib.ts_tensor_div_scalar_(this._handle, scalar)
  }

  // ==================== Convolution Operations ====================

  /**
   * Apply 2D convolution
   *
   * @param weight - Convolution kernel [OutChannels, InChannels/groups, KernelH, KernelW]
   * @param bias - Optional bias [OutChannels]
   * @param stride - Stride [strideH, strideW]
   * @param padding - Padding [paddingH, paddingW]
   * @param dilation - Dilation [dilationH, dilationW]
   * @param groups - Number of groups for grouped convolution
   * @returns Convolved tensor
   */
  conv2d<WeightS extends Shape, BiasS extends Shape>(
    weight: Tensor<WeightS, D>,
    bias: Tensor<BiasS, D> | null,
    stride: [number, number] = [1, 1],
    padding: [number, number] = [0, 0],
    dilation: [number, number] = [1, 1],
    groups: number = 1,
  ): Tensor<Shape, D> {
    this._checkValid()
    weight._checkValid()
    const lib = getLib()

    const handle = lib.ts_tensor_conv2d(
      this._handle,
      weight._handle,
      bias?._handle ?? null,
      stride[0],
      stride[1],
      padding[0],
      padding[1],
      dilation[0],
      dilation[1],
      groups,
    )

    checkNull(handle, 'Failed to apply conv2d')

    // Compute output shape: [N, OutChannels, H_out, W_out]
    const N = this.shape[0]!
    const outChannels = weight.shape[0]!
    const H_in = this.shape[2]!
    const W_in = this.shape[3]!
    const kernelH = weight.shape[2]!
    const kernelW = weight.shape[3]!

    const H_out = Math.floor((H_in + 2 * padding[0] - dilation[0] * (kernelH - 1) - 1) / stride[0] + 1)
    const W_out = Math.floor((W_in + 2 * padding[1] - dilation[1] * (kernelW - 1) - 1) / stride[1] + 1)

    const resultShape = [N, outChannels, H_out, W_out] as const

    return new Tensor<typeof resultShape, D>(handle!, resultShape, this.dtype)
  }

  // ==================== Pooling Operations ====================

  /**
   * Apply 2D max pooling
   *
   * @param kernelSize - Pooling kernel [kernelH, kernelW]
   * @param stride - Stride [strideH, strideW] (default: kernelSize)
   * @param padding - Padding [paddingH, paddingW]
   * @returns Pooled tensor
   */
  maxPool2d(
    kernelSize: [number, number],
    stride?: [number, number],
    padding: [number, number] = [0, 0],
  ): Tensor<Shape, D> {
    this._checkValid()
    const actualStride = stride ?? kernelSize
    validatePoolingParams({
      kernelSize,
      stride: actualStride,
      padding,
    })
    const lib = getLib()

    const handle = lib.ts_tensor_max_pool2d(
      this._handle,
      kernelSize[0],
      kernelSize[1],
      actualStride[0],
      actualStride[1],
      padding[0],
      padding[1],
    )

    checkNull(handle, 'Failed to apply max_pool2d')

    // Compute output shape
    const N = this.shape[0]!
    const C = this.shape[1]!
    const H_in = this.shape[2]!
    const W_in = this.shape[3]!

    const H_out = Math.floor((H_in + 2 * padding[0] - kernelSize[0]) / actualStride[0] + 1)
    const W_out = Math.floor((W_in + 2 * padding[1] - kernelSize[1]) / actualStride[1] + 1)

    const resultShape = [N, C, H_out, W_out] as const

    return new Tensor<typeof resultShape, D>(handle!, resultShape, this.dtype)
  }

  /**
   * Apply 2D average pooling
   *
   * @param kernelSize - Pooling kernel [kernelH, kernelW]
   * @param stride - Stride [strideH, strideW] (default: kernelSize)
   * @param padding - Padding [paddingH, paddingW]
   * @returns Pooled tensor
   */
  avgPool2d(
    kernelSize: [number, number],
    stride?: [number, number],
    padding: [number, number] = [0, 0],
  ): Tensor<Shape, D> {
    this._checkValid()
    const actualStride = stride ?? kernelSize
    validatePoolingParams({
      kernelSize,
      stride: actualStride,
      padding,
    })
    const lib = getLib()

    const handle = lib.ts_tensor_avg_pool2d(
      this._handle,
      kernelSize[0],
      kernelSize[1],
      actualStride[0],
      actualStride[1],
      padding[0],
      padding[1],
    )

    checkNull(handle, 'Failed to apply avg_pool2d')

    // Compute output shape
    const N = this.shape[0]!
    const C = this.shape[1]!
    const H_in = this.shape[2]!
    const W_in = this.shape[3]!

    const H_out = Math.floor((H_in + 2 * padding[0] - kernelSize[0]) / actualStride[0] + 1)
    const W_out = Math.floor((W_in + 2 * padding[1] - kernelSize[1]) / actualStride[1] + 1)

    const resultShape = [N, C, H_out, W_out] as const

    return new Tensor<typeof resultShape, D>(handle!, resultShape, this.dtype)
  }

  // ==================== Regularization ====================

  /**
   * Apply dropout
   *
   * During training, randomly zeroes some elements with probability p
   * and scales the remaining elements by 1/(1-p).
   *
   * @param p - Probability of an element to be zeroed (default: 0.5)
   * @param training - Whether in training mode (default: true)
   * @returns Tensor with dropout applied
   */
  dropout(p: number = 0.5, training: boolean = true): Tensor<S, D> {
    this._checkValid()
    validateProbability(p, 'p (dropout probability)')
    const lib = getLib()

    const handle = lib.ts_tensor_dropout(this._handle, p, training ? 1 : 0)

    checkNull(handle, 'Failed to apply dropout')

    return new Tensor<S, D>(handle!, this.shape, this.dtype)
  }

  // ==================== Normalization ====================

  /**
   * Apply batch normalization
   *
   * @param weight - Scale parameter (gamma)
   * @param bias - Shift parameter (beta)
   * @param runningMean - Running mean for inference
   * @param runningVar - Running variance for inference
   * @param training - Whether in training mode
   * @param momentum - Momentum for running stats update
   * @param eps - Small value for numerical stability
   * @returns Normalized tensor
   */
  batchNorm(
    weight: Tensor<Shape, D> | null,
    bias: Tensor<Shape, D> | null,
    runningMean: Tensor<Shape, D> | null,
    runningVar: Tensor<Shape, D> | null,
    training: boolean = true,
    momentum: number = 0.1,
    eps: number = 1e-5,
  ): Tensor<S, D> {
    this._checkValid()
    validateRange(momentum, 0, 1, 'momentum')
    validatePositive(eps, 'eps')
    const lib = getLib()

    const handle = lib.ts_tensor_batch_norm(
      this._handle,
      weight?._handle ?? null,
      bias?._handle ?? null,
      runningMean?._handle ?? null,
      runningVar?._handle ?? null,
      training ? 1 : 0,
      momentum,
      eps,
    )

    checkNull(handle, 'Failed to apply batch_norm')

    return new Tensor<S, D>(handle!, this.shape, this.dtype)
  }

  /**
   * Apply layer normalization
   *
   * @param normalizedShape - Shape over which to normalize
   * @param weight - Scale parameter (gamma)
   * @param bias - Shift parameter (beta)
   * @param eps - Small value for numerical stability
   * @returns Normalized tensor
   */
  layerNorm(
    normalizedShape: readonly number[],
    weight: Tensor<Shape, D> | null,
    bias: Tensor<Shape, D> | null,
    eps: number = 1e-5,
  ): Tensor<S, D> {
    this._checkValid()
    validatePositive(eps, 'eps')
    for (let i = 0; i < normalizedShape.length; i++) {
      validatePositiveInt(normalizedShape[i]!, `normalizedShape[${i}]`)
    }
    const lib = getLib()

    // Convert normalized shape to BigInt64Array for FFI
    const shapeArray = new BigInt64Array(normalizedShape.map((dim) => BigInt(dim)))

    const handle = lib.ts_tensor_layer_norm(
      this._handle,
      shapeArray,
      weight?._handle ?? null,
      bias?._handle ?? null,
      eps,
    )

    checkNull(handle, 'Failed to apply layer_norm')

    return new Tensor<S, D>(handle!, this.shape, this.dtype)
  }

  // ==================== Module Integration ====================

  /**
   * Pipe tensor through a module
   *
   * Enables functional-style module composition
   *
   * @template OutS - Output shape type
   * @param module - Module with forward method
   * @returns Result of module.forward(this)
   *
   * @example
   * ```ts
   * const x = zeros([1, 28, 28], DType.float32);
   * const y = x
   *   .pipe(conv1)
   *   .pipe(relu)
   *   .pipe(pool)
   *   .pipe(linear);
   * ```
   */
  pipe<OutS extends Shape>(module: { forward(x: Tensor<S, D, Dev>): Tensor<OutS, D, Dev> }): Tensor<OutS, D, Dev> {
    return module.forward(this)
  }

  // ==================== String Representation ====================

  /**
   * String representation of tensor
   *
   * @example
   * ```ts
   * const t = zeros([2, 3], DType.float32);
   * console.log(t.toString()); // "Tensor<[2, 3], float32>"
   * ```
   */
  toString(): string {
    return `Tensor<[${this.shape.join(', ')}], ${this.dtype.name}>`
  }

  /**
   * JSON representation of tensor
   *
   * @example
   * ```ts
   * const t = zeros([2, 3], DType.float32);
   * console.log(JSON.stringify(t));
   * ```
   */
  toJSON(): object {
    return {
      shape: this.shape,
      dtype: this.dtype.name,
      data: Array.from(this.toArray() as Iterable<number | bigint>),
    }
  }

  // ==================== Comparison Operations ====================

  /**
   * Element-wise equality comparison
   *
   * @param other - Tensor to compare with
   * @returns Boolean tensor with true where elements are equal
   *
   * @example
   * ```ts
   * import { device } from '@ts-torch/core'
   * const cpu = device.cpu()
   * const a = cpu.tensor([1, 2, 3], [3] as const)
   * const b = cpu.tensor([1, 0, 3], [3] as const)
   * const eq = a.eq(b) // [true, false, true]
   * ```
   */
  eq(other: Tensor<S, D>): Tensor<S, DType<'bool'>> {
    this._checkValid()
    const lib = getLib()

    const handle = lib.ts_tensor_eq(this._handle, other._handle)

    checkNull(handle, 'Failed to compute eq')

    const boolDtype = { name: 'bool' } as DType<'bool'>
    return new Tensor<S, DType<'bool'>>(handle!, this.shape, boolDtype)
  }

  // ==================== Indexing Operations ====================

  /**
   * Select elements along a dimension using an index tensor
   *
   * This operation selects rows (or elements along any dimension) based on
   * the indices provided. Useful for batching and data loading on GPU.
   *
   * @param dim - Dimension to index along
   * @param index - 1D tensor of indices (int64)
   * @returns New tensor with selected elements
   *
   * @example
   * ```ts
   * import { device, int64 } from '@ts-torch/core'
   * const cpu = device.cpu()
   * // Select rows 0, 2, 1 from a 4x3 tensor
   * const data = cpu.tensor([[1,2,3], [4,5,6], [7,8,9], [10,11,12]], [4, 3] as const)
   * const indices = cpu.tensor([0, 2, 1], [3] as const, int64)
   * const selected = data.indexSelect(0, indices) // [[1,2,3], [7,8,9], [4,5,6]]
   * ```
   */
  indexSelect<IndexS extends readonly [number]>(
    dim: number,
    index: Tensor<IndexS, DType<'int64'>>,
  ): Tensor<Shape, D> {
    this._checkValid()
    validateDimension(dim, this.ndim, 'dim')
    const lib = getLib()

    const handle =       lib.ts_tensor_index_select(this._handle, dim, index._handle)

    checkNull(handle, 'Failed to apply index_select')

    // Compute output shape: replace dim's size with index length
    const newShape = [...this.shape] as number[]
    newShape[dim] = index.shape[0]

    return new Tensor<Shape, D>(handle!, newShape as unknown as Shape, this.dtype)
  }

  /**
   * Returns the indices of maximum values along a dimension
   *
   * @param dim - Dimension to reduce
   * @param keepdim - Whether to keep the reduced dimension
   * @returns Tensor of indices (int64)
   *
   * @example
   * ```ts
   * import { device } from '@ts-torch/core'
   * const cpu = device.cpu()
   * const logits = cpu.tensor([[1, 3, 2], [4, 2, 5]], [2, 3] as const)
   * const preds = logits.argmax(1) // [1, 2] - indices of max in each row
   * ```
   */
  argmax(dim: number, keepdim: boolean = false): Tensor<Shape, DType<'int64'>> {
    this._checkValid()
    validateDimension(dim, this.ndim, 'dim')
    const lib = getLib()

    const handle = lib.ts_tensor_argmax(this._handle, dim, keepdim)

    checkNull(handle, 'Failed to apply argmax')

    // Compute output shape
    let newShape: number[]
    if (keepdim) {
      newShape = [...this.shape] as number[]
      newShape[dim] = 1
    } else {
      newShape = (this.shape as readonly number[]).filter((_, i) => i !== dim)
    }

    // Import DType properly for int64
    const int64Dtype = { name: 'int64' } as DType<'int64'>
    return new Tensor<Shape, DType<'int64'>>(handle!, newShape as unknown as Shape, int64Dtype)
  }

  /**
   * Narrow (slice) tensor along a dimension - ZERO COPY view
   *
   * This is the most efficient way to get a contiguous slice of a tensor.
   * Returns a view into the same memory - no data is copied.
   *
   * @param dim - Dimension to narrow along
   * @param start - Starting index
   * @param length - Length of the slice
   * @returns View tensor (shares memory with original)
   *
   * @example
   * ```ts
   * // Get batch 0-512 from a 60000x784 tensor
   * const batch = allImages.narrow(0, 0, 512); // [512, 784] - zero copy!
   * ```
   */
  narrow(dim: number, start: number, length: number): Tensor<Shape, D> {
    this._checkValid()
    validateDimension(dim, this.ndim, 'dim')
    const lib = getLib()

    const handle = lib.ts_tensor_narrow(this._handle, dim, start, length)

    checkNull(handle, 'Failed to narrow tensor')

    // Compute output shape
    const newShape = [...this.shape] as number[]
    newShape[dim] = length

    return new Tensor<Shape, D>(handle!, newShape as unknown as Shape, this.dtype)
  }

  // ==================== Advanced Tensor Operations ====================

  /**
   * Returns upper triangular part of matrix
   *
   * Elements on and above the diagonal are kept, elements below are set to zero.
   *
   * @param diagonal - Offset from main diagonal (default: 0)
   *   - diagonal=0: main diagonal
   *   - diagonal>0: above main diagonal
   *   - diagonal<0: below main diagonal
   * @returns Upper triangular tensor
   *
   * @example
   * ```ts
   * const a = cpu.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [3, 3])
   * a.triu()   // [[1, 2, 3], [0, 5, 6], [0, 0, 9]]
   * a.triu(1)  // [[0, 2, 3], [0, 0, 6], [0, 0, 0]]
   * a.triu(-1) // [[1, 2, 3], [4, 5, 6], [0, 8, 9]]
   * ```
   */
  triu(diagonal: number = 0): Tensor<S, D> {
    this._checkValid()
    const lib = getLib()

    const handle = lib.ts_tensor_triu(this._handle, diagonal)

    checkNull(handle, 'Failed to compute triu')

    return new Tensor<S, D>(handle!, this.shape, this.dtype)
  }

  /**
   * Returns lower triangular part of matrix
   *
   * Elements on and below the diagonal are kept, elements above are set to zero.
   *
   * @param diagonal - Offset from main diagonal (default: 0)
   * @returns Lower triangular tensor
   *
   * @example
   * ```ts
   * const a = cpu.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [3, 3])
   * a.tril()  // [[1, 0, 0], [4, 5, 0], [7, 8, 9]]
   * ```
   */
  tril(diagonal: number = 0): Tensor<S, D> {
    this._checkValid()
    const lib = getLib()

    const handle = lib.ts_tensor_tril(this._handle, diagonal)

    checkNull(handle, 'Failed to compute tril')

    return new Tensor<S, D>(handle!, this.shape, this.dtype)
  }

  /**
   * Fill elements where mask is true with value
   *
   * Critical for attention masking in transformers.
   *
   * @param mask - Boolean tensor of same shape as input
   * @param value - Value to fill where mask is true
   * @returns Tensor with masked positions filled
   *
   * @example
   * ```ts
   * // Causal attention masking
   * const mask = cpu.ones([seqLen, seqLen]).triu(1) // upper triangle
   * const scores = attention.maskedFill(mask, -Infinity)
   * ```
   */
  maskedFill(mask: Tensor<S, DType<'bool'>>, value: number): Tensor<S, D> {
    this._checkValid()
    const lib = getLib()

    const handle =       lib.ts_tensor_masked_fill(this._handle, mask._handle, value)

    checkNull(handle, 'Failed to apply masked_fill')

    return new Tensor<S, D>(handle!, this.shape, this.dtype)
  }

  /**
   * Batched matrix multiplication
   *
   * Performs matrix multiplication on batched 3D tensors.
   * Input: [B, M, K] @ [B, K, N] -> [B, M, N]
   *
   * @param other - Second tensor [B, K, N]
   * @returns Result tensor [B, M, N]
   *
   * @example
   * ```ts
   * // Batched attention: Q @ K^T
   * const Q = cpu.randn([batchSize, seqLen, headDim])
   * const Kt = cpu.randn([batchSize, headDim, seqLen])
   * const scores = Q.bmm(Kt) // [batchSize, seqLen, seqLen]
   * ```
   */
  bmm<OutN extends number>(
    other: Tensor<readonly [number, number, OutN], D>,
  ): Tensor<readonly [number, number, OutN], D> {
    this._checkValid()
    const lib = getLib()

    const handle =       lib.ts_tensor_bmm(this._handle, other._handle)

    checkNull(handle, 'Failed to apply bmm')

    // Compute output shape [B, M, N]
    const batch = (this.shape as readonly number[])[0]
    const m = (this.shape as readonly number[])[1]
    const n = other.shape[2]
    const newShape = [batch, m, n] as const

    return new Tensor<readonly [number, number, OutN], D>(
      handle!,
      newShape as unknown as readonly [number, number, OutN],
      this.dtype,
    )
  }

  /**
   * Gather values along an axis using indices
   *
   * For each position in index, gathers the value from input at that index
   * along the specified dimension.
   *
   * @param dim - Dimension along which to gather
   * @param index - Index tensor (int64)
   * @returns Gathered tensor with same shape as index
   *
   * @example
   * ```ts
   * const src = cpu.tensor([[1, 2], [3, 4]], [2, 2])
   * const idx = cpu.tensor([[0, 0], [1, 0]], [2, 2], int64)
   * src.gather(1, idx) // [[1, 1], [4, 3]]
   * ```
   */
  gather(dim: number, index: Tensor<Shape, DType<'int64'>>): Tensor<Shape, D> {
    this._checkValid()
    validateDimension(dim, this.ndim, 'dim')
    const lib = getLib()

    const handle =       lib.ts_tensor_gather(this._handle, dim, index._handle)

    checkNull(handle, 'Failed to apply gather')

    // Output shape is same as index shape
    return new Tensor<Shape, D>(handle!, index.shape, this.dtype)
  }

  /**
   * Scatter values into tensor at positions specified by index
   *
   * @param dim - Dimension along which to scatter
   * @param index - Index tensor (int64)
   * @param src - Source values to scatter
   * @returns New tensor with scattered values
   *
   * @example
   * ```ts
   * const dst = cpu.zeros([2, 3])
   * const idx = cpu.tensor([[0, 2]], [1, 2], int64)
   * const src = cpu.tensor([[1, 2]], [1, 2])
   * dst.scatter(1, idx, src) // [[1, 0, 2], [0, 0, 0]]
   * ```
   */
  scatter(
    dim: number,
    index: Tensor<Shape, DType<'int64'>>,
    src: Tensor<Shape, D>,
  ): Tensor<S, D> {
    this._checkValid()
    validateDimension(dim, this.ndim, 'dim')
    const lib = getLib()

    const handle =       lib.ts_tensor_scatter(this._handle, dim, index._handle, src._handle)

    checkNull(handle, 'Failed to apply scatter')

    return new Tensor<S, D>(handle!, this.shape, this.dtype)
  }

  /**
   * Returns top k elements and their indices
   *
   * Essential for top-k sampling in language models.
   *
   * @param k - Number of top elements to return
   * @param dim - Dimension to sort along (default: -1, last dimension)
   * @param largest - Return largest elements if true (default), smallest if false
   * @param sorted - Return elements in sorted order (default: true)
   * @returns Tuple of [values, indices]
   *
   * @example
   * ```ts
   * const logits = cpu.tensor([1.0, 5.0, 3.0, 2.0], [4])
   * const [values, indices] = logits.topk(2)
   * // values: [5.0, 3.0], indices: [1, 2]
   * ```
   */
  topk(
    k: number,
    dim: number = -1,
    largest: boolean = true,
    sorted: boolean = true,
  ): [Tensor<Shape, D>, Tensor<Shape, DType<'int64'>>] {
    this._checkValid()
    const normalizedDim = dim < 0 ? this.ndim + dim : dim
    validateDimension(normalizedDim, this.ndim, 'dim')
    const lib = getLib()

    // Napi wrapper returns [values_handle, indices_handle] as a JS array
    const result = lib.ts_tensor_topk(this._handle, k, normalizedDim, largest, sorted) as [Pointer, Pointer]

    checkNull(result?.[0], 'Failed to compute topk')

    // Compute output shape
    const newShape = [...(this.shape as readonly number[])]
    newShape[normalizedDim] = k

    const int64Dtype = { name: 'int64' } as DType<'int64'>

    const values = new Tensor<Shape, D>(result[0]!, newShape as unknown as Shape, this.dtype)
    const indices = new Tensor<Shape, DType<'int64'>>(
      result[1]!,
      newShape as unknown as Shape,
      int64Dtype,
    )

    return [values, indices]
  }

  /**
   * Sort tensor along a dimension
   *
   * @param dim - Dimension to sort along (default: -1)
   * @param descending - Sort in descending order (default: false)
   * @returns Tuple of [sorted values, indices]
   *
   * @example
   * ```ts
   * const x = cpu.tensor([3, 1, 4, 1, 5], [5])
   * const [sorted, indices] = x.sort()
   * // sorted: [1, 1, 3, 4, 5], indices: [1, 3, 0, 2, 4]
   * ```
   */
  sort(
    dim: number = -1,
    descending: boolean = false,
  ): [Tensor<S, D>, Tensor<S, DType<'int64'>>] {
    this._checkValid()
    const normalizedDim = dim < 0 ? this.ndim + dim : dim
    validateDimension(normalizedDim, this.ndim, 'dim')
    const lib = getLib()

    // Napi wrapper returns [values_handle, indices_handle] as a JS array
    const result = lib.ts_tensor_sort(this._handle, normalizedDim, descending) as [Pointer, Pointer]

    checkNull(result?.[0], 'Failed to sort tensor')

    const int64Dtype = { name: 'int64' } as DType<'int64'>

    const values = new Tensor<S, D>(result[0]!, this.shape, this.dtype)
    const indices = new Tensor<S, DType<'int64'>>(
      result[1]!,
      this.shape,
      int64Dtype,
    )

    return [values, indices]
  }

  /**
   * Conditional element selection
   *
   * Returns a tensor where output[i] = x[i] if condition[i] else y[i]
   *
   * @param condition - Boolean condition tensor
   * @param y - Values to use where condition is false
   * @returns Tensor with conditionally selected values
   *
   * @example
   * ```ts
   * const x = cpu.tensor([1, 2, 3], [3])
   * const y = cpu.tensor([4, 5, 6], [3])
   * const cond = cpu.tensor([true, false, true], [3], bool)
   * Tensor.where(cond, x, y) // [1, 5, 3]
   * ```
   */
  static where<S extends Shape, D extends DType<string>>(
    condition: Tensor<S, DType<'bool'>>,
    x: Tensor<S, D>,
    y: Tensor<S, D>,
  ): Tensor<S, D> {
    x._checkValid()
    y._checkValid()
    condition._checkValid()
    const lib = getLib()

    const handle =       lib.ts_tensor_where(condition._handle, x._handle, y._handle)

    checkNull(handle, 'Failed to apply where')

    return new Tensor<S, D>(handle!, x.shape, x.dtype)
  }

  /**
   * Find indices of non-zero elements
   *
   * @returns 2D tensor of shape [num_nonzero, ndim] where each row
   *          contains the indices of a non-zero element
   *
   * @example
   * ```ts
   * const x = cpu.tensor([[1, 0], [0, 2]], [2, 2])
   * x.nonzero() // [[0, 0], [1, 1]] - positions of 1 and 2
   * ```
   */
  nonzero(): Tensor<readonly [number, number], DType<'int64'>> {
    this._checkValid()
    const lib = getLib()

    const handle =       lib.ts_tensor_nonzero(this._handle)

    checkNull(handle, 'Failed to find nonzero elements')

    // Shape is determined at runtime [num_nonzero, ndim]
    const int64Dtype = { name: 'int64' } as DType<'int64'>
    const tensorResult = new Tensor<Shape, DType<'int64'>>(
      handle!,
      [0, this.ndim] as unknown as Shape, // Placeholder, actual shape from native
      int64Dtype,
    )

    return tensorResult as Tensor<readonly [number, number], DType<'int64'>>
  }

  /**
   * Repeat tensor along dimensions
   *
   * @param repeats - Number of repetitions for each dimension
   * @returns Repeated tensor
   *
   * @example
   * ```ts
   * const x = cpu.tensor([[1, 2], [3, 4]], [2, 2])
   * x.repeat([2, 3]) // [[1,2,1,2,1,2], [3,4,3,4,3,4], [1,2,1,2,1,2], [3,4,3,4,3,4]]
   * // shape: [4, 6]
   * ```
   */
  repeat(repeats: number[]): Tensor<Shape, D> {
    this._checkValid()
    if (repeats.length !== this.ndim) {
      throw new Error(`repeats length ${repeats.length} must match tensor ndim ${this.ndim}`)
    }
    const lib = getLib()

    // Convert to BigInt array for FFI
    const repeatsPtr = new BigInt64Array(repeats.map((r) => BigInt(r)))

    const handle =       lib.ts_tensor_repeat(this._handle, repeatsPtr, repeats.length)

    checkNull(handle, 'Failed to repeat tensor')

    // Compute output shape
    const newShape = (this.shape as readonly number[]).map((s, i) => s * repeats[i]!)

    return new Tensor<Shape, D>(handle!, newShape as unknown as Shape, this.dtype)
  }

  /**
   * Expand tensor to larger size (broadcast without copying)
   *
   * Only dimensions of size 1 can be expanded. -1 means keep original size.
   *
   * @param sizes - Target sizes for each dimension
   * @returns Expanded tensor (view, no data copy)
   *
   * @example
   * ```ts
   * const x = cpu.tensor([[1], [2], [3]], [3, 1])
   * x.expand([3, 4]) // [[1,1,1,1], [2,2,2,2], [3,3,3,3]]
   * ```
   */
  expand(sizes: number[]): Tensor<Shape, D> {
    this._checkValid()
    const lib = getLib()

    // Convert to BigInt array for FFI
    const sizesPtr = new BigInt64Array(sizes.map((s) => BigInt(s)))

    const handle =       lib.ts_tensor_expand(this._handle, sizesPtr, sizes.length)

    checkNull(handle, 'Failed to expand tensor')

    // Compute output shape (-1 means keep original)
    const newShape = sizes.map((s, i) =>
      s === -1 ? (this.shape as readonly number[])[i] : s,
    )

    return new Tensor<Shape, D>(handle!, newShape as unknown as Shape, this.dtype)
  }

  // ==================== Fused Operations (Phase 3: Performance) ====================

  /**
   * Fused linear + ReLU: relu(x @ W^T + b)
   *
   * Combines linear layer and ReLU activation in a single FFI call.
   *
   * @param weight - Weight tensor [outFeatures, inFeatures]
   * @param bias - Optional bias tensor [outFeatures]
   * @returns Output tensor with linear + ReLU applied
   *
   * @internal Used by nn.functional.linearRelu
   */
  linearRelu<OutFeatures extends number>(
    weight: Tensor<readonly [OutFeatures, number], D>,
    bias?: Tensor<readonly [OutFeatures], D>,
  ): Tensor<Shape, D> {
    this._checkValid()
    const lib = getLib()

    const handle =       lib.ts_tensor_linear_relu(this._handle, weight._handle, bias?._handle ?? null)

    checkNull(handle, 'Failed to apply fused linearRelu')

    // Compute output shape: replace last dim with outFeatures
    const newShape = [...(this.shape as readonly number[])]
    newShape[newShape.length - 1] = weight.shape[0]

    return new Tensor<Shape, D>(handle!, newShape as unknown as Shape, this.dtype)
  }

  /**
   * Fused linear + Sigmoid: sigmoid(x @ W^T + b)
   *
   * @param weight - Weight tensor [outFeatures, inFeatures]
   * @param bias - Optional bias tensor [outFeatures]
   * @returns Output tensor with linear + sigmoid applied
   *
   * @internal Used by nn.functional.linearSigmoid
   */
  linearSigmoid<OutFeatures extends number>(
    weight: Tensor<readonly [OutFeatures, number], D>,
    bias?: Tensor<readonly [OutFeatures], D>,
  ): Tensor<Shape, D> {
    this._checkValid()
    const lib = getLib()

    const handle =       lib.ts_tensor_linear_sigmoid(this._handle, weight._handle, bias?._handle ?? null)

    checkNull(handle, 'Failed to apply fused linearSigmoid')

    const newShape = [...(this.shape as readonly number[])]
    newShape[newShape.length - 1] = weight.shape[0]

    return new Tensor<Shape, D>(handle!, newShape as unknown as Shape, this.dtype)
  }

  /**
   * Fused linear + Tanh: tanh(x @ W^T + b)
   *
   * @param weight - Weight tensor [outFeatures, inFeatures]
   * @param bias - Optional bias tensor [outFeatures]
   * @returns Output tensor with linear + tanh applied
   *
   * @internal Used by nn.functional.linearTanh
   */
  linearTanh<OutFeatures extends number>(
    weight: Tensor<readonly [OutFeatures, number], D>,
    bias?: Tensor<readonly [OutFeatures], D>,
  ): Tensor<Shape, D> {
    this._checkValid()
    const lib = getLib()

    const handle =       lib.ts_tensor_linear_tanh(this._handle, weight._handle, bias?._handle ?? null)

    checkNull(handle, 'Failed to apply fused linearTanh')

    const newShape = [...(this.shape as readonly number[])]
    newShape[newShape.length - 1] = weight.shape[0]

    return new Tensor<Shape, D>(handle!, newShape as unknown as Shape, this.dtype)
  }

  /**
   * Fused add + ReLU: relu(this + other)
   *
   * @param other - Tensor to add
   * @returns relu(this + other)
   *
   * @internal Used by nn.functional.addRelu
   */
  addRelu(other: Tensor<S, D>): Tensor<S, D> {
    this._checkValid()
    const lib = getLib()

    const handle = lib.ts_tensor_add_relu(this._handle, other._handle)

    checkNull(handle, 'Failed to apply fused addRelu')

    return new Tensor<S, D>(handle!, this.shape, this.dtype)
  }
}

/**
 * Type alias for any tensor regardless of shape, dtype, or device
 */
export type AnyTensor = Tensor<Shape, DType<string>, DeviceType>
