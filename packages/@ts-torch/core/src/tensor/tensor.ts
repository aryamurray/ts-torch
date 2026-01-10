/**
 * Core Tensor class for ts-torch
 *
 * Wraps native tensor handles and provides type-safe operations with
 * compile-time shape checking.
 */

import type { Shape } from '../types/shape.js'
import type { DType } from '../types/dtype.js'
import type { Device } from '../types/index.js'
import type { MatMulShape, TransposeShape } from '../types/tensor.js'
import { getLib } from '../ffi/loader.js'
import { withError, checkNull, TorchError, ErrorCode, type Pointer } from '../ffi/error.js'
import { BytesPerElement } from '../types/dtype.js'
import { escapeTensor as scopeEscapeTensor, inScope } from '../memory/scope.js'

/**
 * Core Tensor class representing a multi-dimensional array with type-safe operations
 *
 * @template S - Shape type as readonly tuple of dimensions
 * @template D - Data type
 *
 * @example
 * ```ts
 * const t = zeros([2, 3], DType.float32);
 * const result = t.add(ones([2, 3], DType.float32));
 * ```
 */
export class Tensor<S extends Shape = Shape, D extends DType<string> = DType<'float32'>> {
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
   *
   * @internal
   * Use factory functions (zeros, ones, etc.) instead of calling constructor directly
   */
  constructor(handle: Pointer, shape: S, dtype: D) {
    this._handle = handle
    this.shape = shape
    this.dtype = dtype

    // Register with current scope if exists
    this._registerWithScope()
  }

  /**
   * Register tensor with current memory scope for automatic cleanup
   * @internal
   */
  private _registerWithScope(): void {
    // TODO: Implement scope tracking
    // For now, tensors must be manually freed or will be GC'd
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

    const result = withError((err) => lib.ts_tensor_requires_grad(this._handle, err))
    return result !== 0 // Convert i32 to boolean
  }

  /**
   * Enable or disable gradient tracking
   */
  set requiresGrad(value: boolean) {
    this._checkValid()
    const lib = getLib()

    withError((err) => lib.ts_tensor_set_requires_grad(this._handle, value, err))

    // Clear gradient cache
    this._gradCache = undefined
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

    const lib = getLib()
    lib.ts_tensor_delete(this._handle)
    this._freed = true
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

    const handle = withError((err) => lib.ts_tensor_clone(this._handle, err))

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

    const handle = withError((err) => lib.ts_tensor_detach(this._handle, err))

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

    // Allocate buffer for data
    const byteSize = this.numel * (BytesPerElement[this.dtype.name as keyof typeof BytesPerElement] || 4)

    let buffer: ArrayBuffer
    let result: Float32Array | Float64Array | Int32Array | BigInt64Array

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

      default:
        throw new Error(`Unsupported dtype: ${this.dtype.name}`)
    }

    // Copy data from native memory (koffi accepts ArrayBuffer directly)
    withError((err) => lib.ts_tensor_copy_to_buffer(this._handle, buffer, BigInt(byteSize), err))

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
   * @returns New tensor with result
   *
   * @example
   * ```ts
   * const a = ones([2, 3], DType.float32);
   * const b = ones([2, 3], DType.float32);
   * const c = a.add(b); // [[2, 2, 2], [2, 2, 2]]
   * ```
   */
  add(other: Tensor<S, D>): Tensor<S, D> {
    this._checkValid()
    other._checkValid()
    const lib = getLib()

    const handle = withError((err) => lib.ts_tensor_add(this._handle, other._handle, err))

    checkNull(handle, 'Failed to add tensors')

    return new Tensor<S, D>(handle!, this.shape, this.dtype)
  }

  /**
   * Element-wise subtraction
   *
   * @param other - Tensor to subtract (must have same shape)
   * @returns New tensor with result
   *
   * @example
   * ```ts
   * const a = ones([2, 3], DType.float32);
   * const b = ones([2, 3], DType.float32);
   * const c = a.sub(b); // [[0, 0, 0], [0, 0, 0]]
   * ```
   */
  sub(other: Tensor<S, D>): Tensor<S, D> {
    this._checkValid()
    other._checkValid()
    const lib = getLib()

    const handle = withError((err) => lib.ts_tensor_sub(this._handle, other._handle, err))

    checkNull(handle, 'Failed to subtract tensors')

    return new Tensor<S, D>(handle!, this.shape, this.dtype)
  }

  /**
   * Element-wise multiplication
   *
   * @param other - Tensor to multiply (must have same shape)
   * @returns New tensor with result
   *
   * @example
   * ```ts
   * const a = fromArray([[1, 2], [3, 4]], [2, 2], DType.float32);
   * const b = fromArray([[2, 2], [2, 2]], [2, 2], DType.float32);
   * const c = a.mul(b); // [[2, 4], [6, 8]]
   * ```
   */
  mul(other: Tensor<S, D>): Tensor<S, D> {
    this._checkValid()
    other._checkValid()
    const lib = getLib()

    const handle = withError((err) => lib.ts_tensor_mul(this._handle, other._handle, err))

    checkNull(handle, 'Failed to multiply tensors')

    return new Tensor<S, D>(handle!, this.shape, this.dtype)
  }

  /**
   * Element-wise division
   *
   * @param other - Tensor to divide by (must have same shape)
   * @returns New tensor with result
   *
   * @example
   * ```ts
   * const a = fromArray([[2, 4], [6, 8]], [2, 2], DType.float32);
   * const b = fromArray([[2, 2], [2, 2]], [2, 2], DType.float32);
   * const c = a.div(b); // [[1, 2], [3, 4]]
   * ```
   */
  div(other: Tensor<S, D>): Tensor<S, D> {
    this._checkValid()
    other._checkValid()
    const lib = getLib()

    const handle = withError((err) => lib.ts_tensor_div(this._handle, other._handle, err))

    checkNull(handle, 'Failed to divide tensors')

    return new Tensor<S, D>(handle!, this.shape, this.dtype)
  }

  // ==================== Matrix Operations ====================

  /**
   * Matrix multiplication with type-safe shape inference
   *
   * @template S2 - Shape of other tensor
   * @param other - Tensor to multiply with
   * @returns New tensor with result shape computed at compile time
   *
   * @example
   * ```ts
   * const a = zeros([2, 3], DType.float32);
   * const b = zeros([3, 4], DType.float32);
   * const c = a.matmul(b); // Type: Tensor<[2, 4], DType<"float32">>
   * ```
   */
  matmul<S2 extends Shape>(other: Tensor<S2, D>): Tensor<MatMulShape<S, S2>, D> {
    this._checkValid()
    other._checkValid()
    const lib = getLib()

    const handle = withError((err) => lib.ts_tensor_matmul(this._handle, other._handle, err))

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
    const lib = getLib()

    const handle = withError((err) => lib.ts_tensor_transpose(this._handle, dim0, dim1, err))

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
    const lib = getLib()

    // Convert shape to BigInt64Array for FFI (koffi accepts ArrayBuffer directly)
    const shapeArray = new BigInt64Array(shape.map((dim) => BigInt(dim)))

    const handle = withError((err) => lib.ts_tensor_reshape(this._handle, shapeArray.buffer, shape.length, err))

    checkNull(handle, 'Failed to reshape tensor')

    return new Tensor<NewS, D>(handle!, shape, this.dtype)
  }

  // ==================== Reductions ====================

  /**
   * Sum all elements to scalar
   *
   * @returns Scalar tensor with sum of all elements
   *
   * @example
   * ```ts
   * const a = fromArray([[1, 2], [3, 4]], [2, 2], DType.float32);
   * const sum = a.sum(); // Tensor(10)
   * ```
   */
  sum(): Tensor<readonly [], D> {
    this._checkValid()
    const lib = getLib()

    const handle = withError((err) => lib.ts_tensor_sum(this._handle, err))

    checkNull(handle, 'Failed to compute sum')

    return new Tensor<readonly [], D>(handle!, [] as const, this.dtype)
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

    const handle = withError((err) => lib.ts_tensor_mean(this._handle, err))

    checkNull(handle, 'Failed to compute mean')

    return new Tensor<readonly [], D>(handle!, [] as const, this.dtype)
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

    const handle = withError((err) => lib.ts_tensor_relu(this._handle, err))

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

    const handle = withError((err) => lib.ts_tensor_sigmoid(this._handle, err))

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
    const lib = getLib()

    const handle = withError((err) => lib.ts_tensor_softmax(this._handle, dim, err))

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
    const lib = getLib()

    const handle = withError((err) => lib.ts_tensor_log_softmax(this._handle, dim, err))

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

    const handle = withError((err) => lib.ts_tensor_log(this._handle, err))

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

    const handle = withError((err) => lib.ts_tensor_exp(this._handle, err))

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

    const handle = withError((err) => lib.ts_tensor_neg(this._handle, err))

    checkNull(handle, 'Failed to apply neg')

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
    const lib = getLib()

    const handle = withError((err) => lib.ts_tensor_add_scalar(this._handle, scalar, err))

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
    const lib = getLib()

    const handle = withError((err) => lib.ts_tensor_sub_scalar(this._handle, scalar, err))

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
    const lib = getLib()

    const handle = withError((err) => lib.ts_tensor_mul_scalar(this._handle, scalar, err))

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
    const lib = getLib()

    const handle = withError((err) => lib.ts_tensor_div_scalar(this._handle, scalar, err))

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

    withError((err) => lib.ts_tensor_backward(this._handle, err))
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

    withError((err) => lib.ts_tensor_zero_grad(this._handle, err))
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
      return this._gradCache
    }

    const lib = getLib()

    const handle = withError((err) => lib.ts_tensor_grad(this._handle, err))

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
   * @param device - Target device ('cpu', 'cuda', 'mps')
   * @returns New tensor on target device
   *
   * @example
   * ```ts
   * const a = zeros([2, 3], DType.float32); // CPU
   * const b = a.to('cuda'); // CUDA device
   * ```
   */
  to(device: Device): Tensor<S, D> {
    this._checkValid()
    const lib = getLib()

    // Map device string to enum
    let deviceType: number
    let deviceId = 0

    switch (device) {
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
        throw new Error(`Unknown device: ${device}`)
    }

    const handle = withError((err) => lib.ts_tensor_to_device(this._handle, deviceType, deviceId, err))

    checkNull(handle, 'Failed to move tensor to device')

    return new Tensor<S, D>(handle!, this.shape, this.dtype)
  }

  /**
   * Move tensor to CPU
   *
   * @returns New tensor on CPU
   *
   * @example
   * ```ts
   * const a = zeros([2, 3], DType.float32).to('cuda');
   * const b = a.cpu(); // Back to CPU
   * ```
   */
  cpu(): Tensor<S, D> {
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
   * const a = zeros([2, 3], DType.float32);
   * const b = a.cuda(); // CUDA:0
   * const c = a.cuda(1); // CUDA:1
   * ```
   */
  cuda(deviceIndex = 0): Tensor<S, D> {
    this._checkValid()
    const lib = getLib()

    const handle = withError((err) => lib.ts_tensor_to_device(this._handle, 1, deviceIndex, err))

    checkNull(handle, 'Failed to move tensor to CUDA')

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
  pipe<OutS extends Shape>(module: { forward(x: Tensor<S, D>): Tensor<OutS, D> }): Tensor<OutS, D> {
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
}
