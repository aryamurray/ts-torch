/**
 * Tensor type operations for compile-time shape inference
 *
 * Provides type-level functions to compute result shapes of tensor operations,
 * enabling compile-time verification of shape compatibility.
 */

import type { DTypeName } from './dtype'
import type { Shape, Rank, SetDim, RemoveDim, InsertDim, NumElements, Reverse } from './shape'

/**
 * Device type for tensors
 */
export type DeviceType = 'cpu' | 'cuda' | 'mps'

/**
 * Core tensor type combining shape, dtype, and device information
 *
 * @template S - The shape as a tuple of dimensions
 * @template D - The data type
 * @template Dev - The device type ('cpu' | 'cuda' | 'mps')
 *
 * @example
 * ```ts
 * type Image = TensorType<[3, 224, 224], "float32", "cuda">;
 * type Embeddings = TensorType<[512, 768], "float16", "cpu">;
 * type Scalar = TensorType<[], "int32", "cpu">;
 * ```
 */
export interface TensorType<S extends Shape = Shape, D extends DTypeName = DTypeName, Dev extends DeviceType = DeviceType> {
  readonly shape: S
  readonly dtype: D
  readonly device: Dev
  readonly ndim: Rank<S>
}

/**
 * Utility type to ensure two tensors are on the same device
 * Returns the device type if they match, never otherwise
 *
 * @template Dev1 - Device of first tensor
 * @template Dev2 - Device of second tensor
 *
 * @example
 * ```ts
 * type Same = SameDevice<'cuda', 'cuda'>; // 'cuda'
 * type Different = SameDevice<'cpu', 'cuda'>; // never
 * ```
 */
export type SameDevice<Dev1 extends DeviceType, Dev2 extends DeviceType> = Dev1 extends Dev2
  ? Dev2 extends Dev1
    ? Dev1
    : never
  : never

/**
 * Computes the output shape of matrix multiplication
 *
 * Rules:
 * - 2D x 2D: [M, K] x [K, N] -> [M, N]
 * - ND x 2D: [..., M, K] x [K, N] -> [..., M, N]
 * - ND x ND: [..., M, K] x [..., K, N] -> [..., M, N] (batch dims must match)
 *
 * @template S1 - Shape of first tensor
 * @template S2 - Shape of second tensor
 *
 * @example
 * ```ts
 * type R1 = MatMulShape<[100, 50], [50, 20]>; // [100, 20]
 * type R2 = MatMulShape<[8, 100, 50], [50, 20]>; // [8, 100, 20]
 * type R3 = MatMulShape<[8, 100, 50], [8, 50, 20]>; // [8, 100, 20]
 * ```
 */
export type MatMulShape<S1 extends Shape, S2 extends Shape> =
  // Both must be at least 2D
  Rank<S1> extends 0 | 1
    ? never
    : Rank<S2> extends 0 | 1
      ? never
      : // Extract last two dimensions
        S1 extends readonly [...infer Batch1 extends readonly number[], infer M extends number, infer K1 extends number]
        ? S2 extends readonly [
            ...infer Batch2 extends readonly number[],
            infer K2 extends number,
            infer N extends number,
          ]
          ? // Check K dimensions match
            K1 extends K2
            ? // Handle batch dimensions
              Rank<S2> extends 2
              ? // S2 is 2D: just append to batch
                readonly [...Batch1, M, N]
              : // Both have batch: batch dims must broadcast
                BroadcastShape<Batch1, Batch2> extends infer B extends Shape
                ? readonly [...B, M, N]
                : never
            : never // K dimensions don't match
          : never
        : never

/**
 * Computes the shape after transposing dimensions D0 and D1
 *
 * @template S - Original shape
 * @template D0 - First dimension index
 * @template D1 - Second dimension index
 *
 * @example
 * ```ts
 * type R1 = TransposeShape<[2, 3, 4], 0, 2>; // [4, 3, 2]
 * type R2 = TransposeShape<[100, 50], 0, 1>; // [50, 100]
 * ```
 */
export type TransposeShape<S extends Shape, D0 extends number, D1 extends number> = D0 extends keyof S
  ? D1 extends keyof S
    ? {
        [K in keyof S]: K extends `${D0}` ? S[D1] : K extends `${D1}` ? S[D0] : S[K]
      } extends infer R extends readonly number[]
      ? R
      : never
    : never
  : never

/**
 * Validates that reshape preserves element count
 *
 * @template From - Original shape
 * @template To - Target shape
 *
 * @example
 * ```ts
 * type Valid = ReshapeValid<[2, 3, 4], [6, 4]>; // [6, 4]
 * type Invalid = ReshapeValid<[2, 3, 4], [5, 5]>; // never
 * ```
 */
export type ReshapeValid<From extends Shape, To extends Shape> =
  NumElements<From> extends NumElements<To> ? (NumElements<To> extends NumElements<From> ? To : never) : never

/**
 * Removes dimension D if it has size 1
 *
 * @template S - Original shape
 * @template D - Dimension index to squeeze (if undefined, squeeze all size-1 dims)
 *
 * @example
 * ```ts
 * type R1 = SqueezeShape<[1, 3, 1, 4], 0>; // [3, 1, 4]
 * type R2 = SqueezeShape<[1, 3, 1, 4], undefined>; // [3, 4] (squeeze all)
 * ```
 */
export type SqueezeShape<S extends Shape, D extends number | undefined = undefined> = D extends number
  ? D extends keyof S
    ? S[D] extends 1
      ? RemoveDim<S, D>
      : S // Can't squeeze non-1 dimension
    : never // Invalid dimension
  : SqueezeAllOnes<S>

/**
 * Helper to squeeze all dimensions of size 1
 * @internal
 */
type SqueezeAllOnes<S extends Shape> = S extends readonly [
  infer Head extends number,
  ...infer Tail extends readonly number[],
]
  ? Head extends 1
    ? SqueezeAllOnes<Tail>
    : readonly [Head, ...SqueezeAllOnes<Tail>]
  : readonly []

/**
 * Adds a dimension of size 1 at index D
 *
 * @template S - Original shape
 * @template D - Dimension index to insert at
 *
 * @example
 * ```ts
 * type R1 = UnsqueezeShape<[3, 4], 0>; // [1, 3, 4]
 * type R2 = UnsqueezeShape<[3, 4], 1>; // [3, 1, 4]
 * type R3 = UnsqueezeShape<[3, 4], 2>; // [3, 4, 1]
 * ```
 */
export type UnsqueezeShape<S extends Shape, D extends number> = D extends number
  ? D extends Rank<S>
    ? // Append to end
      readonly [...S, 1]
    : InsertDim<S, D, 1>
  : never

/**
 * Concatenates two tensors along dimension D
 *
 * @template S1 - Shape of first tensor
 * @template S2 - Shape of second tensor
 * @template D - Dimension to concatenate along
 *
 * @example
 * ```ts
 * type R1 = ConcatShape<[2, 3, 4], [2, 5, 4], 1>; // [2, 8, 4]
 * type R2 = ConcatShape<[10, 20], [30, 20], 0>; // [40, 20]
 * ```
 */
export type ConcatShape<S1 extends Shape, S2 extends Shape, D extends number> =
  Rank<S1> extends Rank<S2>
    ? D extends keyof S1
      ? D extends keyof S2
        ? // Check all other dimensions match
          ShapesMatchExcept<S1, S2, D> extends true
          ? S1[D] extends number
            ? S2[D] extends number
              ? // Sum the concat dimension
                SetDim<S1, D, Add<S1[D], S2[D]>>
              : never
            : never
          : never
        : never
      : never
    : never

/**
 * Helper to add two numbers at type level
 * @internal
 */
type Add<A extends number, B extends number> = A extends 0
  ? B
  : B extends 0
    ? A
    : number extends A
      ? number
      : number extends B
        ? number
        : number // Fallback for general case

/**
 * Check if all dimensions except D match between two shapes
 * @internal
 */
type ShapesMatchExcept<S1 extends Shape, S2 extends Shape, D extends number> = {
  [K in keyof S1]: K extends `${D}` ? true : K extends keyof S2 ? (S1[K] extends S2[K] ? true : false) : false
}[number] extends true
  ? true
  : false

/**
 * Computes the broadcast shape of two tensors
 *
 * Broadcasting rules (NumPy/PyTorch style):
 * - Dimensions are aligned from the right
 * - Each dimension pair must be equal or one must be 1
 * - Result dimension is the maximum of the two
 *
 * @template S1 - Shape of first tensor
 * @template S2 - Shape of second tensor
 *
 * @example
 * ```ts
 * type R1 = BroadcastShape<[1, 3, 4], [2, 1, 4]>; // [2, 3, 4]
 * type R2 = BroadcastShape<[5, 1], [3, 4]>; // [5, 3, 4]
 * type R3 = BroadcastShape<[], [3, 4]>; // [3, 4]
 * ```
 */
export type BroadcastShape<S1 extends Shape, S2 extends Shape> =
  BroadcastShapeImpl<Reverse<S1>, Reverse<S2>> extends infer R extends Shape ? Reverse<R> : never

/**
 * Implementation of broadcasting working from right to left
 * @internal
 */
type BroadcastShapeImpl<S1 extends Shape, S2 extends Shape> = S1 extends readonly [
  infer H1 extends number,
  ...infer T1 extends readonly number[],
]
  ? S2 extends readonly [infer H2 extends number, ...infer T2 extends readonly number[]]
    ? // Both have dimensions: check compatibility
      H1 extends H2
      ? readonly [H1, ...BroadcastShapeImpl<T1, T2>]
      : H1 extends 1
        ? readonly [H2, ...BroadcastShapeImpl<T1, T2>]
        : H2 extends 1
          ? readonly [H1, ...BroadcastShapeImpl<T1, T2>]
          : never // Incompatible dimensions
    : // S2 exhausted: use remaining S1
      S1
  : S2 extends readonly [infer _H2 extends number, ...infer _T2 extends readonly number[]]
    ? // S1 exhausted: use remaining S2
      S2
    : // Both exhausted
      readonly []

/**
 * Computes the shape after reducing along dimension D
 *
 * @template S - Original shape
 * @template D - Dimension to reduce
 * @template KeepDim - Whether to keep dimension as size 1
 *
 * @example
 * ```ts
 * type R1 = ReduceShape<[2, 3, 4], 1, false>; // [2, 4]
 * type R2 = ReduceShape<[2, 3, 4], 1, true>; // [2, 1, 4]
 * ```
 */
export type ReduceShape<S extends Shape, D extends number, KeepDim extends boolean = false> = KeepDim extends true
  ? SetDim<S, D, 1>
  : RemoveDim<S, D>

/**
 * Computes the shape after permuting dimensions
 *
 * @template S - Original shape
 * @template Perm - Permutation of dimension indices
 *
 * @example
 * ```ts
 * type R1 = PermuteShape<[2, 3, 4], [2, 0, 1]>; // [4, 2, 3]
 * type R2 = PermuteShape<[2, 3, 4, 5], [0, 2, 1, 3]>; // [2, 4, 3, 5]
 * ```
 */
export type PermuteShape<S extends Shape, Perm extends readonly number[]> =
  Rank<S> extends Rank<Perm>
    ? {
        [K in keyof Perm]: Perm[K] extends keyof S ? S[Perm[K]] : never
      } extends infer R extends readonly number[]
      ? R
      : never
    : never

/**
 * Computes the shape after expanding dimension D to size N
 * Dimension D must be size 1 in the original shape
 *
 * @template S - Original shape
 * @template D - Dimension index to expand
 * @template N - New size for dimension D
 *
 * @example
 * ```ts
 * type R1 = ExpandShape<[1, 3, 4], 0, 8>; // [8, 3, 4]
 * type R2 = ExpandShape<[2, 1, 4], 1, 10>; // [2, 10, 4]
 * ```
 */
export type ExpandShape<S extends Shape, D extends number, N extends number> = D extends keyof S
  ? S[D] extends 1
    ? SetDim<S, D, N>
    : never // Can only expand size-1 dimensions
  : never

/**
 * Computes the shape of a slice operation
 *
 * @template S - Original shape
 * @template D - Dimension to slice
 * @template Start - Start index (inclusive)
 * @template End - End index (exclusive)
 * @template Step - Step size
 *
 * For simplicity, we return number for the sliced dimension
 * since computing exact size requires complex type-level arithmetic
 */
export type SliceShape<
  S extends Shape,
  D extends number,
  _Start extends number = 0,
  _End extends number = number,
  _Step extends number = 1,
> = D extends keyof S ? SetDim<S, D, number> : never

/**
 * Flattens a shape from dimension Start to End into a single dimension
 *
 * @template S - Original shape
 * @template Start - Start dimension (inclusive)
 * @template End - End dimension (exclusive)
 *
 * @example
 * ```ts
 * type R1 = FlattenShape<[2, 3, 4, 5], 1, 3>; // [2, 12, 5]
 * type R2 = FlattenShape<[2, 3, 4], 0, 3>; // [24]
 * ```
 */
export type FlattenShape<S extends Shape, Start extends number = 0, End extends number = Rank<S>> = S extends readonly [
  ...infer Prefix,
  ...infer Middle,
  ...infer Suffix,
]
  ? Prefix extends { length: Start }
    ? Middle extends Shape
      ? End extends number
        ? readonly [...Prefix, NumElements<Middle>, ...Suffix]
        : never
      : never
    : never
  : never
