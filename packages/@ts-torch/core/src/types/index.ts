/**
 * ts-torch type system
 *
 * Advanced TypeScript types for compile-time tensor shape and dtype checking.
 * Enables catching shape mismatches and type errors at compile time rather than runtime.
 *
 * @example
 * ```ts
 * import type { TensorType, MatMulShape, DTypeName } from '@ts-torch/core/types';
 *
 * // Define tensor types
 * type Matrix = TensorType<[100, 50], "float32">;
 * type Vector = TensorType<[50, 1], "float32">;
 *
 * // Compute result shape at compile time
 * type Result = MatMulShape<[100, 50], [50, 1]>; // [100, 1]
 *
 * // This would be a compile error:
 * type Invalid = MatMulShape<[100, 50], [60, 1]>; // never (incompatible)
 * ```
 *
 * @module @ts-torch/core/types
 */

// Data types
export type {
  DType,
  DTypeName,
  DTypeToTypedArray,
  DTypeElement,
  PromoteDType,
} from "./dtype";

export {
  DType as DTypeConstants,
  DTypeValue,
  BytesPerElement,
  isDTypeName,
  getDType,
} from "./dtype";

// Shape types and operations
export type {
  Shape,
  Dim,
  ValidDim,
  ExtractDimLabels,
  NumElements,
  ValidateShape,
  ShapeEqual,
  Rank,
  GetDim,
  SetDim,
  Reverse,
  Concat,
  RemoveDim,
  InsertDim,
} from "./shape";

// Tensor types and operations
export type {
  TensorType,
  MatMulShape,
  TransposeShape,
  ReshapeValid,
  SqueezeShape,
  UnsqueezeShape,
  ConcatShape,
  BroadcastShape,
  ReduceShape,
  PermuteShape,
  ExpandShape,
  SliceShape,
  FlattenShape,
} from "./tensor";

// Import for internal use
import type { TensorType } from "./tensor";
import type { DTypeElement as DTypeElementInternal, DTypeName as DTypeNameInternal } from "./dtype";
import type { Shape as ShapeInternal } from "./shape";

/**
 * Utility type to extract the element type from a TensorType
 *
 * @template T - The TensorType
 */
export type ElementType<T extends TensorType> =
  DTypeElementInternal<T["dtype"]>;

/**
 * Utility type to extract the shape from a TensorType
 *
 * @template T - The TensorType
 */
export type ExtractShape<T extends TensorType> = T["shape"];

/**
 * Utility type to extract the dtype from a TensorType
 *
 * @template T - The TensorType
 */
export type ExtractDType<T extends TensorType> = T["dtype"];

/**
 * Utility type to check if a shape is compatible for broadcasting
 *
 * @template S1 - First shape
 * @template S2 - Second shape
 */
export type IsBroadcastable<
  S1 extends ShapeInternal,
  S2 extends ShapeInternal
> = import("./tensor").BroadcastShape<S1, S2> extends never ? false : true;

/**
 * Utility type to check if two shapes can be matrix multiplied
 *
 * @template S1 - First shape
 * @template S2 - Second shape
 */
export type IsMatMulCompatible<
  S1 extends ShapeInternal,
  S2 extends ShapeInternal
> = import("./tensor").MatMulShape<S1, S2> extends never ? false : true;

/**
 * Legacy compatibility exports
 * @deprecated Use the new type system above
 */

/**
 * Tensor device type
 */
export type Device = 'cpu' | 'cuda' | 'mps';

/**
 * Tensor stride information
 */
export type Stride = readonly number[];

/**
 * Tensor metadata
 */
export interface TensorMetadata {
  shape: ShapeInternal;
  dtype: DTypeNameInternal;
  device: Device;
  stride: Stride;
  requiresGrad: boolean;
}

/**
 * Tensor creation options
 */
export interface TensorOptions {
  dtype?: DTypeNameInternal;
  device?: Device;
  requiresGrad?: boolean;
}

/**
 * Gradient computation mode
 */
export interface GradMode {
  enabled: boolean;
}
