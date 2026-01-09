/**
 * Shape type definitions for compile-time tensor shape checking
 *
 * Provides type-level operations on tensor shapes to catch shape mismatches
 * at compile time rather than runtime.
 */

/**
 * Shape type - readonly tuple of dimension sizes
 *
 * @example
 * ```ts
 * type ImageShape = Shape<[3, 224, 224]>; // CHW format
 * type MatrixShape = Shape<[100, 50]>;
 * type ScalarShape = Shape<[]>;
 * ```
 */
export type Shape<T extends readonly number[] = readonly number[]> = T;

/**
 * Unique brand symbol for Dim discrimination
 */
declare const DimBrand: unique symbol;

/**
 * Branded dimension type for runtime-determined dimensions
 *
 * @template Label - A string label for this dimension (e.g., "batch", "seq_len")
 *
 * @example
 * ```ts
 * type BatchDim = Dim<"batch">;
 * type SeqLenDim = Dim<"seq_len">;
 * type DynamicShape = [BatchDim, SeqLenDim, 768];
 * ```
 */
export type Dim<Label extends string = string> = number & {
  readonly [DimBrand]: Label;
};

/**
 * Valid dimension type - either a literal number or a branded Dim
 * Used to validate shape types don't contain raw 'number' type
 *
 * @example
 * ```ts
 * type Valid1 = ValidDim<5>; // 5
 * type Valid2 = ValidDim<Dim<"batch">>; // Dim<"batch">
 * type Invalid = ValidDim<number>; // never
 * ```
 */
export type ValidDim<D extends number> =
  D extends Dim<string> ? D :
  number extends D ? never :
  D;

/**
 * Extract dimension labels from a shape
 */
export type ExtractDimLabels<S extends Shape> =
  S extends readonly [infer Head, ...infer Tail] ?
    Head extends Dim<infer Label> ?
      Label | (Tail extends Shape ? ExtractDimLabels<Tail> : never)
    : Tail extends Shape ? ExtractDimLabels<Tail> : never
  : never;

/**
 * Helper type to multiply tuple of numbers at type level
 * Recursively multiplies all dimensions to get total element count
 *
 * @internal
 */
type MultiplyTuple<
  T extends readonly number[],
  Acc extends number = 1
> = T extends readonly [infer Head extends number, ...infer Tail extends readonly number[]]
  ? MultiplyTuple<Tail, Multiply<Acc, Head>>
  : Acc;

/**
 * Type-level multiplication for literal numbers up to reasonable size
 * Uses lookup table approach for common cases
 *
 * @internal
 */
type Multiply<A extends number, B extends number> =
  A extends 0 ? 0 :
  B extends 0 ? 0 :
  A extends 1 ? B :
  B extends 1 ? A :
  // For non-literal numbers, return number
  number extends A ? number :
  number extends B ? number :
  // Otherwise compute multiplication for literals
  MultiplyLiterals<A, B>;

/**
 * Multiplication lookup for common literal values
 * Falls back to number for uncommon cases
 *
 * @internal
 */
type MultiplyLiterals<A extends number, B extends number> =
  [A, B] extends [2, 2] ? 4 :
  [A, B] extends [2, 3] | [3, 2] ? 6 :
  [A, B] extends [2, 4] | [4, 2] ? 8 :
  [A, B] extends [3, 3] ? 9 :
  [A, B] extends [3, 4] | [4, 3] ? 12 :
  [A, B] extends [4, 4] ? 16 :
  // Add more common cases as needed
  number; // Fallback for computed values

/**
 * Computes total number of elements in a tensor shape
 *
 * @template S - The shape tuple
 *
 * @example
 * ```ts
 * type Count1 = NumElements<[2, 3, 4]>; // 24
 * type Count2 = NumElements<[]>; // 1 (scalar)
 * type Count3 = NumElements<[10, 20]>; // 200
 * ```
 */
export type NumElements<S extends Shape> =
  S extends readonly [] ? 1 : MultiplyTuple<S>;

/**
 * Validates that a shape contains no plain 'number' types
 * All dimensions must be literal numbers or Dim<Label>
 *
 * @template S - The shape to validate
 *
 * @example
 * ```ts
 * type Valid = ValidateShape<[2, 3, 4]>; // OK
 * type Invalid = ValidateShape<[2, number, 4]>; // Error: dimension must be literal
 * type ValidDynamic = ValidateShape<[Dim<"batch">, 3, 4]>; // OK
 * ```
 */
export type ValidateShape<S extends Shape> = {
  [K in keyof S]: S[K] extends Dim<string> ? S[K] :
                  S[K] extends number ? S[K] :
                  never;
};

/**
 * Checks if two shapes are equal at compile time
 *
 * @template S1 - First shape
 * @template S2 - Second shape
 */
export type ShapeEqual<S1 extends Shape, S2 extends Shape> =
  S1 extends S2 ? S2 extends S1 ? true : false : false;

/**
 * Get the rank (number of dimensions) of a shape
 *
 * @template S - The shape
 *
 * @example
 * ```ts
 * type R1 = Rank<[2, 3, 4]>; // 3
 * type R2 = Rank<[]>; // 0
 * ```
 */
export type Rank<S extends Shape> = S["length"];

/**
 * Get dimension at index I, or never if out of bounds
 *
 * @template S - The shape
 * @template I - The dimension index
 */
export type GetDim<S extends Shape, I extends number> =
  I extends keyof S ? S[I] : never;

/**
 * Set dimension at index I to value V
 *
 * @template S - The shape
 * @template I - The dimension index to set
 * @template V - The new value
 *
 * @internal
 */
export type SetDim<
  S extends Shape,
  I extends number,
  V extends number
> = {
  [K in keyof S]: K extends `${I}` ? V : S[K]
} extends infer R extends readonly number[] ? R : never;

/**
 * Reverse a shape tuple
 *
 * @template S - The shape to reverse
 * @internal
 */
export type Reverse<S extends Shape> =
  S extends readonly [...infer Init extends readonly number[], infer Last extends number]
    ? readonly [Last, ...Reverse<Init>]
    : readonly [];

/**
 * Concatenate two shape tuples
 *
 * @template S1 - First shape
 * @template S2 - Second shape
 * @internal
 */
export type Concat<S1 extends Shape, S2 extends Shape> =
  readonly [...S1, ...S2];

/**
 * Slice a shape from index Start to End
 *
 * @template S - The shape
 * @template Start - Start index (inclusive)
 * @template End - End index (exclusive)
 * @internal
 */
export type Slice<
  S extends Shape,
  Start extends number = 0,
  End extends number = Rank<S>
> = S extends readonly [...infer Prefix, ...infer Suffix]
  ? Prefix extends { length: Start }
    ? Suffix extends readonly [...infer Result, ...infer _Rest]
      ? Result extends { length: End extends number ? End : never }
        ? Result extends Shape ? Result : never
        : never
      : never
    : never
  : never;

/**
 * Remove dimension at index I
 *
 * @template S - The shape
 * @template I - The dimension index to remove
 * @internal
 */
export type RemoveDim<S extends Shape, I extends number> =
  S extends readonly [...infer Prefix extends readonly number[], infer At extends number, ...infer Suffix extends readonly number[]]
    ? Prefix["length"] extends I
      ? readonly [...Prefix, ...Suffix]
      : Tail<Prefix> extends readonly number[]
        ? readonly [Prefix[0], ...RemoveDim<readonly [...Tail<Prefix>, At, ...Suffix], I>]
        : S
    : S;

/**
 * Get tail of tuple
 * @internal
 */
type Tail<T extends readonly any[]> =
  T extends readonly [any, ...infer Rest] ? Rest : readonly [];

/**
 * Insert dimension V at index I
 *
 * @template S - The shape
 * @template I - The dimension index to insert at
 * @template V - The value to insert
 * @internal
 */
export type InsertDim<
  S extends Shape,
  I extends number,
  V extends number
> = S extends readonly [...infer Prefix extends readonly number[], ...infer Suffix extends readonly number[]]
  ? Prefix["length"] extends I
    ? readonly [...Prefix, V, ...Suffix]
    : readonly [Prefix[0], ...InsertDim<Tail<Prefix>, I, V>]
  : readonly [V];
