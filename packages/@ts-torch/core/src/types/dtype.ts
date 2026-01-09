/**
 * Data type definitions for ts-torch tensors
 *
 * Provides branded types for compile-time type safety and runtime dtype constants
 * matching common ML framework conventions (PyTorch, NumPy).
 */

/**
 * Unique brand symbol for DType discrimination
 */
declare const DTypeBrand: unique symbol;

/**
 * Branded DType type for compile-time type safety
 *
 * @template Name - The dtype name (e.g., "float32", "int64")
 *
 * @example
 * ```ts
 * const dtype: DType<"float32"> = DType.float32;
 * ```
 */
export type DType<Name extends string = string> = {
  readonly [DTypeBrand]: Name;
  readonly name: Name;
  readonly value: number;
  readonly bytes: number;
};

/**
 * Runtime dtype constant values matching C enum convention
 * These values are used for interop with native/WASM backends
 */
export const enum DTypeValue {
  float32 = 0,
  float64 = 1,
  int32 = 2,
  int64 = 3,
  float16 = 4,
  bfloat16 = 5,
  bool = 6,
}

/**
 * Concrete DType instances
 */
export const DType = {
  float16: {
    name: "float16",
    value: DTypeValue.float16,
    bytes: 2,
  } as DType<"float16">,

  float32: {
    name: "float32",
    value: DTypeValue.float32,
    bytes: 4,
  } as DType<"float32">,

  float64: {
    name: "float64",
    value: DTypeValue.float64,
    bytes: 8,
  } as DType<"float64">,

  int32: {
    name: "int32",
    value: DTypeValue.int32,
    bytes: 4,
  } as DType<"int32">,

  int64: {
    name: "int64",
    value: DTypeValue.int64,
    bytes: 8,
  } as DType<"int64">,

  bool: {
    name: "bool",
    value: DTypeValue.bool,
    bytes: 1,
  } as DType<"bool">,

  bfloat16: {
    name: "bfloat16",
    value: DTypeValue.bfloat16,
    bytes: 2,
  } as DType<"bfloat16">,
} as const;

/**
 * Union type of all supported dtype names
 */
export type DTypeName = keyof typeof DType;

/**
 * Maps DType to the corresponding TypedArray type
 *
 * @template D - The DType name
 *
 * @remarks
 * Note: float16 and bfloat16 use Uint16Array as backing storage,
 * with conversion logic handled at runtime.
 */
export type DTypeToTypedArray<D extends DTypeName> =
  D extends "float32" ? Float32Array :
  D extends "float64" ? Float64Array :
  D extends "int32" ? Int32Array :
  D extends "int64" ? BigInt64Array :
  D extends "float16" ? Uint16Array :
  D extends "bfloat16" ? Uint16Array :
  D extends "bool" ? Uint8Array :
  never;

/**
 * Maps DType to the element type (number, bigint, boolean)
 *
 * @template D - The DType name
 */
export type DTypeElement<D extends DTypeName> =
  D extends "float16" | "float32" | "float64" | "bfloat16" ? number :
  D extends "int32" ? number :
  D extends "int64" ? bigint :
  D extends "bool" ? boolean :
  never;

/**
 * Bytes per element for each dtype
 */
export const BytesPerElement: Record<DTypeName, number> = {
  float16: 2,
  float32: 4,
  float64: 8,
  int32: 4,
  int64: 8,
  bool: 1,
  bfloat16: 2,
} as const;

/**
 * Type guard to check if a string is a valid dtype name
 *
 * @param name - The string to check
 * @returns True if name is a valid dtype name
 */
export function isDTypeName(name: string): name is DTypeName {
  return name in DType;
}

/**
 * Get DType by name with runtime validation
 *
 * @param name - The dtype name
 * @returns The DType constant
 * @throws Error if name is not a valid dtype
 */
export function getDType<N extends DTypeName>(name: N): DType<N> {
  if (!isDTypeName(name)) {
    throw new Error(`Invalid dtype name: ${name}`);
  }
  return DType[name] as DType<N>;
}

/**
 * Determine the common dtype for mixed operations
 * Uses standard type promotion rules (int -> float, lower precision -> higher)
 *
 * @template D1 - First dtype name
 * @template D2 - Second dtype name
 */
export type PromoteDType<D1 extends DTypeName, D2 extends DTypeName> =
  D1 extends D2 ? D1 :
  // float64 promotes all other types
  D1 extends "float64" ? "float64" :
  D2 extends "float64" ? "float64" :
  // float32 promotes all except float64
  D1 extends "float32" ? "float32" :
  D2 extends "float32" ? "float32" :
  // bfloat16 promotes to float32 when mixed with float types
  [D1, D2] extends [infer A, infer B] ?
    A extends "bfloat16" ?
      B extends "float16" | "bfloat16" ? "float32" : B extends DTypeName ? B : never
    : B extends "bfloat16" ?
      A extends "float16" | "bfloat16" ? "float32" : A extends DTypeName ? A : never
    : never
  : never;
