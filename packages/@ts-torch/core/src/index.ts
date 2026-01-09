/**
 * @ts-torch/core - Core tensor operations & FFI bindings
 *
 * This package provides the foundational tensor operations and Foreign Function Interface (FFI)
 * bindings for ts-torch, enabling high-performance tensor computations backed by native implementations.
 */

// ===============================
// Type Exports
// ===============================

export type { Shape, ValidDim, Dim } from "./types/shape.js";
export type {
  DType as DTypeType,
  DTypeName,
  DTypeToTypedArray,
  DTypeElement,
} from "./types/dtype.js";
export type { TensorType, MatMulShape, TransposeShape } from "./types/tensor.js";

// ===============================
// Runtime DType Values
// ===============================

// Export the full DType namespace
import { DType as DTypeNamespace } from "./types/dtype.js";
export { DTypeNamespace as DType };

// Individual dtype constants for direct access
export const float16 = DTypeNamespace.float16;
export const float32 = DTypeNamespace.float32;
export const float64 = DTypeNamespace.float64;
export const int32 = DTypeNamespace.int32;
export const int64 = DTypeNamespace.int64;
export const bool = DTypeNamespace.bool;
export const bfloat16 = DTypeNamespace.bfloat16;

// ===============================
// Tensor Class
// ===============================

export { Tensor } from "./tensor/tensor.js";

// ===============================
// Memory Management
// ===============================

export { run, runAsync, inScope, scopeDepth } from "./memory/scope.js";

// ===============================
// Debug Mode
// ===============================

export { DebugMode } from "./debug.js";

// ===============================
// Device
// ===============================

/**
 * Device class for specifying tensor compute location
 *
 * @example
 * ```ts
 * const cpuDevice = Device.cpu();
 * const gpuDevice = Device.cuda(0);
 * const mpsDevice = Device.mps();
 * ```
 */
export class Device {
  private constructor(
    public readonly type: "cpu" | "cuda" | "mps",
    public readonly index: number,
  ) {}

  static cpu(): Device {
    return new Device("cpu", 0);
  }

  static cuda(index = 0): Device {
    return new Device("cuda", index);
  }

  static mps(): Device {
    return new Device("mps", 0);
  }

  toString(): string {
    return this.type === "cpu" ? "cpu" : `${this.type}:${this.index}`;
  }
}

// ===============================
// torch Namespace
// ===============================

import { run as runScope, runAsync as runAsyncScope } from "./memory/scope.js";
import { Tensor } from "./tensor/tensor.js";
import {
  zeros as zerosFactory,
  ones as onesFactory,
  randn as randnFactory,
  empty as emptyFactory,
  fromArray as fromArrayFactory,
  createArange,
  createTensorFromData,
} from "./tensor/factory.js";
import type { Shape } from "./types/shape.js";
import type { DType } from "./types/dtype.js";
import { getLib } from "./ffi/loader.js";

/**
 * Main torch namespace providing PyTorch-like API
 *
 * @example
 * ```ts
 * import { torch } from '@ts-torch/core';
 *
 * const x = torch.zeros([2, 3]);
 * const y = torch.ones([2, 3]);
 * const z = x.add(y);
 *
 * const result = torch.run(() => {
 *   const a = torch.randn([100, 100]);
 *   const b = torch.randn([100, 100]);
 *   const c = a.matmul(b);
 *   return c.escape();
 * });
 * ```
 */
export const torch = {
  // ==================== Memory Scopes ====================

  /**
   * Execute code with scoped memory management
   *
   * @template T - Return type
   * @param fn - Function to execute
   * @returns Result of function
   */
  run: runScope,

  /**
   * Execute async code with scoped memory management
   *
   * @template T - Return type
   * @param fn - Async function to execute
   * @returns Promise of function result
   */
  runAsync: runAsyncScope,

  // ==================== Tensor Creation ====================

  /**
   * Create a tensor filled with zeros
   *
   * @template S - Shape type
   * @template D - DType type
   * @param shape - Tensor shape
   * @param dtype - Data type (default: float32)
   * @returns New tensor filled with zeros
   *
   * @example
   * ```ts
   * const t = torch.zeros([2, 3] as const);
   * const t2 = torch.zeros([10, 20] as const, torch.float64);
   * ```
   */
  zeros<S extends Shape, D extends DType<string> = typeof float32>(
    shape: S,
    dtype?: D,
  ): Tensor<S, D> {
    return zerosFactory(shape, dtype);
  },

  /**
   * Create a tensor filled with ones
   *
   * @template S - Shape type
   * @template D - DType type
   * @param shape - Tensor shape
   * @param dtype - Data type (default: float32)
   * @returns New tensor filled with ones
   *
   * @example
   * ```ts
   * const t = torch.ones([2, 3] as const);
   * ```
   */
  ones<S extends Shape, D extends DType<string> = typeof float32>(
    shape: S,
    dtype?: D,
  ): Tensor<S, D> {
    return onesFactory(shape, dtype);
  },

  /**
   * Create a tensor with random normal distribution
   *
   * @template S - Shape type
   * @template D - DType type
   * @param shape - Tensor shape
   * @param dtype - Data type (default: float32)
   * @returns New tensor with random values
   *
   * @example
   * ```ts
   * const t = torch.randn([100, 50] as const);
   * ```
   */
  randn<S extends Shape, D extends DType<string> = typeof float32>(
    shape: S,
    dtype?: D,
  ): Tensor<S, D> {
    return randnFactory(shape, dtype);
  },

  /**
   * Create an uninitialized tensor
   *
   * @template S - Shape type
   * @template D - DType type
   * @param shape - Tensor shape
   * @param dtype - Data type (default: float32)
   * @returns New uninitialized tensor
   *
   * @example
   * ```ts
   * const t = torch.empty([1000, 1000] as const);
   * ```
   */
  empty<S extends Shape, D extends DType<string> = typeof float32>(
    shape: S,
    dtype?: D,
  ): Tensor<S, D> {
    return emptyFactory(shape, dtype);
  },

  /**
   * Create tensor from data array
   *
   * @template S - Shape type
   * @template D - DType type
   * @param data - Flat array of data
   * @param shape - Tensor shape
   * @param dtype - Data type (default: float32)
   * @returns New tensor with data
   *
   * @example
   * ```ts
   * const t = torch.tensor(
   *   [1, 2, 3, 4, 5, 6],
   *   [2, 3] as const
   * );
   * ```
   */
  tensor<S extends Shape, D extends DType<string> = typeof float32>(
    data: number[] | Float32Array | Float64Array,
    shape: S,
    dtype?: D,
  ): Tensor<S, D> {
    return fromArrayFactory(data, shape, dtype);
  },

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
   * const t = torch.arange(0, 10); // [0, 1, 2, ..., 9]
   * const t2 = torch.arange(0, 1, 0.1); // [0.0, 0.1, ..., 0.9]
   * ```
   */
  arange<D extends DType<string> = typeof float32>(
    start: number,
    end: number,
    step?: number,
    dtype?: D,
  ): Tensor<readonly [number], D> {
    return createArange(start, end, step, dtype);
  },

  /**
   * Create tensor from nested arrays (auto-infer shape)
   *
   * @template D - DType type
   * @param data - Nested array data
   * @param dtype - Data type (default: float32)
   * @returns New tensor
   *
   * @example
   * ```ts
   * const t = torch.from([[1, 2], [3, 4]]);
   * ```
   */
  from<D extends DType<string> = typeof float32>(
    data: number | number[] | number[][] | number[][][] | number[][][][],
    dtype?: D,
  ): Tensor<readonly number[], D> {
    return createTensorFromData(data, dtype);
  },

  // ==================== CUDA Utilities ====================

  /**
   * CUDA utilities and device management
   */
  cuda: {
    /**
     * Check if CUDA is available on this system
     *
     * @returns True if CUDA is available
     *
     * @example
     * ```ts
     * if (torch.cuda.isAvailable()) {
     *   console.log('CUDA is available');
     * }
     * ```
     */
    isAvailable(): boolean {
      try {
        const lib = getLib();
        return lib.symbols.ts_cuda_is_available() !== 0; // Convert i32 to boolean
      } catch {
        return false;
      }
    },

    /**
     * Get number of CUDA devices
     *
     * @returns Number of available CUDA devices
     *
     * @example
     * ```ts
     * const numDevices = torch.cuda.deviceCount();
     * console.log(`Found ${numDevices} CUDA devices`);
     * ```
     */
    deviceCount(): number {
      try {
        const lib = getLib();
        return lib.symbols.ts_cuda_device_count();
      } catch {
        return 0;
      }
    },

    /**
     * Synchronize CUDA device (wait for all operations to complete)
     *
     * @param device - Device index (default: 0)
     *
     * @example
     * ```ts
     * torch.cuda.synchronize();
     * ```
     */
    synchronize(_device = 0): void {
      // TODO: Implement when FFI symbol is available
      console.warn("torch.cuda.synchronize() not yet implemented");
    },
  },

  // ==================== Version Info ====================

  /**
   * Get library version information
   *
   * @returns Version object with major, minor, patch
   *
   * @example
   * ```ts
   * const version = torch.version();
   * console.log(`ts-torch v${version.major}.${version.minor}.${version.patch}`);
   * ```
   */
  version(): { major: number; minor: number; patch: number } {
    // TODO: Get from native library when available
    return { major: 0, minor: 1, patch: 0 };
  },

  // ==================== Device Management ====================

  /**
   * Device class for convenience access
   */
  Device,

  // ==================== DType Constants ====================

  /**
   * Data type constants for convenience
   */
  float16,
  float32,
  float64,
  int32,
  int64,
  bool,
  bfloat16,
};

// ===============================
// Default Export
// ===============================

/**
 * Default export for convenience
 *
 * @example
 * ```ts
 * import torch from '@ts-torch/core';
 *
 * const x = torch.zeros([2, 3]);
 * ```
 */
export default torch;
