/**
 * @ts-torch/core - Core tensor operations & FFI bindings
 *
 * This package provides the foundational tensor operations and Foreign Function Interface (FFI)
 * bindings for ts-torch, enabling high-performance tensor computations backed by native implementations.
 *
 * ## Declarative API
 *
 * ```ts
 * import { device, Data, run } from '@ts-torch/core'
 *
 * const cuda = device.cuda(0)
 *
 * // Create tensors directly on GPU
 * const x = cuda.zeros([784, 128])
 * const y = cuda.randn([128, 10])
 *
 * // Memory-scoped operations
 * run(() => {
 *   const z = x.matmul(y)
 *   return z.escape()
 * })
 * ```
 */

// ===============================
// Internal Imports for CUDA Utilities
// ===============================

import { getLib } from './ffi/loader.js'
import { Logger } from './logger.js'

// ===============================
// Type Exports
// ===============================

export type { Shape, ValidDim, Dim } from './types/shape.js'
export type { DType as DTypeType, DTypeName, DTypeToTypedArray, DTypeElement } from './types/dtype.js'
export type { TensorType, MatMulShape, TransposeShape, DeviceType, SameDevice } from './types/tensor.js'

// ===============================
// Runtime DType Values
// ===============================

import { DType as DTypeNamespace } from './types/dtype.js'
export { DTypeNamespace as DType }

// Individual dtype constants for direct access
export const float16 = DTypeNamespace.float16
export const float32 = DTypeNamespace.float32
export const float64 = DTypeNamespace.float64
export const int32 = DTypeNamespace.int32
export const int64 = DTypeNamespace.int64
export const bool = DTypeNamespace.bool
export const bfloat16 = DTypeNamespace.bfloat16

// ===============================
// Tensor Class & Operations
// ===============================

export { Tensor } from './tensor/tensor.js'
export { cat } from './tensor/factory.js'

// ===============================
// Memory Management
// ===============================

export { run, runAsync, inScope, scopeDepth } from './memory/scope.js'

// ===============================
// Debug Mode
// ===============================

export { DebugMode } from './debug.js'

// ===============================
// Unified Logger
// ===============================

export { Logger, verboseToLevel } from './logger.js'
export type { LogLevel, LogHandler, LoggerConfig } from './logger.js'

// ===============================
// Declarative Device Context
// ===============================

export { device, DeviceContext } from './device/index.js'

// ===============================
// Declarative Data Pipeline
// ===============================

export { Data, DataPipeline } from './data/index.js'
export type { Dataset, BatchableDataset, TensorPair, Batch } from './data/index.js'

// ===============================
// CUDA Utilities
// ===============================

/**
 * CUDA utilities namespace
 */
export const cuda = {
  /**
   * Check if CUDA is available on this system
   *
   * @returns True if CUDA is available
   *
   * @example
   * ```ts
   * if (cuda.isAvailable()) {
   *   const gpu = device.cuda(0)
   * }
   * ```
   */
  isAvailable(): boolean {
    try {
      const lib = getLib()
      return (lib.ts_cuda_is_available() as number) !== 0
    } catch {
      return false
    }
  },

  /**
   * Get number of CUDA devices
   *
   * @returns Number of available CUDA devices
   *
   * @example
   * ```ts
   * console.log(`Found ${cuda.deviceCount()} CUDA devices`)
   * ```
   */
  deviceCount(): number {
    try {
      const lib = getLib()
      return lib.ts_cuda_device_count() as number
    } catch {
      return 0
    }
  },

  /**
   * Synchronize CUDA device (wait for all operations to complete)
   *
   * @param deviceIndex - Device index (default: 0)
   */
  synchronize(_deviceIndex = 0): void {
    // TODO: Implement when FFI symbol is available
    Logger.warn('cuda.synchronize() not yet implemented')
  },
}

// ===============================
// Validation Utilities
// ===============================

export {
  ValidationError,
  validateFinite,
  validatePositive,
  validateNonNegative,
  validatePositiveInt,
  validateNonNegativeInt,
  validateRange,
  validateProbability,
  validateShapesCompatible,
  validateMatmulShapes,
  validateDimension,
  validateReshape,
  validateShape,
  validateDtype,
  checkNull,
  validateLinearParams,
  validateConv2dParams,
  validatePoolingParams,
  validateNormParams,
  validateSGDParams,
  validateAdamParams,
  validateRMSpropParams,
  validateDataLoaderParams,
  validateDatasetNotEmpty,
  validateScalar,
  validateNonZero,
} from './validation/index.js'

