/**
 * FFI module for ts-torch
 * Provides low-level bindings to the native C library via Node-API
 *
 * @module @ts-torch/core/ffi
 *
 * @example
 * ```ts
 * import { getLib } from '@ts-torch/core/ffi';
 *
 * const lib = getLib();
 * const shape = new BigInt64Array([2, 3]);
 *
 * const handle = lib.ts_tensor_zeros(shape, 0, 0, 0); // (shape, dtype, device, deviceIndex)
 * ```
 */

// Re-export library loader
export { getLib, closeLib, getLibraryPath, getPlatformPackage, type NativeModule } from './loader.js'

// Re-export error handling
export {
  TorchError,
  ErrorCode,
  createError,
  checkError,
  checkErrorBuffer,
  withError,
  checkNull,
  validateShape,
  validateDtype,
  ERROR_STRUCT_SIZE,
  type Pointer,
  type ErrorBuffer,
} from './error.js'

// Re-export buffer pooling
export { errorPool, shapeCache } from './buffer-pool.js'

// Re-export koffi for direct access (used for address/decode in cat/stack operations)
export { default as koffi } from 'koffi'
