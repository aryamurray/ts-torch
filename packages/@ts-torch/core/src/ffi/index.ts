/**
 * FFI module for ts-torch
 * Provides low-level bindings to the native C library
 *
 * @module @ts-torch/core/ffi
 *
 * @example
 * ```ts
 * import { getLib, createError, checkError, TensorHandle } from '@ts-torch/core/ffi';
 *
 * const lib = getLib();
 * const shape = new BigInt64Array([2, 3]);
 *
 * const handle = withError(err => lib.ts_tensor_zeros(
 *   shape.buffer,
 *   2, // ndim
 *   0, // dtype (f32)
 *   0, // device
 *   0, // device_index
 *   err
 * ));
 * ```
 */

// Re-export library loader
export { getLib, closeLib, getLibraryPath, getPlatformPackage, type KoffiLibrary } from './loader.js'

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

// Re-export FFI symbols and types
export { FFI_SYMBOLS, type TensorHandle, type FFISymbols } from './symbols.js'

// Re-export koffi for direct access if needed
export { default as koffi } from 'koffi'
