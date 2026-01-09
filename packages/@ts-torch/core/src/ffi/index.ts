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
 * const err = createError();
 * const shape = new BigInt64Array([2, 3]);
 *
 * const handle = lib.symbols.ts_tensor_zeros(
 *   ptr(shape),
 *   2, // ndim
 *   0, // dtype (f32)
 *   false, // requires_grad
 *   err
 * );
 *
 * checkError(err);
 * ```
 */

// Re-export library loader
export { getLib, closeLib, getLibraryPath, getPlatformPackage } from './loader.js'

// Re-export error handling
export {
  TorchError,
  ErrorCode,
  createError,
  checkError,
  withError,
  checkNull,
  validateShape,
  validateDtype,
  ERROR_STRUCT_SIZE,
} from './error.js'

// Re-export FFI symbols and types
export { FFI_SYMBOLS, type TensorHandle, type FFISymbols } from './symbols.js'

// Re-export useful Bun FFI utilities
export { ptr, type Pointer, type Library } from 'bun:ffi'
