/**
 * FFI symbol definitions for ts-torch native library
 * Defines all C function signatures with proper Bun FFI types
 */

import { FFIType, type Pointer } from "bun:ffi";

/**
 * Opaque pointer type representing a tensor handle in native code
 */
export type TensorHandle = Pointer;

/**
 * FFI symbol definitions mapping C function names to their signatures
 * Each symbol defines args (parameter types) and returns (return type)
 */
export const FFI_SYMBOLS = {
  // ==================== Tensor Creation ====================

  ts_tensor_zeros: {
    args: [FFIType.ptr, FFIType.i32, FFIType.i32, FFIType.bool, FFIType.ptr],
    returns: FFIType.ptr,
  },

  ts_tensor_ones: {
    args: [FFIType.ptr, FFIType.i32, FFIType.i32, FFIType.bool, FFIType.ptr],
    returns: FFIType.ptr,
  },

  ts_tensor_randn: {
    args: [FFIType.ptr, FFIType.i32, FFIType.i32, FFIType.bool, FFIType.ptr],
    returns: FFIType.ptr,
  },

  ts_tensor_from_buffer: {
    args: [FFIType.ptr, FFIType.ptr, FFIType.i32, FFIType.i32, FFIType.bool, FFIType.ptr],
    returns: FFIType.ptr,
  },

  ts_tensor_empty: {
    args: [FFIType.ptr, FFIType.i32, FFIType.i32, FFIType.bool, FFIType.ptr],
    returns: FFIType.ptr,
  },

  // ==================== Tensor Properties ====================

  ts_tensor_ndim: {
    args: [FFIType.ptr],
    returns: FFIType.i32,
  },

  ts_tensor_size: {
    args: [FFIType.ptr, FFIType.i32],
    returns: FFIType.i64,
  },

  ts_tensor_shape: {
    args: [FFIType.ptr, FFIType.ptr],
    returns: FFIType.void,
  },

  ts_tensor_dtype: {
    args: [FFIType.ptr],
    returns: FFIType.i32,
  },

  ts_tensor_numel: {
    args: [FFIType.ptr],
    returns: FFIType.i64,
  },

  ts_tensor_requires_grad: {
    args: [FFIType.ptr],
    returns: FFIType.bool,
  },

  ts_tensor_set_requires_grad: {
    args: [FFIType.ptr, FFIType.bool],
    returns: FFIType.void,
  },

  // ==================== Tensor Memory ====================

  ts_tensor_delete: {
    args: [FFIType.ptr],
    returns: FFIType.void,
  },

  ts_tensor_clone: {
    args: [FFIType.ptr, FFIType.ptr],
    returns: FFIType.ptr,
  },

  ts_tensor_detach: {
    args: [FFIType.ptr, FFIType.ptr],
    returns: FFIType.ptr,
  },

  ts_tensor_data_ptr: {
    args: [FFIType.ptr],
    returns: FFIType.ptr,
  },

  ts_tensor_copy_to_buffer: {
    args: [FFIType.ptr, FFIType.ptr, FFIType.i64],
    returns: FFIType.void,
  },

  // ==================== Tensor Operations ====================

  ts_tensor_add: {
    args: [FFIType.ptr, FFIType.ptr, FFIType.ptr],
    returns: FFIType.ptr,
  },

  ts_tensor_sub: {
    args: [FFIType.ptr, FFIType.ptr, FFIType.ptr],
    returns: FFIType.ptr,
  },

  ts_tensor_mul: {
    args: [FFIType.ptr, FFIType.ptr, FFIType.ptr],
    returns: FFIType.ptr,
  },

  ts_tensor_div: {
    args: [FFIType.ptr, FFIType.ptr, FFIType.ptr],
    returns: FFIType.ptr,
  },

  ts_tensor_matmul: {
    args: [FFIType.ptr, FFIType.ptr, FFIType.ptr],
    returns: FFIType.ptr,
  },

  ts_tensor_transpose: {
    args: [FFIType.ptr, FFIType.i32, FFIType.i32, FFIType.ptr],
    returns: FFIType.ptr,
  },

  ts_tensor_reshape: {
    args: [FFIType.ptr, FFIType.ptr, FFIType.i32, FFIType.ptr],
    returns: FFIType.ptr,
  },

  // ==================== Reductions ====================

  ts_tensor_sum: {
    args: [FFIType.ptr, FFIType.ptr],
    returns: FFIType.ptr,
  },

  ts_tensor_mean: {
    args: [FFIType.ptr, FFIType.ptr],
    returns: FFIType.ptr,
  },

  ts_tensor_max: {
    args: [FFIType.ptr, FFIType.ptr],
    returns: FFIType.ptr,
  },

  ts_tensor_min: {
    args: [FFIType.ptr, FFIType.ptr],
    returns: FFIType.ptr,
  },

  // ==================== Activations ====================

  ts_tensor_relu: {
    args: [FFIType.ptr, FFIType.ptr],
    returns: FFIType.ptr,
  },

  ts_tensor_sigmoid: {
    args: [FFIType.ptr, FFIType.ptr],
    returns: FFIType.ptr,
  },

  ts_tensor_softmax: {
    args: [FFIType.ptr, FFIType.i32, FFIType.ptr],
    returns: FFIType.ptr,
  },

  ts_tensor_tanh: {
    args: [FFIType.ptr, FFIType.ptr],
    returns: FFIType.ptr,
  },

  // ==================== Autograd ====================

  ts_tensor_backward: {
    args: [FFIType.ptr, FFIType.ptr],
    returns: FFIType.void,
  },

  ts_tensor_grad: {
    args: [FFIType.ptr],
    returns: FFIType.ptr,
  },

  ts_tensor_zero_grad: {
    args: [FFIType.ptr],
    returns: FFIType.void,
  },

  // ==================== Scope Management ====================

  ts_scope_begin: {
    args: [],
    returns: FFIType.void,
  },

  ts_scope_end: {
    args: [],
    returns: FFIType.void,
  },

  ts_scope_register_tensor: {
    args: [FFIType.ptr],
    returns: FFIType.void,
  },

  ts_scope_escape_tensor: {
    args: [FFIType.ptr],
    returns: FFIType.void,
  },

  // ==================== Device ====================

  ts_cuda_is_available: {
    args: [],
    returns: FFIType.bool,
  },

  ts_cuda_device_count: {
    args: [],
    returns: FFIType.i32,
  },

  ts_tensor_to_device: {
    args: [FFIType.ptr, FFIType.i32, FFIType.i32, FFIType.ptr],
    returns: FFIType.ptr,
  },

  // ==================== Version ====================

  ts_version: {
    args: [FFIType.ptr, FFIType.ptr, FFIType.ptr],
    returns: FFIType.void,
  },
} as const;

export type FFISymbols = typeof FFI_SYMBOLS;
