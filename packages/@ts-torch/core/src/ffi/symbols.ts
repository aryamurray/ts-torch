/**
 * FFI symbol definitions for ts-torch native library
 * Defines all C function signatures with proper Bun FFI types
 *
 * These MUST match the signatures in native/include/ts_torch.h
 */

import { FFIType, type Pointer } from 'bun:ffi'

/**
 * Opaque pointer type representing a tensor handle in native code
 */
export type TensorHandle = Pointer

/**
 * FFI symbol definitions mapping C function names to their signatures
 * Each symbol defines args (parameter types) and returns (return type)
 */
export const FFI_SYMBOLS = {
  // ==================== Version ====================

  ts_version: {
    args: [],
    returns: FFIType.ptr, // returns const char*
  },

  // ==================== Tensor Creation ====================

  // ts_tensor_zeros(shape, ndim, dtype, device, device_index, error) -> TensorHandle
  ts_tensor_zeros: {
    args: [FFIType.ptr, FFIType.i32, FFIType.i32, FFIType.i32, FFIType.i32, FFIType.ptr],
    returns: FFIType.ptr,
  },

  ts_tensor_ones: {
    args: [FFIType.ptr, FFIType.i32, FFIType.i32, FFIType.i32, FFIType.i32, FFIType.ptr],
    returns: FFIType.ptr,
  },

  ts_tensor_randn: {
    args: [FFIType.ptr, FFIType.i32, FFIType.i32, FFIType.i32, FFIType.i32, FFIType.ptr],
    returns: FFIType.ptr,
  },

  ts_tensor_empty: {
    args: [FFIType.ptr, FFIType.i32, FFIType.i32, FFIType.i32, FFIType.i32, FFIType.ptr],
    returns: FFIType.ptr,
  },

  // ts_tensor_from_buffer(data, shape, ndim, dtype, device, device_index, error) -> TensorHandle
  ts_tensor_from_buffer: {
    args: [FFIType.ptr, FFIType.ptr, FFIType.i32, FFIType.i32, FFIType.i32, FFIType.i32, FFIType.ptr],
    returns: FFIType.ptr,
  },

  // ==================== Tensor Properties ====================

  // ts_tensor_ndim(tensor, error) -> int64_t
  ts_tensor_ndim: {
    args: [FFIType.ptr, FFIType.ptr],
    returns: FFIType.i64,
  },

  // ts_tensor_size(tensor, dim, error) -> int64_t
  ts_tensor_size: {
    args: [FFIType.ptr, FFIType.i32, FFIType.ptr],
    returns: FFIType.i64,
  },

  // ts_tensor_shape(tensor, out_shape, error)
  ts_tensor_shape: {
    args: [FFIType.ptr, FFIType.ptr, FFIType.ptr],
    returns: FFIType.void,
  },

  // ts_tensor_dtype(tensor, error) -> ts_DType
  ts_tensor_dtype: {
    args: [FFIType.ptr, FFIType.ptr],
    returns: FFIType.i32,
  },

  // ts_tensor_numel(tensor, error) -> int64_t
  ts_tensor_numel: {
    args: [FFIType.ptr, FFIType.ptr],
    returns: FFIType.i64,
  },

  // ts_tensor_requires_grad(tensor, error) -> int
  ts_tensor_requires_grad: {
    args: [FFIType.ptr, FFIType.ptr],
    returns: FFIType.i32,
  },

  // ts_tensor_set_requires_grad(tensor, requires_grad, error)
  ts_tensor_set_requires_grad: {
    args: [FFIType.ptr, FFIType.bool, FFIType.ptr],
    returns: FFIType.void,
  },

  // ==================== Tensor Memory ====================

  ts_tensor_delete: {
    args: [FFIType.ptr],
    returns: FFIType.void,
  },

  // ts_tensor_clone(tensor, error) -> TensorHandle
  ts_tensor_clone: {
    args: [FFIType.ptr, FFIType.ptr],
    returns: FFIType.ptr,
  },

  // ts_tensor_detach(tensor, error) -> TensorHandle
  ts_tensor_detach: {
    args: [FFIType.ptr, FFIType.ptr],
    returns: FFIType.ptr,
  },

  // ts_tensor_data_ptr(tensor, error) -> void*
  ts_tensor_data_ptr: {
    args: [FFIType.ptr, FFIType.ptr],
    returns: FFIType.ptr,
  },

  // ts_tensor_copy_to_buffer(tensor, buffer, size, error)
  ts_tensor_copy_to_buffer: {
    args: [FFIType.ptr, FFIType.ptr, FFIType.i64, FFIType.ptr],
    returns: FFIType.void,
  },

  // ==================== Tensor Operations ====================

  // ts_tensor_add(a, b, error) -> TensorHandle
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

  // ts_tensor_transpose(tensor, dim0, dim1, error) -> TensorHandle
  ts_tensor_transpose: {
    args: [FFIType.ptr, FFIType.i32, FFIType.i32, FFIType.ptr],
    returns: FFIType.ptr,
  },

  // ts_tensor_reshape(tensor, shape, ndim, error) -> TensorHandle
  ts_tensor_reshape: {
    args: [FFIType.ptr, FFIType.ptr, FFIType.i32, FFIType.ptr],
    returns: FFIType.ptr,
  },

  // ==================== Reductions ====================

  // ts_tensor_sum(tensor, error) -> TensorHandle
  ts_tensor_sum: {
    args: [FFIType.ptr, FFIType.ptr],
    returns: FFIType.ptr,
  },

  // ts_tensor_mean(tensor, error) -> TensorHandle
  ts_tensor_mean: {
    args: [FFIType.ptr, FFIType.ptr],
    returns: FFIType.ptr,
  },

  // ==================== Activations ====================

  // ts_tensor_relu(tensor, error) -> TensorHandle
  ts_tensor_relu: {
    args: [FFIType.ptr, FFIType.ptr],
    returns: FFIType.ptr,
  },

  // ts_tensor_sigmoid(tensor, error) -> TensorHandle
  ts_tensor_sigmoid: {
    args: [FFIType.ptr, FFIType.ptr],
    returns: FFIType.ptr,
  },

  // ts_tensor_softmax(tensor, dim, error) -> TensorHandle
  ts_tensor_softmax: {
    args: [FFIType.ptr, FFIType.i32, FFIType.ptr],
    returns: FFIType.ptr,
  },

  // ts_tensor_tanh(tensor, error) -> TensorHandle
  ts_tensor_tanh: {
    args: [FFIType.ptr, FFIType.ptr],
    returns: FFIType.ptr,
  },

  // ts_tensor_log(tensor, error) -> TensorHandle
  ts_tensor_log: {
    args: [FFIType.ptr, FFIType.ptr],
    returns: FFIType.ptr,
  },

  // ts_tensor_exp(tensor, error) -> TensorHandle
  ts_tensor_exp: {
    args: [FFIType.ptr, FFIType.ptr],
    returns: FFIType.ptr,
  },

  // ts_tensor_neg(tensor, error) -> TensorHandle
  ts_tensor_neg: {
    args: [FFIType.ptr, FFIType.ptr],
    returns: FFIType.ptr,
  },

  // ts_tensor_log_softmax(tensor, dim, error) -> TensorHandle
  ts_tensor_log_softmax: {
    args: [FFIType.ptr, FFIType.i32, FFIType.ptr],
    returns: FFIType.ptr,
  },

  // ==================== Scalar Operations ====================

  // ts_tensor_add_scalar(tensor, scalar, error) -> TensorHandle
  ts_tensor_add_scalar: {
    args: [FFIType.ptr, FFIType.f64, FFIType.ptr],
    returns: FFIType.ptr,
  },

  // ts_tensor_sub_scalar(tensor, scalar, error) -> TensorHandle
  ts_tensor_sub_scalar: {
    args: [FFIType.ptr, FFIType.f64, FFIType.ptr],
    returns: FFIType.ptr,
  },

  // ts_tensor_mul_scalar(tensor, scalar, error) -> TensorHandle
  ts_tensor_mul_scalar: {
    args: [FFIType.ptr, FFIType.f64, FFIType.ptr],
    returns: FFIType.ptr,
  },

  // ts_tensor_div_scalar(tensor, scalar, error) -> TensorHandle
  ts_tensor_div_scalar: {
    args: [FFIType.ptr, FFIType.f64, FFIType.ptr],
    returns: FFIType.ptr,
  },

  // ts_tensor_zero_grad(tensor, error)
  ts_tensor_zero_grad: {
    args: [FFIType.ptr, FFIType.ptr],
    returns: FFIType.void,
  },

  // ==================== Loss Functions ====================

  // ts_tensor_nll_loss(log_probs, targets, error) -> TensorHandle
  ts_tensor_nll_loss: {
    args: [FFIType.ptr, FFIType.ptr, FFIType.ptr],
    returns: FFIType.ptr,
  },

  // ts_tensor_cross_entropy_loss(logits, targets, error) -> TensorHandle
  ts_tensor_cross_entropy_loss: {
    args: [FFIType.ptr, FFIType.ptr, FFIType.ptr],
    returns: FFIType.ptr,
  },

  // ts_tensor_mse_loss(input, target, error) -> TensorHandle
  ts_tensor_mse_loss: {
    args: [FFIType.ptr, FFIType.ptr, FFIType.ptr],
    returns: FFIType.ptr,
  },

  // ==================== Autograd ====================

  // ts_tensor_backward(tensor, error)
  ts_tensor_backward: {
    args: [FFIType.ptr, FFIType.ptr],
    returns: FFIType.void,
  },

  // ts_tensor_grad(tensor, error) -> TensorHandle
  ts_tensor_grad: {
    args: [FFIType.ptr, FFIType.ptr],
    returns: FFIType.ptr,
  },

  // ==================== Scope Management ====================

  // ts_scope_begin() -> ScopeHandle
  ts_scope_begin: {
    args: [],
    returns: FFIType.ptr,
  },

  // ts_scope_end(scope)
  ts_scope_end: {
    args: [FFIType.ptr],
    returns: FFIType.void,
  },

  // ts_scope_register_tensor(scope, tensor)
  ts_scope_register_tensor: {
    args: [FFIType.ptr, FFIType.ptr],
    returns: FFIType.void,
  },

  // ts_scope_escape_tensor(scope, tensor) -> TensorHandle
  ts_scope_escape_tensor: {
    args: [FFIType.ptr, FFIType.ptr],
    returns: FFIType.ptr,
  },

  // ==================== Device ====================

  ts_cuda_is_available: {
    args: [],
    returns: FFIType.i32,
  },

  ts_cuda_device_count: {
    args: [],
    returns: FFIType.i32,
  },

  // ts_tensor_to_device(tensor, device_type, device_index, error) -> TensorHandle
  ts_tensor_to_device: {
    args: [FFIType.ptr, FFIType.i32, FFIType.i32, FFIType.ptr],
    returns: FFIType.ptr,
  },
} as const

export type FFISymbols = typeof FFI_SYMBOLS
