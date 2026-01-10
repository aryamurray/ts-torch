/**
 * FFI symbol definitions for ts-torch native library
 * Defines all C function signatures with koffi-compatible types
 *
 * These MUST match the signatures in native/include/ts_torch.h
 */

/**
 * Opaque pointer type representing a tensor handle in native code
 * In koffi, pointers are represented as opaque values
 */
export type TensorHandle = unknown

/**
 * koffi type string aliases for clarity
 */
const ptr = 'void*' as const
const i32 = 'int' as const
const i64 = 'int64_t' as const
const f64 = 'double' as const
const bool_ = 'bool' as const
const void_ = 'void' as const

/**
 * FFI symbol definitions mapping C function names to their signatures
 * Each symbol defines args (parameter types) and returns (return type)
 */
export const FFI_SYMBOLS = {
  // ==================== Version ====================

  ts_version: {
    args: [] as const,
    returns: ptr, // returns const char*
  },

  // ==================== Tensor Creation ====================

  // ts_tensor_zeros(shape, ndim, dtype, device, device_index, error) -> TensorHandle
  ts_tensor_zeros: {
    args: [ptr, i32, i32, i32, i32, ptr] as const,
    returns: ptr,
  },

  ts_tensor_ones: {
    args: [ptr, i32, i32, i32, i32, ptr] as const,
    returns: ptr,
  },

  ts_tensor_randn: {
    args: [ptr, i32, i32, i32, i32, ptr] as const,
    returns: ptr,
  },

  ts_tensor_empty: {
    args: [ptr, i32, i32, i32, i32, ptr] as const,
    returns: ptr,
  },

  // ts_tensor_from_buffer(data, shape, ndim, dtype, device, device_index, error) -> TensorHandle
  ts_tensor_from_buffer: {
    args: [ptr, ptr, i32, i32, i32, i32, ptr] as const,
    returns: ptr,
  },

  // ==================== Tensor Properties ====================

  // ts_tensor_ndim(tensor, error) -> int64_t
  ts_tensor_ndim: {
    args: [ptr, ptr] as const,
    returns: i64,
  },

  // ts_tensor_size(tensor, dim, error) -> int64_t
  ts_tensor_size: {
    args: [ptr, i32, ptr] as const,
    returns: i64,
  },

  // ts_tensor_shape(tensor, out_shape, error)
  ts_tensor_shape: {
    args: [ptr, ptr, ptr] as const,
    returns: void_,
  },

  // ts_tensor_dtype(tensor, error) -> ts_DType
  ts_tensor_dtype: {
    args: [ptr, ptr] as const,
    returns: i32,
  },

  // ts_tensor_numel(tensor, error) -> int64_t
  ts_tensor_numel: {
    args: [ptr, ptr] as const,
    returns: i64,
  },

  // ts_tensor_requires_grad(tensor, error) -> int
  ts_tensor_requires_grad: {
    args: [ptr, ptr] as const,
    returns: i32,
  },

  // ts_tensor_set_requires_grad(tensor, requires_grad, error)
  ts_tensor_set_requires_grad: {
    args: [ptr, bool_, ptr] as const,
    returns: void_,
  },

  // ==================== Tensor Memory ====================

  ts_tensor_delete: {
    args: [ptr] as const,
    returns: void_,
  },

  // ts_tensor_clone(tensor, error) -> TensorHandle
  ts_tensor_clone: {
    args: [ptr, ptr] as const,
    returns: ptr,
  },

  // ts_tensor_detach(tensor, error) -> TensorHandle
  ts_tensor_detach: {
    args: [ptr, ptr] as const,
    returns: ptr,
  },

  // ts_tensor_data_ptr(tensor, error) -> void*
  ts_tensor_data_ptr: {
    args: [ptr, ptr] as const,
    returns: ptr,
  },

  // ts_tensor_copy_to_buffer(tensor, buffer, size, error)
  ts_tensor_copy_to_buffer: {
    args: [ptr, ptr, i64, ptr] as const,
    returns: void_,
  },

  // ==================== Tensor Operations ====================

  // ts_tensor_add(a, b, error) -> TensorHandle
  ts_tensor_add: {
    args: [ptr, ptr, ptr] as const,
    returns: ptr,
  },

  ts_tensor_sub: {
    args: [ptr, ptr, ptr] as const,
    returns: ptr,
  },

  ts_tensor_mul: {
    args: [ptr, ptr, ptr] as const,
    returns: ptr,
  },

  ts_tensor_div: {
    args: [ptr, ptr, ptr] as const,
    returns: ptr,
  },

  ts_tensor_matmul: {
    args: [ptr, ptr, ptr] as const,
    returns: ptr,
  },

  // ts_tensor_transpose(tensor, dim0, dim1, error) -> TensorHandle
  ts_tensor_transpose: {
    args: [ptr, i32, i32, ptr] as const,
    returns: ptr,
  },

  // ts_tensor_reshape(tensor, shape, ndim, error) -> TensorHandle
  ts_tensor_reshape: {
    args: [ptr, ptr, i32, ptr] as const,
    returns: ptr,
  },

  // ==================== Reductions ====================

  // ts_tensor_sum(tensor, error) -> TensorHandle
  ts_tensor_sum: {
    args: [ptr, ptr] as const,
    returns: ptr,
  },

  // ts_tensor_mean(tensor, error) -> TensorHandle
  ts_tensor_mean: {
    args: [ptr, ptr] as const,
    returns: ptr,
  },

  // ==================== Activations ====================

  // ts_tensor_relu(tensor, error) -> TensorHandle
  ts_tensor_relu: {
    args: [ptr, ptr] as const,
    returns: ptr,
  },

  // ts_tensor_sigmoid(tensor, error) -> TensorHandle
  ts_tensor_sigmoid: {
    args: [ptr, ptr] as const,
    returns: ptr,
  },

  // ts_tensor_softmax(tensor, dim, error) -> TensorHandle
  ts_tensor_softmax: {
    args: [ptr, i32, ptr] as const,
    returns: ptr,
  },

  // ts_tensor_tanh(tensor, error) -> TensorHandle
  ts_tensor_tanh: {
    args: [ptr, ptr] as const,
    returns: ptr,
  },

  // ts_tensor_log(tensor, error) -> TensorHandle
  ts_tensor_log: {
    args: [ptr, ptr] as const,
    returns: ptr,
  },

  // ts_tensor_exp(tensor, error) -> TensorHandle
  ts_tensor_exp: {
    args: [ptr, ptr] as const,
    returns: ptr,
  },

  // ts_tensor_neg(tensor, error) -> TensorHandle
  ts_tensor_neg: {
    args: [ptr, ptr] as const,
    returns: ptr,
  },

  // ts_tensor_log_softmax(tensor, dim, error) -> TensorHandle
  ts_tensor_log_softmax: {
    args: [ptr, i32, ptr] as const,
    returns: ptr,
  },

  // ==================== Scalar Operations ====================

  // ts_tensor_add_scalar(tensor, scalar, error) -> TensorHandle
  ts_tensor_add_scalar: {
    args: [ptr, f64, ptr] as const,
    returns: ptr,
  },

  // ts_tensor_sub_scalar(tensor, scalar, error) -> TensorHandle
  ts_tensor_sub_scalar: {
    args: [ptr, f64, ptr] as const,
    returns: ptr,
  },

  // ts_tensor_mul_scalar(tensor, scalar, error) -> TensorHandle
  ts_tensor_mul_scalar: {
    args: [ptr, f64, ptr] as const,
    returns: ptr,
  },

  // ts_tensor_div_scalar(tensor, scalar, error) -> TensorHandle
  ts_tensor_div_scalar: {
    args: [ptr, f64, ptr] as const,
    returns: ptr,
  },

  // ts_tensor_zero_grad(tensor, error)
  ts_tensor_zero_grad: {
    args: [ptr, ptr] as const,
    returns: void_,
  },

  // ==================== Loss Functions ====================

  // ts_tensor_nll_loss(log_probs, targets, error) -> TensorHandle
  ts_tensor_nll_loss: {
    args: [ptr, ptr, ptr] as const,
    returns: ptr,
  },

  // ts_tensor_cross_entropy_loss(logits, targets, error) -> TensorHandle
  ts_tensor_cross_entropy_loss: {
    args: [ptr, ptr, ptr] as const,
    returns: ptr,
  },

  // ts_tensor_mse_loss(input, target, error) -> TensorHandle
  ts_tensor_mse_loss: {
    args: [ptr, ptr, ptr] as const,
    returns: ptr,
  },

  // ==================== Autograd ====================

  // ts_tensor_backward(tensor, error)
  ts_tensor_backward: {
    args: [ptr, ptr] as const,
    returns: void_,
  },

  // ts_tensor_grad(tensor, error) -> TensorHandle
  ts_tensor_grad: {
    args: [ptr, ptr] as const,
    returns: ptr,
  },

  // ==================== Scope Management ====================

  // ts_scope_begin() -> ScopeHandle
  ts_scope_begin: {
    args: [] as const,
    returns: ptr,
  },

  // ts_scope_end(scope)
  ts_scope_end: {
    args: [ptr] as const,
    returns: void_,
  },

  // ts_scope_register_tensor(scope, tensor)
  ts_scope_register_tensor: {
    args: [ptr, ptr] as const,
    returns: void_,
  },

  // ts_scope_escape_tensor(scope, tensor) -> TensorHandle
  ts_scope_escape_tensor: {
    args: [ptr, ptr] as const,
    returns: ptr,
  },

  // ==================== Device ====================

  ts_cuda_is_available: {
    args: [] as const,
    returns: i32,
  },

  ts_cuda_device_count: {
    args: [] as const,
    returns: i32,
  },

  // ts_tensor_to_device(tensor, device_type, device_index, error) -> TensorHandle
  ts_tensor_to_device: {
    args: [ptr, i32, i32, ptr] as const,
    returns: ptr,
  },
} as const

export type FFISymbols = typeof FFI_SYMBOLS
