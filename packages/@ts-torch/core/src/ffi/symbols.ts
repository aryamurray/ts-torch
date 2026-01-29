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

  // ==================== Tensor Creation (int32 fast path) ====================
  // These use int32 shapes to avoid BigInt conversion overhead

  ts_tensor_zeros_i32: {
    args: [ptr, i32, i32, i32, i32, ptr] as const,
    returns: ptr,
  },

  ts_tensor_ones_i32: {
    args: [ptr, i32, i32, i32, i32, ptr] as const,
    returns: ptr,
  },

  ts_tensor_randn_i32: {
    args: [ptr, i32, i32, i32, i32, ptr] as const,
    returns: ptr,
  },

  ts_tensor_rand_i32: {
    args: [ptr, i32, i32, i32, i32, ptr] as const,
    returns: ptr,
  },

  ts_tensor_empty_i32: {
    args: [ptr, i32, i32, i32, i32, ptr] as const,
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

  // ts_tensor_cat(tensors_array, num_tensors, dim, error) -> TensorHandle
  // Concatenates tensors along the specified dimension
  ts_tensor_cat: {
    args: [ptr, i32, i32, ptr] as const,
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

  // ts_tensor_sqrt(tensor, error) -> TensorHandle
  ts_tensor_sqrt: {
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

  // ==================== In-place Operations ====================

  // ts_tensor_sub_inplace(tensor, other, error) - tensor.data -= other
  ts_tensor_sub_inplace: {
    args: [ptr, ptr, ptr] as const,
    returns: void_,
  },

  // ts_tensor_add_scaled_inplace(tensor, other, scalar, error) - tensor.data += scalar * other
  ts_tensor_add_scaled_inplace: {
    args: [ptr, ptr, f64, ptr] as const,
    returns: void_,
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

  // ==================== Dimension-wise Reductions ====================

  // ts_tensor_sum_dim(tensor, dim, keepdim, error) -> TensorHandle
  ts_tensor_sum_dim: {
    args: [ptr, i64, i32, ptr] as const,
    returns: ptr,
  },

  // ts_tensor_mean_dim(tensor, dim, keepdim, error) -> TensorHandle
  ts_tensor_mean_dim: {
    args: [ptr, i64, i32, ptr] as const,
    returns: ptr,
  },

  // ts_tensor_var(tensor, dim, unbiased, keepdim, error) -> TensorHandle
  ts_tensor_var: {
    args: [ptr, i64, i32, i32, ptr] as const,
    returns: ptr,
  },

  // ==================== Comparison Operations ====================

  // ts_tensor_eq(a, b, error) -> TensorHandle
  ts_tensor_eq: {
    args: [ptr, ptr, ptr] as const,
    returns: ptr,
  },

  // ts_tensor_ne(a, b, error) -> TensorHandle
  ts_tensor_ne: {
    args: [ptr, ptr, ptr] as const,
    returns: ptr,
  },

  // ts_tensor_lt(a, b, error) -> TensorHandle
  ts_tensor_lt: {
    args: [ptr, ptr, ptr] as const,
    returns: ptr,
  },

  // ts_tensor_le(a, b, error) -> TensorHandle
  ts_tensor_le: {
    args: [ptr, ptr, ptr] as const,
    returns: ptr,
  },

  // ts_tensor_gt(a, b, error) -> TensorHandle
  ts_tensor_gt: {
    args: [ptr, ptr, ptr] as const,
    returns: ptr,
  },

  // ts_tensor_ge(a, b, error) -> TensorHandle
  ts_tensor_ge: {
    args: [ptr, ptr, ptr] as const,
    returns: ptr,
  },

  // ==================== Random Tensors ====================

  // ts_tensor_rand(shape, ndim, dtype, device, device_index, error) -> TensorHandle
  ts_tensor_rand: {
    args: [ptr, i32, i32, i32, i32, ptr] as const,
    returns: ptr,
  },

  // ==================== Convolution Operations ====================

  // ts_tensor_conv2d(input, weight, bias, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups, error) -> TensorHandle
  ts_tensor_conv2d: {
    args: [ptr, ptr, ptr, i64, i64, i64, i64, i64, i64, i64, ptr] as const,
    returns: ptr,
  },

  // ==================== Pooling Operations ====================

  // ts_tensor_max_pool2d(input, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, error) -> TensorHandle
  ts_tensor_max_pool2d: {
    args: [ptr, i64, i64, i64, i64, i64, i64, ptr] as const,
    returns: ptr,
  },

  // ts_tensor_avg_pool2d(input, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, error) -> TensorHandle
  ts_tensor_avg_pool2d: {
    args: [ptr, i64, i64, i64, i64, i64, i64, ptr] as const,
    returns: ptr,
  },

  // ==================== Regularization ====================

  // ts_tensor_dropout(input, p, training, error) -> TensorHandle
  ts_tensor_dropout: {
    args: [ptr, f64, i32, ptr] as const,
    returns: ptr,
  },

  // ==================== Normalization ====================

  // ts_tensor_batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps, error) -> TensorHandle
  ts_tensor_batch_norm: {
    args: [ptr, ptr, ptr, ptr, ptr, i32, f64, f64, ptr] as const,
    returns: ptr,
  },

  // ts_tensor_layer_norm(input, normalized_shape, normalized_shape_len, weight, bias, eps, error) -> TensorHandle
  ts_tensor_layer_norm: {
    args: [ptr, ptr, i32, ptr, ptr, f64, ptr] as const,
    returns: ptr,
  },

  // ==================== Indexing Operations ====================

  // ts_tensor_index_select(tensor, dim, index, error) -> TensorHandle
  ts_tensor_index_select: {
    args: [ptr, i64, ptr, ptr] as const,
    returns: ptr,
  },

  // ts_tensor_argmax(tensor, dim, keepdim, error) -> TensorHandle
  ts_tensor_argmax: {
    args: [ptr, i64, i32, ptr] as const,
    returns: ptr,
  },

// ts_tensor_narrow(tensor, dim, start, length, error) -> TensorHandle (view, zero-copy)
  ts_tensor_narrow: {
    args: [ptr, i64, i64, i64, ptr] as const,
    returns: ptr,
  },

  // ts_tensor_minimum(a, b, error) -> TensorHandle (element-wise min)
  ts_tensor_minimum: {
    args: [ptr, ptr, ptr] as const,
    returns: ptr,
  },

  // ts_tensor_maximum(a, b, error) -> TensorHandle (element-wise max)
  ts_tensor_maximum: {
    args: [ptr, ptr, ptr] as const,
    returns: ptr,
  },

  // ts_tensor_clamp(tensor, min, max, error) -> TensorHandle
  ts_tensor_clamp: {
    args: [ptr, f64, f64, ptr] as const,
    returns: ptr,
  },

  // ts_tensor_clamp_min(tensor, min, error) -> TensorHandle
  ts_tensor_clamp_min: {
    args: [ptr, f64, ptr] as const,
    returns: ptr,
  },

  // ts_tensor_clamp_max(tensor, max, error) -> TensorHandle
  ts_tensor_clamp_max: {
    args: [ptr, f64, ptr] as const,
    returns: ptr,
  },

  // ==================== Advanced Tensor Operations ====================

  // ts_tensor_triu(tensor, diagonal, error) -> TensorHandle (upper triangular)
  ts_tensor_triu: {
    args: [ptr, i64, ptr] as const,
    returns: ptr,
  },

  // ts_tensor_tril(tensor, diagonal, error) -> TensorHandle (lower triangular)
  ts_tensor_tril: {
    args: [ptr, i64, ptr] as const,
    returns: ptr,
  },

  // ts_tensor_masked_fill(tensor, mask, value, error) -> TensorHandle
  ts_tensor_masked_fill: {
    args: [ptr, ptr, f64, ptr] as const,
    returns: ptr,
  },

  // ts_tensor_bmm(a, b, error) -> TensorHandle (batched matrix multiply)
  ts_tensor_bmm: {
    args: [ptr, ptr, ptr] as const,
    returns: ptr,
  },

  // ts_tensor_gather(input, dim, index, error) -> TensorHandle
  ts_tensor_gather: {
    args: [ptr, i64, ptr, ptr] as const,
    returns: ptr,
  },

  // ts_tensor_scatter(input, dim, index, src, error) -> TensorHandle
  ts_tensor_scatter: {
    args: [ptr, i64, ptr, ptr, ptr] as const,
    returns: ptr,
  },

  // ts_tensor_topk(tensor, k, dim, largest, sorted, indices_out, error) -> TensorHandle
  // Returns values tensor; indices tensor handle is written to indices_out
  ts_tensor_topk: {
    args: [ptr, i64, i64, i32, i32, ptr, ptr] as const,
    returns: ptr,
  },

  // ts_tensor_sort(tensor, dim, descending, indices_out, error) -> TensorHandle
  // Returns sorted values tensor; indices tensor handle is written to indices_out
  ts_tensor_sort: {
    args: [ptr, i64, i32, ptr, ptr] as const,
    returns: ptr,
  },

  // ts_tensor_where(condition, x, y, error) -> TensorHandle
  ts_tensor_where: {
    args: [ptr, ptr, ptr, ptr] as const,
    returns: ptr,
  },

  // ts_tensor_nonzero(tensor, error) -> TensorHandle
  ts_tensor_nonzero: {
    args: [ptr, ptr] as const,
    returns: ptr,
  },

  // ts_tensor_repeat(tensor, repeats, num_dims, error) -> TensorHandle
  ts_tensor_repeat: {
    args: [ptr, ptr, i32, ptr] as const,
    returns: ptr,
  },

  // ts_tensor_expand(tensor, sizes, num_dims, error) -> TensorHandle
  ts_tensor_expand: {
    args: [ptr, ptr, i32, ptr] as const,
    returns: ptr,
  },

  // ==================== Einstein Summation ====================

  // ts_tensor_einsum(equation, tensors_array, num_tensors, error) -> TensorHandle
  ts_tensor_einsum: {
    args: [ptr, ptr, i32, ptr] as const,
    returns: ptr,
  },

  // ==================== In-Place Operations (Phase 4) ====================

  // ts_tensor_add_(tensor, other, error)
  ts_tensor_add_: {
    args: [ptr, ptr, ptr] as const,
    returns: void_,
  },

  // ts_tensor_sub_(tensor, other, error)
  ts_tensor_sub_: {
    args: [ptr, ptr, ptr] as const,
    returns: void_,
  },

  // ts_tensor_mul_(tensor, other, error)
  ts_tensor_mul_: {
    args: [ptr, ptr, ptr] as const,
    returns: void_,
  },

  // ts_tensor_mul_scalar_(tensor, scalar, error)
  ts_tensor_mul_scalar_: {
    args: [ptr, f64, ptr] as const,
    returns: void_,
  },

  // ts_tensor_add_alpha_(tensor, other, alpha, error)
  ts_tensor_add_alpha_: {
    args: [ptr, ptr, f64, ptr] as const,
    returns: void_,
  },

  // ts_tensor_optim_add_(tensor, other, alpha, error)
  ts_tensor_optim_add_: {
    args: [ptr, ptr, f64, ptr] as const,
    returns: void_,
  },

  // ts_tensor_zero_grad_(tensor, error)
  ts_tensor_zero_grad_: {
    args: [ptr, ptr] as const,
    returns: void_,
  },

  // ts_tensor_div_(tensor, other, error)
  ts_tensor_div_: {
    args: [ptr, ptr, ptr] as const,
    returns: void_,
  },

  // ts_tensor_div_scalar_(tensor, scalar, error)
  ts_tensor_div_scalar_: {
    args: [ptr, f64, ptr] as const,
    returns: void_,
  },

  // ==================== Out= Operations (Pre-allocated output) ====================

  // ts_tensor_add_out(a, b, out, error)
  ts_tensor_add_out: {
    args: [ptr, ptr, ptr, ptr] as const,
    returns: void_,
  },

  // ts_tensor_sub_out(a, b, out, error)
  ts_tensor_sub_out: {
    args: [ptr, ptr, ptr, ptr] as const,
    returns: void_,
  },

  // ts_tensor_mul_out(a, b, out, error)
  ts_tensor_mul_out: {
    args: [ptr, ptr, ptr, ptr] as const,
    returns: void_,
  },

  // ts_tensor_div_out(a, b, out, error)
  ts_tensor_div_out: {
    args: [ptr, ptr, ptr, ptr] as const,
    returns: void_,
  },

  // ts_tensor_matmul_out(a, b, out, error)
  ts_tensor_matmul_out: {
    args: [ptr, ptr, ptr, ptr] as const,
    returns: void_,
  },

  // ==================== Fused Operations (Phase 3) ====================

  // ts_tensor_linear_relu(input, weight, bias, error) -> TensorHandle
  ts_tensor_linear_relu: {
    args: [ptr, ptr, ptr, ptr] as const,
    returns: ptr,
  },

  // ts_tensor_linear_sigmoid(input, weight, bias, error) -> TensorHandle
  ts_tensor_linear_sigmoid: {
    args: [ptr, ptr, ptr, ptr] as const,
    returns: ptr,
  },

  // ts_tensor_linear_tanh(input, weight, bias, error) -> TensorHandle
  ts_tensor_linear_tanh: {
    args: [ptr, ptr, ptr, ptr] as const,
    returns: ptr,
  },

  // ts_tensor_add_relu(a, b, error) -> TensorHandle
  ts_tensor_add_relu: {
    args: [ptr, ptr, ptr] as const,
    returns: ptr,
  },

  // ==================== Batching API (Phase 2) ====================

  // ts_batch_begin(error) -> BatchHandle
  ts_batch_begin: {
    args: [ptr] as const,
    returns: ptr,
  },

  // ts_batch_end(batch, error) -> TensorHandle
  ts_batch_end: {
    args: [ptr, ptr] as const,
    returns: ptr,
  },

  // ts_batch_abort(batch)
  ts_batch_abort: {
    args: [ptr] as const,
    returns: void_,
  },

  // ts_batch_is_recording() -> int
  ts_batch_is_recording: {
    args: [] as const,
    returns: i32,
  },

  // ==================== Direct Batched Operations ====================

  // ts_tensor_chain_matmul(tensors, count, error) -> TensorHandle
  ts_tensor_chain_matmul: {
    args: [ptr, i32, ptr] as const,
    returns: ptr,
  },

  // ts_tensor_mlp_forward(input, weights, biases, num_layers, apply_relu, error) -> TensorHandle
  ts_tensor_mlp_forward: {
    args: [ptr, ptr, ptr, i32, i32, ptr] as const,
    returns: ptr,
  },

  // ==================== Thread Controls (Phase 6) ====================

  // ts_set_num_threads(num_threads)
  ts_set_num_threads: {
    args: [i32] as const,
    returns: void_,
  },

  // ts_get_num_threads() -> int
  ts_get_num_threads: {
    args: [] as const,
    returns: i32,
  },
} as const

export type FFISymbols = typeof FFI_SYMBOLS
