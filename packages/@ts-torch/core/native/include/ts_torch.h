#ifndef TS_TORCH_H
#define TS_TORCH_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
    #ifdef TS_TORCH_EXPORTS
        #define TS_TORCH_API __declspec(dllexport)
    #else
        #define TS_TORCH_API __declspec(dllimport)
    #endif
#else
    #define TS_TORCH_API __attribute__((visibility("default")))
#endif

// Opaque handle types
typedef struct ts_Tensor* ts_TensorHandle;
typedef struct ts_Module* ts_ModuleHandle;
typedef struct ts_Optimizer* ts_OptimizerHandle;
typedef struct ts_Scope* ts_ScopeHandle;

// Error handling structure
typedef struct {
    int code;
    char message[256];
} ts_Error;

// Data type enumeration
typedef enum {
    TS_DTYPE_FLOAT32 = 0,
    TS_DTYPE_FLOAT64 = 1,
    TS_DTYPE_INT32 = 2,
    TS_DTYPE_INT64 = 3,
    TS_DTYPE_BOOL = 4,
    TS_DTYPE_FLOAT16 = 5,
    TS_DTYPE_BFLOAT16 = 6,
    TS_DTYPE_UINT8 = 7,
    TS_DTYPE_INT8 = 8,
    TS_DTYPE_INT16 = 9
} ts_DType;

// Device type enumeration
typedef enum {
    TS_DEVICE_CPU = 0,
    TS_DEVICE_CUDA = 1,
    TS_DEVICE_MPS = 2
} ts_DeviceType;

// Version information
TS_TORCH_API const char* ts_version(void);

// Error handling
TS_TORCH_API void ts_error_clear(ts_Error* error);
TS_TORCH_API int ts_error_occurred(const ts_Error* error);

// Tensor creation functions (int64 shape - for large dimensions)
TS_TORCH_API ts_TensorHandle ts_tensor_zeros(
    const int64_t* shape,
    size_t ndim,
    ts_DType dtype,
    ts_DeviceType device,
    int device_index,
    ts_Error* error
);

TS_TORCH_API ts_TensorHandle ts_tensor_ones(
    const int64_t* shape,
    size_t ndim,
    ts_DType dtype,
    ts_DeviceType device,
    int device_index,
    ts_Error* error
);

TS_TORCH_API ts_TensorHandle ts_tensor_randn(
    const int64_t* shape,
    size_t ndim,
    ts_DType dtype,
    ts_DeviceType device,
    int device_index,
    ts_Error* error
);

TS_TORCH_API ts_TensorHandle ts_tensor_empty(
    const int64_t* shape,
    size_t ndim,
    ts_DType dtype,
    ts_DeviceType device,
    int device_index,
    ts_Error* error
);

// Tensor creation functions (int32 shape - fast path, avoids BigInt overhead)
// Use these when all shape dimensions fit in int32 (< 2^31)
TS_TORCH_API ts_TensorHandle ts_tensor_zeros_i32(
    const int32_t* shape,
    size_t ndim,
    ts_DType dtype,
    ts_DeviceType device,
    int device_index,
    ts_Error* error
);

TS_TORCH_API ts_TensorHandle ts_tensor_ones_i32(
    const int32_t* shape,
    size_t ndim,
    ts_DType dtype,
    ts_DeviceType device,
    int device_index,
    ts_Error* error
);

TS_TORCH_API ts_TensorHandle ts_tensor_randn_i32(
    const int32_t* shape,
    size_t ndim,
    ts_DType dtype,
    ts_DeviceType device,
    int device_index,
    ts_Error* error
);

TS_TORCH_API ts_TensorHandle ts_tensor_rand_i32(
    const int32_t* shape,
    size_t ndim,
    ts_DType dtype,
    ts_DeviceType device,
    int device_index,
    ts_Error* error
);

TS_TORCH_API ts_TensorHandle ts_tensor_empty_i32(
    const int32_t* shape,
    size_t ndim,
    ts_DType dtype,
    ts_DeviceType device,
    int device_index,
    ts_Error* error
);

TS_TORCH_API ts_TensorHandle ts_tensor_from_buffer(
    const void* data,
    const int64_t* shape,
    size_t ndim,
    ts_DType dtype,
    ts_DeviceType device,
    int device_index,
    ts_Error* error
);

// Tensor properties
TS_TORCH_API int64_t ts_tensor_ndim(ts_TensorHandle tensor, ts_Error* error);

TS_TORCH_API int64_t ts_tensor_size(
    ts_TensorHandle tensor,
    int64_t dim,
    ts_Error* error
);

TS_TORCH_API void ts_tensor_shape(
    ts_TensorHandle tensor,
    int64_t* shape,
    size_t* ndim,
    ts_Error* error
);

TS_TORCH_API ts_DType ts_tensor_dtype(ts_TensorHandle tensor, ts_Error* error);

TS_TORCH_API int64_t ts_tensor_numel(ts_TensorHandle tensor, ts_Error* error);

TS_TORCH_API ts_DeviceType ts_tensor_device_type(
    ts_TensorHandle tensor,
    ts_Error* error
);

TS_TORCH_API int ts_tensor_device_index(
    ts_TensorHandle tensor,
    ts_Error* error
);

// Tensor memory management
TS_TORCH_API void ts_tensor_delete(ts_TensorHandle tensor);

TS_TORCH_API ts_TensorHandle ts_tensor_clone(
    ts_TensorHandle tensor,
    ts_Error* error
);

TS_TORCH_API ts_TensorHandle ts_tensor_detach(
    ts_TensorHandle tensor,
    ts_Error* error
);

TS_TORCH_API void* ts_tensor_data_ptr(ts_TensorHandle tensor, ts_Error* error);

TS_TORCH_API void ts_tensor_copy_to_buffer(
    ts_TensorHandle tensor,
    void* buffer,
    size_t buffer_size,
    ts_Error* error
);

// Tensor operations
TS_TORCH_API ts_TensorHandle ts_tensor_add(
    ts_TensorHandle a,
    ts_TensorHandle b,
    ts_Error* error
);

TS_TORCH_API ts_TensorHandle ts_tensor_sub(
    ts_TensorHandle a,
    ts_TensorHandle b,
    ts_Error* error
);

TS_TORCH_API ts_TensorHandle ts_tensor_mul(
    ts_TensorHandle a,
    ts_TensorHandle b,
    ts_Error* error
);

TS_TORCH_API ts_TensorHandle ts_tensor_div(
    ts_TensorHandle a,
    ts_TensorHandle b,
    ts_Error* error
);

TS_TORCH_API ts_TensorHandle ts_tensor_matmul(
    ts_TensorHandle a,
    ts_TensorHandle b,
    ts_Error* error
);

TS_TORCH_API ts_TensorHandle ts_tensor_transpose(
    ts_TensorHandle tensor,
    int64_t dim0,
    int64_t dim1,
    ts_Error* error
);

TS_TORCH_API ts_TensorHandle ts_tensor_reshape(
    ts_TensorHandle tensor,
    const int64_t* shape,
    size_t ndim,
    ts_Error* error
);

// Concatenate tensors along a dimension
TS_TORCH_API ts_TensorHandle ts_tensor_cat(
    ts_TensorHandle* tensors,
    size_t num_tensors,
    int64_t dim,
    ts_Error* error
);

TS_TORCH_API ts_TensorHandle ts_tensor_sum(
    ts_TensorHandle tensor,
    ts_Error* error
);

TS_TORCH_API ts_TensorHandle ts_tensor_sum_dim(
    ts_TensorHandle tensor,
    int64_t dim,
    int keepdim,
    ts_Error* error
);

TS_TORCH_API ts_TensorHandle ts_tensor_mean(
    ts_TensorHandle tensor,
    ts_Error* error
);

TS_TORCH_API ts_TensorHandle ts_tensor_mean_dim(
    ts_TensorHandle tensor,
    int64_t dim,
    int keepdim,
    ts_Error* error
);

// Activation functions
TS_TORCH_API ts_TensorHandle ts_tensor_relu(
    ts_TensorHandle tensor,
    ts_Error* error
);

TS_TORCH_API ts_TensorHandle ts_tensor_sigmoid(
    ts_TensorHandle tensor,
    ts_Error* error
);

TS_TORCH_API ts_TensorHandle ts_tensor_softmax(
    ts_TensorHandle tensor,
    int64_t dim,
    ts_Error* error
);

TS_TORCH_API ts_TensorHandle ts_tensor_tanh(
    ts_TensorHandle tensor,
    ts_Error* error
);

TS_TORCH_API ts_TensorHandle ts_tensor_log(
    ts_TensorHandle tensor,
    ts_Error* error
);

TS_TORCH_API ts_TensorHandle ts_tensor_exp(
    ts_TensorHandle tensor,
    ts_Error* error
);

TS_TORCH_API ts_TensorHandle ts_tensor_neg(
    ts_TensorHandle tensor,
    ts_Error* error
);

TS_TORCH_API ts_TensorHandle ts_tensor_sqrt(
    ts_TensorHandle tensor,
    ts_Error* error
);

TS_TORCH_API ts_TensorHandle ts_tensor_log_softmax(
    ts_TensorHandle tensor,
    int64_t dim,
    ts_Error* error
);

// Scalar operations
TS_TORCH_API ts_TensorHandle ts_tensor_add_scalar(
    ts_TensorHandle tensor,
    double scalar,
    ts_Error* error
);

TS_TORCH_API ts_TensorHandle ts_tensor_sub_scalar(
    ts_TensorHandle tensor,
    double scalar,
    ts_Error* error
);

TS_TORCH_API ts_TensorHandle ts_tensor_mul_scalar(
    ts_TensorHandle tensor,
    double scalar,
    ts_Error* error
);

TS_TORCH_API ts_TensorHandle ts_tensor_div_scalar(
    ts_TensorHandle tensor,
    double scalar,
    ts_Error* error
);

// Gradient operations
TS_TORCH_API void ts_tensor_zero_grad(
    ts_TensorHandle tensor,
    ts_Error* error
);

// Autograd operations
TS_TORCH_API void ts_tensor_backward(
    ts_TensorHandle tensor,
    ts_Error* error
);

TS_TORCH_API ts_TensorHandle ts_tensor_grad(
    ts_TensorHandle tensor,
    ts_Error* error
);

TS_TORCH_API void ts_tensor_set_requires_grad(
    ts_TensorHandle tensor,
    int requires_grad,
    ts_Error* error
);

TS_TORCH_API int ts_tensor_requires_grad(
    ts_TensorHandle tensor,
    ts_Error* error
);

TS_TORCH_API int ts_tensor_is_leaf(
    ts_TensorHandle tensor,
    ts_Error* error
);

// Scope management for automatic memory cleanup
TS_TORCH_API ts_ScopeHandle ts_scope_begin(void);

TS_TORCH_API void ts_scope_end(ts_ScopeHandle scope);

TS_TORCH_API void ts_scope_register_tensor(
    ts_ScopeHandle scope,
    ts_TensorHandle tensor
);

TS_TORCH_API ts_TensorHandle ts_scope_escape_tensor(
    ts_ScopeHandle scope,
    ts_TensorHandle tensor
);

// Device operations
TS_TORCH_API int ts_cuda_is_available(void);

TS_TORCH_API int ts_cuda_device_count(void);

TS_TORCH_API ts_TensorHandle ts_tensor_to_device(
    ts_TensorHandle tensor,
    ts_DeviceType device,
    int device_index,
    ts_Error* error
);

TS_TORCH_API ts_TensorHandle ts_tensor_cpu(
    ts_TensorHandle tensor,
    ts_Error* error
);

TS_TORCH_API ts_TensorHandle ts_tensor_cuda(
    ts_TensorHandle tensor,
    int device_index,
    ts_Error* error
);

// Loss functions
TS_TORCH_API ts_TensorHandle ts_tensor_nll_loss(
    ts_TensorHandle log_probs,
    ts_TensorHandle targets,
    ts_Error* error
);

TS_TORCH_API ts_TensorHandle ts_tensor_cross_entropy_loss(
    ts_TensorHandle logits,
    ts_TensorHandle targets,
    ts_Error* error
);

TS_TORCH_API ts_TensorHandle ts_tensor_mse_loss(
    ts_TensorHandle input,
    ts_TensorHandle target,
    ts_Error* error
);

// In-place operations (for optimizer updates)
TS_TORCH_API void ts_tensor_sub_inplace(
    ts_TensorHandle tensor,
    ts_TensorHandle other,
    ts_Error* error
);

TS_TORCH_API void ts_tensor_add_scaled_inplace(
    ts_TensorHandle tensor,
    ts_TensorHandle other,
    double scalar,
    ts_Error* error
);

// Comparison operations
TS_TORCH_API ts_TensorHandle ts_tensor_eq(
    ts_TensorHandle a,
    ts_TensorHandle b,
    ts_Error* error
);

TS_TORCH_API ts_TensorHandle ts_tensor_ne(
    ts_TensorHandle a,
    ts_TensorHandle b,
    ts_Error* error
);

TS_TORCH_API ts_TensorHandle ts_tensor_lt(
    ts_TensorHandle a,
    ts_TensorHandle b,
    ts_Error* error
);

TS_TORCH_API ts_TensorHandle ts_tensor_le(
    ts_TensorHandle a,
    ts_TensorHandle b,
    ts_Error* error
);

TS_TORCH_API ts_TensorHandle ts_tensor_gt(
    ts_TensorHandle a,
    ts_TensorHandle b,
    ts_Error* error
);

TS_TORCH_API ts_TensorHandle ts_tensor_ge(
    ts_TensorHandle a,
    ts_TensorHandle b,
    ts_Error* error
);

// Random tensor creation
TS_TORCH_API ts_TensorHandle ts_tensor_rand(
    const int64_t* shape,
    size_t ndim,
    ts_DType dtype,
    ts_DeviceType device,
    int device_index,
    ts_Error* error
);

// Variance operation
TS_TORCH_API ts_TensorHandle ts_tensor_var(
    ts_TensorHandle tensor,
    int64_t dim,
    int unbiased,
    int keepdim,
    ts_Error* error
);

// Conv2d operation
TS_TORCH_API ts_TensorHandle ts_tensor_conv2d(
    ts_TensorHandle input,
    ts_TensorHandle weight,
    ts_TensorHandle bias,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_h,
    int64_t padding_w,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t groups,
    ts_Error* error
);

// MaxPool2d operation
TS_TORCH_API ts_TensorHandle ts_tensor_max_pool2d(
    ts_TensorHandle input,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_h,
    int64_t padding_w,
    ts_Error* error
);

// AvgPool2d operation
TS_TORCH_API ts_TensorHandle ts_tensor_avg_pool2d(
    ts_TensorHandle input,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_h,
    int64_t padding_w,
    ts_Error* error
);

// Dropout operation (training mode only - returns input * mask / (1-p))
TS_TORCH_API ts_TensorHandle ts_tensor_dropout(
    ts_TensorHandle input,
    double p,
    int training,
    ts_Error* error
);

// BatchNorm2d operation
TS_TORCH_API ts_TensorHandle ts_tensor_batch_norm(
    ts_TensorHandle input,
    ts_TensorHandle weight,
    ts_TensorHandle bias,
    ts_TensorHandle running_mean,
    ts_TensorHandle running_var,
    int training,
    double momentum,
    double eps,
    ts_Error* error
);

// LayerNorm operation
TS_TORCH_API ts_TensorHandle ts_tensor_layer_norm(
    ts_TensorHandle input,
    const int64_t* normalized_shape,
    size_t normalized_shape_len,
    ts_TensorHandle weight,
    ts_TensorHandle bias,
    double eps,
    ts_Error* error
);

// Index select operation - selects elements along a dimension using an index tensor
TS_TORCH_API ts_TensorHandle ts_tensor_index_select(
    ts_TensorHandle tensor,
    int64_t dim,
    ts_TensorHandle index,
    ts_Error* error
);

// Argmax operation - returns indices of maximum values along a dimension
TS_TORCH_API ts_TensorHandle ts_tensor_argmax(
    ts_TensorHandle tensor,
    int64_t dim,
    int keepdim,
    ts_Error* error
);

// Narrow operation - zero-copy slice along a dimension (returns view)
TS_TORCH_API ts_TensorHandle ts_tensor_narrow(
    ts_TensorHandle tensor,
    int64_t dim,
    int64_t start,
    int64_t length,
    ts_Error* error
);

// Element-wise minimum of two tensors
TS_TORCH_API ts_TensorHandle ts_tensor_minimum(
    ts_TensorHandle a,
    ts_TensorHandle b,
    ts_Error* error
);

// Element-wise maximum of two tensors
TS_TORCH_API ts_TensorHandle ts_tensor_maximum(
    ts_TensorHandle a,
    ts_TensorHandle b,
    ts_Error* error
);

// Clamp tensor values to range [min, max]
TS_TORCH_API ts_TensorHandle ts_tensor_clamp(
    ts_TensorHandle tensor,
    double min_val,
    double max_val,
    ts_Error* error
);

// Clamp tensor values to minimum
TS_TORCH_API ts_TensorHandle ts_tensor_clamp_min(
    ts_TensorHandle tensor,
    double min_val,
    ts_Error* error
);

// Clamp tensor values to maximum
TS_TORCH_API ts_TensorHandle ts_tensor_clamp_max(
    ts_TensorHandle tensor,
    double max_val,
    ts_Error* error
);

// ============================================================================
// In-Place Operations (Phase 4: Memory Efficient Updates)
// ============================================================================

/**
 * In-place addition: tensor += other
 *
 * WARNING: Will error if tensor is a leaf with requires_grad=true,
 * matching PyTorch semantics. Use ts_tensor_optim_add_ for optimizer updates.
 *
 * @param tensor - Tensor to modify in-place
 * @param other - Tensor to add
 * @param error - Error output
 */
TS_TORCH_API void ts_tensor_add_(
    ts_TensorHandle tensor,
    ts_TensorHandle other,
    ts_Error* error
);

/**
 * In-place subtraction: tensor -= other
 *
 * @param tensor - Tensor to modify in-place
 * @param other - Tensor to subtract
 * @param error - Error output
 */
TS_TORCH_API void ts_tensor_sub_(
    ts_TensorHandle tensor,
    ts_TensorHandle other,
    ts_Error* error
);

/**
 * In-place multiplication: tensor *= other
 *
 * @param tensor - Tensor to modify in-place
 * @param other - Tensor to multiply with
 * @param error - Error output
 */
TS_TORCH_API void ts_tensor_mul_(
    ts_TensorHandle tensor,
    ts_TensorHandle other,
    ts_Error* error
);

/**
 * In-place scalar multiplication: tensor *= scalar
 *
 * @param tensor - Tensor to modify in-place
 * @param scalar - Scalar to multiply with
 * @param error - Error output
 */
TS_TORCH_API void ts_tensor_mul_scalar_(
    ts_TensorHandle tensor,
    double scalar,
    ts_Error* error
);

/**
 * In-place addition with alpha: tensor += alpha * other
 *
 * @param tensor - Tensor to modify in-place
 * @param other - Tensor to add
 * @param alpha - Scaling factor for other
 * @param error - Error output
 */
TS_TORCH_API void ts_tensor_add_alpha_(
    ts_TensorHandle tensor,
    ts_TensorHandle other,
    double alpha,
    ts_Error* error
);

/**
 * OPTIMIZER-ONLY in-place addition: tensor.data() += alpha * other
 *
 * Uses .data() to bypass autograd. ONLY safe in optimizer.step() context.
 * This allows updating leaf tensors without triggering autograd errors.
 *
 * @param tensor - Tensor to modify in-place
 * @param other - Tensor to add
 * @param alpha - Scaling factor (default 1.0)
 * @param error - Error output
 */
TS_TORCH_API void ts_tensor_optim_add_(
    ts_TensorHandle tensor,
    ts_TensorHandle other,
    double alpha,
    ts_Error* error
);

/**
 * OPTIMIZER-ONLY zero gradient: tensor.grad.zero_() if exists
 *
 * @param tensor - Tensor whose gradient to zero
 * @param error - Error output
 */
TS_TORCH_API void ts_tensor_zero_grad_(
    ts_TensorHandle tensor,
    ts_Error* error
);

/**
 * In-place division: tensor /= other
 *
 * @param tensor - Tensor to modify in-place
 * @param other - Tensor to divide by
 * @param error - Error output
 */
TS_TORCH_API void ts_tensor_div_(
    ts_TensorHandle tensor,
    ts_TensorHandle other,
    ts_Error* error
);

/**
 * In-place scalar division: tensor /= scalar
 *
 * @param tensor - Tensor to modify in-place
 * @param scalar - Scalar to divide by
 * @param error - Error output
 */
TS_TORCH_API void ts_tensor_div_scalar_(
    ts_TensorHandle tensor,
    double scalar,
    ts_Error* error
);

// ============================================================================
// Out= Operations (Pre-allocated output tensors for reduced allocation)
// ============================================================================

/**
 * Addition with pre-allocated output: out = a + b
 * Avoids allocation overhead by writing to existing tensor.
 *
 * @param a - First input tensor
 * @param b - Second input tensor
 * @param out - Pre-allocated output tensor (will be overwritten)
 * @param error - Error output
 */
TS_TORCH_API void ts_tensor_add_out(
    ts_TensorHandle a,
    ts_TensorHandle b,
    ts_TensorHandle out,
    ts_Error* error
);

/**
 * Subtraction with pre-allocated output: out = a - b
 */
TS_TORCH_API void ts_tensor_sub_out(
    ts_TensorHandle a,
    ts_TensorHandle b,
    ts_TensorHandle out,
    ts_Error* error
);

/**
 * Multiplication with pre-allocated output: out = a * b
 */
TS_TORCH_API void ts_tensor_mul_out(
    ts_TensorHandle a,
    ts_TensorHandle b,
    ts_TensorHandle out,
    ts_Error* error
);

/**
 * Division with pre-allocated output: out = a / b
 */
TS_TORCH_API void ts_tensor_div_out(
    ts_TensorHandle a,
    ts_TensorHandle b,
    ts_TensorHandle out,
    ts_Error* error
);

/**
 * Matrix multiplication with pre-allocated output: out = a @ b
 */
TS_TORCH_API void ts_tensor_matmul_out(
    ts_TensorHandle a,
    ts_TensorHandle b,
    ts_TensorHandle out,
    ts_Error* error
);

// ============================================================================
// Fused Operations (Phase 3: Reduced kernel launch overhead)
// ============================================================================

/**
 * Fused linear + ReLU: relu(x @ W^T + b)
 * Combines linear layer and ReLU activation in a single operation.
 *
 * @param input - Input tensor
 * @param weight - Weight tensor
 * @param bias - Optional bias tensor (can be nullptr)
 * @param error - Error output
 * @return Result tensor, or nullptr on error
 */
TS_TORCH_API ts_TensorHandle ts_tensor_linear_relu(
    ts_TensorHandle input,
    ts_TensorHandle weight,
    ts_TensorHandle bias,
    ts_Error* error
);

/**
 * Fused linear + Sigmoid: sigmoid(x @ W^T + b)
 * Combines linear layer and sigmoid activation in a single operation.
 *
 * @param input - Input tensor
 * @param weight - Weight tensor
 * @param bias - Optional bias tensor (can be nullptr)
 * @param error - Error output
 * @return Result tensor, or nullptr on error
 */
TS_TORCH_API ts_TensorHandle ts_tensor_linear_sigmoid(
    ts_TensorHandle input,
    ts_TensorHandle weight,
    ts_TensorHandle bias,
    ts_Error* error
);

/**
 * Fused linear + Tanh: tanh(x @ W^T + b)
 * Combines linear layer and tanh activation in a single operation.
 *
 * @param input - Input tensor
 * @param weight - Weight tensor
 * @param bias - Optional bias tensor (can be nullptr)
 * @param error - Error output
 * @return Result tensor, or nullptr on error
 */
TS_TORCH_API ts_TensorHandle ts_tensor_linear_tanh(
    ts_TensorHandle input,
    ts_TensorHandle weight,
    ts_TensorHandle bias,
    ts_Error* error
);

/**
 * Fused add + ReLU: relu(a + b)
 *
 * @param a - First tensor
 * @param b - Second tensor
 * @param error - Error output
 * @return Result tensor, or nullptr on error
 */
TS_TORCH_API ts_TensorHandle ts_tensor_add_relu(
    ts_TensorHandle a,
    ts_TensorHandle b,
    ts_Error* error
);

// ============================================================================
// Batching API (Phase 2: Native Recording for Reduced FFI Overhead)
// ============================================================================

/**
 * Opaque batch handle for recording mode
 */
typedef struct ts_Batch* ts_BatchHandle;

/**
 * Operation codes for batched execution
 */
typedef enum {
    TS_OP_ADD = 0,
    TS_OP_SUB = 1,
    TS_OP_MUL = 2,
    TS_OP_DIV = 3,
    TS_OP_MATMUL = 4,
    TS_OP_RELU = 5,
    TS_OP_SIGMOID = 6,
    TS_OP_SOFTMAX = 7,
    TS_OP_TANH = 8,
    TS_OP_LINEAR = 9,
    TS_OP_SUM = 10,
    TS_OP_MEAN = 11,
    TS_OP_TRANSPOSE = 12,
    TS_OP_RESHAPE = 13,
} ts_OpCode;

/**
 * Begin a new batch recording session.
 * All tensor operations after this call will record to the batch
 * instead of executing immediately. Returns placeholder handles.
 *
 * @param error - Error output
 * @return Batch handle, or nullptr on error
 */
TS_TORCH_API ts_BatchHandle ts_batch_begin(ts_Error* error);

/**
 * End batch recording and execute all recorded operations.
 * Returns the result of the last operation.
 *
 * @param batch - Batch handle from ts_batch_begin
 * @param error - Error output
 * @return Result tensor handle, or nullptr on error
 */
TS_TORCH_API ts_TensorHandle ts_batch_end(ts_BatchHandle batch, ts_Error* error);

/**
 * Abort batch recording without executing.
 * Frees all placeholders and batch resources.
 *
 * @param batch - Batch handle to abort
 */
TS_TORCH_API void ts_batch_abort(ts_BatchHandle batch);

/**
 * Check if currently in batch recording mode.
 *
 * @return 1 if in batch mode, 0 otherwise
 */
TS_TORCH_API int ts_batch_is_recording(void);

// ============================================================================
// Direct Batched Operations (No recording overhead)
// ============================================================================

/**
 * Chain matrix multiplication: A @ B @ C @ D @ ...
 * Executes all matmuls in a single FFI call.
 *
 * @param tensors - Array of tensor handles to multiply
 * @param count - Number of tensors (minimum 2)
 * @param error - Error output
 * @return Result tensor, or nullptr on error
 */
TS_TORCH_API ts_TensorHandle ts_tensor_chain_matmul(
    ts_TensorHandle* tensors,
    size_t count,
    ts_Error* error
);

/**
 * MLP forward pass: input @ W1 + b1 -> relu -> @ W2 + b2 -> ...
 * Executes full MLP in a single FFI call.
 *
 * @param input - Input tensor
 * @param weights - Array of weight tensors
 * @param biases - Array of bias tensors (can have nullptr elements)
 * @param num_layers - Number of layers
 * @param apply_relu_except_last - 1 to apply ReLU between layers, 0 for no activation
 * @param error - Error output
 * @return Result tensor, or nullptr on error
 */
TS_TORCH_API ts_TensorHandle ts_tensor_mlp_forward(
    ts_TensorHandle input,
    ts_TensorHandle* weights,
    ts_TensorHandle* biases,
    size_t num_layers,
    int apply_relu_except_last,
    ts_Error* error
);

// ============================================================================
// Thread Controls (Phase 6)
// ============================================================================

/**
 * Set the number of threads used by LibTorch for inter-op parallelism.
 * WARNING: This is a global setting that affects all operations.
 *
 * @param num_threads - Number of threads to use (0 = auto)
 */
TS_TORCH_API void ts_set_num_threads(int num_threads);

/**
 * Get the current number of threads used by LibTorch.
 *
 * @return Current number of threads
 */
TS_TORCH_API int ts_get_num_threads(void);

#ifdef __cplusplus
}
#endif

#endif // TS_TORCH_H
