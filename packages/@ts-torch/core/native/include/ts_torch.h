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

// Tensor creation functions
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

#ifdef __cplusplus
}
#endif

#endif // TS_TORCH_H
