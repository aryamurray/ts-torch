#include "ts_torch.h"
#include <torch/torch.h>
#include <cstring>
#include <memory>
#include <vector>
#include <unordered_set>
#include <stdexcept>

// Version string
#define TS_TORCH_VERSION "0.1.0"

// Internal structures
struct ts_Tensor {
    torch::Tensor tensor;

    explicit ts_Tensor(torch::Tensor t) : tensor(std::move(t)) {}
};

struct ts_Module {
    std::shared_ptr<torch::nn::Module> module;

    explicit ts_Module(std::shared_ptr<torch::nn::Module> m) : module(std::move(m)) {}
};

struct ts_Optimizer {
    std::shared_ptr<torch::optim::Optimizer> optimizer;

    explicit ts_Optimizer(std::shared_ptr<torch::optim::Optimizer> opt)
        : optimizer(std::move(opt)) {}
};

struct ts_Scope {
    std::unordered_set<ts_TensorHandle> tensors;
    std::unordered_set<ts_TensorHandle> escaped;
};

// Thread-local scope stack
thread_local std::vector<std::unique_ptr<ts_Scope>> g_scope_stack;

// Helper function to convert ts_DType to torch::ScalarType
static torch::ScalarType dtype_to_scalar_type(ts_DType dtype) {
    switch (dtype) {
        case TS_DTYPE_FLOAT32: return torch::kFloat32;
        case TS_DTYPE_FLOAT64: return torch::kFloat64;
        case TS_DTYPE_INT32: return torch::kInt32;
        case TS_DTYPE_INT64: return torch::kInt64;
        case TS_DTYPE_BOOL: return torch::kBool;
        case TS_DTYPE_FLOAT16: return torch::kFloat16;
        case TS_DTYPE_BFLOAT16: return torch::kBFloat16;
        case TS_DTYPE_UINT8: return torch::kUInt8;
        case TS_DTYPE_INT8: return torch::kInt8;
        case TS_DTYPE_INT16: return torch::kInt16;
        default: return torch::kFloat32;
    }
}

// Helper function to convert torch::ScalarType to ts_DType
static ts_DType scalar_type_to_dtype(torch::ScalarType scalar_type) {
    switch (scalar_type) {
        case torch::kFloat32: return TS_DTYPE_FLOAT32;
        case torch::kFloat64: return TS_DTYPE_FLOAT64;
        case torch::kInt32: return TS_DTYPE_INT32;
        case torch::kInt64: return TS_DTYPE_INT64;
        case torch::kBool: return TS_DTYPE_BOOL;
        case torch::kFloat16: return TS_DTYPE_FLOAT16;
        case torch::kBFloat16: return TS_DTYPE_BFLOAT16;
        case torch::kUInt8: return TS_DTYPE_UINT8;
        case torch::kInt8: return TS_DTYPE_INT8;
        case torch::kInt16: return TS_DTYPE_INT16;
        default: return TS_DTYPE_FLOAT32;
    }
}

// Helper function to convert device type
static torch::Device make_device(ts_DeviceType device_type, int device_index) {
    switch (device_type) {
        case TS_DEVICE_CPU:
            return torch::Device(torch::kCPU);
        case TS_DEVICE_CUDA:
            return torch::Device(torch::kCUDA, device_index);
        case TS_DEVICE_MPS:
            return torch::Device(torch::kMPS);
        default:
            return torch::Device(torch::kCPU);
    }
}

// Helper function to convert torch::Device to ts_DeviceType
static ts_DeviceType device_to_device_type(const torch::Device& device) {
    switch (device.type()) {
        case torch::kCPU: return TS_DEVICE_CPU;
        case torch::kCUDA: return TS_DEVICE_CUDA;
        case torch::kMPS: return TS_DEVICE_MPS;
        default: return TS_DEVICE_CPU;
    }
}

// Helper function to set error
static void set_error(ts_Error* error, int code, const char* message) {
    if (error) {
        error->code = code;
        std::strncpy(error->message, message, sizeof(error->message) - 1);
        error->message[sizeof(error->message) - 1] = '\0';
    }
}

// Helper function to register tensor in current scope
static void register_in_scope(ts_TensorHandle handle) {
    if (!g_scope_stack.empty()) {
        g_scope_stack.back()->tensors.insert(handle);
    }
}

// Version information
const char* ts_version(void) {
    return TS_TORCH_VERSION;
}

// Error handling
void ts_error_clear(ts_Error* error) {
    if (error) {
        error->code = 0;
        error->message[0] = '\0';
    }
}

int ts_error_occurred(const ts_Error* error) {
    return error && error->code != 0;
}

// Tensor creation functions
ts_TensorHandle ts_tensor_zeros(
    const int64_t* shape,
    size_t ndim,
    ts_DType dtype,
    ts_DeviceType device,
    int device_index,
    ts_Error* error
) {
    try {
        std::vector<int64_t> shape_vec(shape, shape + ndim);
        auto options = torch::TensorOptions()
            .dtype(dtype_to_scalar_type(dtype))
            .device(make_device(device, device_index));

        auto tensor = torch::zeros(shape_vec, options);
        auto* handle = new ts_Tensor(std::move(tensor));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

ts_TensorHandle ts_tensor_ones(
    const int64_t* shape,
    size_t ndim,
    ts_DType dtype,
    ts_DeviceType device,
    int device_index,
    ts_Error* error
) {
    try {
        std::vector<int64_t> shape_vec(shape, shape + ndim);
        auto options = torch::TensorOptions()
            .dtype(dtype_to_scalar_type(dtype))
            .device(make_device(device, device_index));

        auto tensor = torch::ones(shape_vec, options);
        auto* handle = new ts_Tensor(std::move(tensor));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

ts_TensorHandle ts_tensor_randn(
    const int64_t* shape,
    size_t ndim,
    ts_DType dtype,
    ts_DeviceType device,
    int device_index,
    ts_Error* error
) {
    try {
        std::vector<int64_t> shape_vec(shape, shape + ndim);
        auto options = torch::TensorOptions()
            .dtype(dtype_to_scalar_type(dtype))
            .device(make_device(device, device_index));

        auto tensor = torch::randn(shape_vec, options);
        auto* handle = new ts_Tensor(std::move(tensor));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

ts_TensorHandle ts_tensor_empty(
    const int64_t* shape,
    size_t ndim,
    ts_DType dtype,
    ts_DeviceType device,
    int device_index,
    ts_Error* error
) {
    try {
        std::vector<int64_t> shape_vec(shape, shape + ndim);
        auto options = torch::TensorOptions()
            .dtype(dtype_to_scalar_type(dtype))
            .device(make_device(device, device_index));

        auto tensor = torch::empty(shape_vec, options);
        auto* handle = new ts_Tensor(std::move(tensor));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

ts_TensorHandle ts_tensor_from_buffer(
    const void* data,
    const int64_t* shape,
    size_t ndim,
    ts_DType dtype,
    ts_DeviceType device,
    int device_index,
    ts_Error* error
) {
    try {
        std::vector<int64_t> shape_vec(shape, shape + ndim);
        auto scalar_type = dtype_to_scalar_type(dtype);

        // Create tensor from data (copies the data)
        auto options = torch::TensorOptions()
            .dtype(scalar_type)
            .device(torch::kCPU);

        auto tensor = torch::from_blob(
            const_cast<void*>(data),
            shape_vec,
            options
        ).clone();

        // Move to target device if not CPU
        if (device != TS_DEVICE_CPU) {
            tensor = tensor.to(make_device(device, device_index));
        }

        auto* handle = new ts_Tensor(std::move(tensor));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

// Tensor properties
int64_t ts_tensor_ndim(ts_TensorHandle tensor, ts_Error* error) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return -1;
        }
        return static_cast<int64_t>(tensor->tensor.dim());
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return -1;
    }
}

int64_t ts_tensor_size(ts_TensorHandle tensor, int64_t dim, ts_Error* error) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return -1;
        }
        return tensor->tensor.size(dim);
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return -1;
    }
}

void ts_tensor_shape(
    ts_TensorHandle tensor,
    int64_t* shape,
    size_t* ndim,
    ts_Error* error
) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return;
        }

        auto sizes = tensor->tensor.sizes();
        *ndim = sizes.size();

        for (size_t i = 0; i < sizes.size(); ++i) {
            shape[i] = sizes[i];
        }
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
    }
}

ts_DType ts_tensor_dtype(ts_TensorHandle tensor, ts_Error* error) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return TS_DTYPE_FLOAT32;
        }
        return scalar_type_to_dtype(tensor->tensor.scalar_type());
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return TS_DTYPE_FLOAT32;
    }
}

int64_t ts_tensor_numel(ts_TensorHandle tensor, ts_Error* error) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return -1;
        }
        return tensor->tensor.numel();
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return -1;
    }
}

ts_DeviceType ts_tensor_device_type(ts_TensorHandle tensor, ts_Error* error) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return TS_DEVICE_CPU;
        }
        return device_to_device_type(tensor->tensor.device());
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return TS_DEVICE_CPU;
    }
}

int ts_tensor_device_index(ts_TensorHandle tensor, ts_Error* error) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return -1;
        }
        return tensor->tensor.device().has_index() ? tensor->tensor.device().index() : 0;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return -1;
    }
}

// Tensor memory management
void ts_tensor_delete(ts_TensorHandle tensor) {
    if (tensor) {
        delete tensor;
    }
}

ts_TensorHandle ts_tensor_clone(ts_TensorHandle tensor, ts_Error* error) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto cloned = tensor->tensor.clone();
        auto* handle = new ts_Tensor(std::move(cloned));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

ts_TensorHandle ts_tensor_detach(ts_TensorHandle tensor, ts_Error* error) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto detached = tensor->tensor.detach();
        auto* handle = new ts_Tensor(std::move(detached));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

void* ts_tensor_data_ptr(ts_TensorHandle tensor, ts_Error* error) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }
        return tensor->tensor.data_ptr();
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

void ts_tensor_copy_to_buffer(
    ts_TensorHandle tensor,
    void* buffer,
    size_t buffer_size,
    ts_Error* error
) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return;
        }

        // Ensure tensor is contiguous and on CPU
        auto cpu_tensor = tensor->tensor.cpu().contiguous();

        size_t tensor_bytes = cpu_tensor.numel() * cpu_tensor.element_size();
        if (buffer_size < tensor_bytes) {
            set_error(error, 1, "Buffer size too small");
            return;
        }

        std::memcpy(buffer, cpu_tensor.data_ptr(), tensor_bytes);
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
    }
}

// Tensor operations
ts_TensorHandle ts_tensor_add(
    ts_TensorHandle a,
    ts_TensorHandle b,
    ts_Error* error
) {
    try {
        if (!a || !b) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = a->tensor + b->tensor;
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

ts_TensorHandle ts_tensor_sub(
    ts_TensorHandle a,
    ts_TensorHandle b,
    ts_Error* error
) {
    try {
        if (!a || !b) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = a->tensor - b->tensor;
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

ts_TensorHandle ts_tensor_mul(
    ts_TensorHandle a,
    ts_TensorHandle b,
    ts_Error* error
) {
    try {
        if (!a || !b) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = a->tensor * b->tensor;
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

ts_TensorHandle ts_tensor_div(
    ts_TensorHandle a,
    ts_TensorHandle b,
    ts_Error* error
) {
    try {
        if (!a || !b) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = a->tensor / b->tensor;
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

ts_TensorHandle ts_tensor_matmul(
    ts_TensorHandle a,
    ts_TensorHandle b,
    ts_Error* error
) {
    try {
        if (!a || !b) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = torch::matmul(a->tensor, b->tensor);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

ts_TensorHandle ts_tensor_transpose(
    ts_TensorHandle tensor,
    int64_t dim0,
    int64_t dim1,
    ts_Error* error
) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = tensor->tensor.transpose(dim0, dim1);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

ts_TensorHandle ts_tensor_reshape(
    ts_TensorHandle tensor,
    const int64_t* shape,
    size_t ndim,
    ts_Error* error
) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        std::vector<int64_t> shape_vec(shape, shape + ndim);
        auto result = tensor->tensor.reshape(shape_vec);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

ts_TensorHandle ts_tensor_cat(
    ts_TensorHandle* tensors,
    size_t num_tensors,
    int64_t dim,
    ts_Error* error
) {
    try {
        if (!tensors || num_tensors == 0) {
            set_error(error, 1, "Null or empty tensor array");
            return nullptr;
        }

        // Build vector of torch::Tensor from handles
        std::vector<torch::Tensor> tensor_vec;
        tensor_vec.reserve(num_tensors);

        for (size_t i = 0; i < num_tensors; i++) {
            if (!tensors[i]) {
                set_error(error, 1, "Null tensor in array");
                return nullptr;
            }
            tensor_vec.push_back(tensors[i]->tensor);
        }

        // Concatenate tensors along specified dimension
        auto result = torch::cat(tensor_vec, dim);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

ts_TensorHandle ts_tensor_sum(ts_TensorHandle tensor, ts_Error* error) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = tensor->tensor.sum();
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

ts_TensorHandle ts_tensor_sum_dim(
    ts_TensorHandle tensor,
    int64_t dim,
    int keepdim,
    ts_Error* error
) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = tensor->tensor.sum(dim, keepdim != 0);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

ts_TensorHandle ts_tensor_mean(ts_TensorHandle tensor, ts_Error* error) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = tensor->tensor.mean();
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

ts_TensorHandle ts_tensor_mean_dim(
    ts_TensorHandle tensor,
    int64_t dim,
    int keepdim,
    ts_Error* error
) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = tensor->tensor.mean(dim, keepdim != 0);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

// Activation functions
ts_TensorHandle ts_tensor_relu(ts_TensorHandle tensor, ts_Error* error) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = torch::relu(tensor->tensor);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

ts_TensorHandle ts_tensor_sigmoid(ts_TensorHandle tensor, ts_Error* error) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = torch::sigmoid(tensor->tensor);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

ts_TensorHandle ts_tensor_softmax(
    ts_TensorHandle tensor,
    int64_t dim,
    ts_Error* error
) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = torch::softmax(tensor->tensor, dim);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

ts_TensorHandle ts_tensor_tanh(ts_TensorHandle tensor, ts_Error* error) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = torch::tanh(tensor->tensor);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

ts_TensorHandle ts_tensor_log(ts_TensorHandle tensor, ts_Error* error) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = torch::log(tensor->tensor);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

ts_TensorHandle ts_tensor_exp(ts_TensorHandle tensor, ts_Error* error) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = torch::exp(tensor->tensor);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

ts_TensorHandle ts_tensor_neg(ts_TensorHandle tensor, ts_Error* error) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = -tensor->tensor;
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

ts_TensorHandle ts_tensor_sqrt(ts_TensorHandle tensor, ts_Error* error) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = torch::sqrt(tensor->tensor);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

ts_TensorHandle ts_tensor_log_softmax(
    ts_TensorHandle tensor,
    int64_t dim,
    ts_Error* error
) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = torch::log_softmax(tensor->tensor, dim);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

// Scalar operations
ts_TensorHandle ts_tensor_add_scalar(
    ts_TensorHandle tensor,
    double scalar,
    ts_Error* error
) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = tensor->tensor + scalar;
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

ts_TensorHandle ts_tensor_sub_scalar(
    ts_TensorHandle tensor,
    double scalar,
    ts_Error* error
) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = tensor->tensor - scalar;
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

ts_TensorHandle ts_tensor_mul_scalar(
    ts_TensorHandle tensor,
    double scalar,
    ts_Error* error
) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = tensor->tensor * scalar;
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

ts_TensorHandle ts_tensor_div_scalar(
    ts_TensorHandle tensor,
    double scalar,
    ts_Error* error
) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = tensor->tensor / scalar;
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

// Gradient operations
void ts_tensor_zero_grad(ts_TensorHandle tensor, ts_Error* error) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return;
        }

        if (tensor->tensor.grad().defined()) {
            tensor->tensor.grad().zero_();
        }
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
    }
}

// Autograd operations
void ts_tensor_backward(ts_TensorHandle tensor, ts_Error* error) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return;
        }

        tensor->tensor.backward();
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
    }
}

ts_TensorHandle ts_tensor_grad(ts_TensorHandle tensor, ts_Error* error) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        // No gradient is a normal condition, not an error - just return nullptr
        if (!tensor->tensor.grad().defined()) {
            return nullptr;
        }

        auto grad = tensor->tensor.grad();
        auto* handle = new ts_Tensor(std::move(grad));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

void ts_tensor_set_requires_grad(
    ts_TensorHandle tensor,
    int requires_grad,
    ts_Error* error
) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return;
        }

        tensor->tensor.set_requires_grad(requires_grad != 0);
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
    }
}

int ts_tensor_requires_grad(ts_TensorHandle tensor, ts_Error* error) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return 0;
        }

        return tensor->tensor.requires_grad() ? 1 : 0;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return 0;
    }
}

int ts_tensor_is_leaf(ts_TensorHandle tensor, ts_Error* error) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return 0;
        }

        return tensor->tensor.is_leaf() ? 1 : 0;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return 0;
    }
}

// Scope management
ts_ScopeHandle ts_scope_begin(void) {
    auto scope = std::make_unique<ts_Scope>();
    auto* handle = scope.get();
    g_scope_stack.push_back(std::move(scope));
    return handle;
}

void ts_scope_end(ts_ScopeHandle scope) {
    if (g_scope_stack.empty() || g_scope_stack.back().get() != scope) {
        return;
    }

    // Delete all tensors that weren't escaped
    for (auto* tensor : scope->tensors) {
        if (scope->escaped.find(tensor) == scope->escaped.end()) {
            ts_tensor_delete(tensor);
        }
    }

    g_scope_stack.pop_back();
}

void ts_scope_register_tensor(ts_ScopeHandle scope, ts_TensorHandle tensor) {
    if (scope && tensor) {
        scope->tensors.insert(tensor);
    }
}

ts_TensorHandle ts_scope_escape_tensor(
    ts_ScopeHandle scope,
    ts_TensorHandle tensor
) {
    if (scope && tensor) {
        scope->escaped.insert(tensor);
    }
    return tensor;
}

// Device operations
int ts_cuda_is_available(void) {
    return torch::cuda::is_available() ? 1 : 0;
}

int ts_cuda_device_count(void) {
    return static_cast<int>(torch::cuda::device_count());
}

ts_TensorHandle ts_tensor_to_device(
    ts_TensorHandle tensor,
    ts_DeviceType device,
    int device_index,
    ts_Error* error
) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = tensor->tensor.to(make_device(device, device_index));
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

ts_TensorHandle ts_tensor_cpu(ts_TensorHandle tensor, ts_Error* error) {
    return ts_tensor_to_device(tensor, TS_DEVICE_CPU, 0, error);
}

ts_TensorHandle ts_tensor_cuda(
    ts_TensorHandle tensor,
    int device_index,
    ts_Error* error
) {
    return ts_tensor_to_device(tensor, TS_DEVICE_CUDA, device_index, error);
}

// Loss functions
ts_TensorHandle ts_tensor_nll_loss(
    ts_TensorHandle log_probs,
    ts_TensorHandle targets,
    ts_Error* error
) {
    try {
        if (!log_probs || !targets) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        // Use PyTorch's nll_loss which handles the indexing properly
        auto result = torch::nll_loss(log_probs->tensor, targets->tensor);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

ts_TensorHandle ts_tensor_cross_entropy_loss(
    ts_TensorHandle logits,
    ts_TensorHandle targets,
    ts_Error* error
) {
    try {
        if (!logits || !targets) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        // Use PyTorch's cross_entropy which combines log_softmax and nll_loss
        auto result = torch::cross_entropy_loss(logits->tensor, targets->tensor);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

ts_TensorHandle ts_tensor_mse_loss(
    ts_TensorHandle input,
    ts_TensorHandle target,
    ts_Error* error
) {
    try {
        if (!input || !target) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = torch::mse_loss(input->tensor, target->tensor);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

// In-place operations (for optimizer updates)
void ts_tensor_sub_inplace(
    ts_TensorHandle tensor,
    ts_TensorHandle other,
    ts_Error* error
) {
    try {
        if (!tensor || !other) {
            set_error(error, 1, "Null tensor handle");
            return;
        }

        // Use .data() to bypass autograd and modify in-place
        tensor->tensor.data().sub_(other->tensor);
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
    }
}

void ts_tensor_add_scaled_inplace(
    ts_TensorHandle tensor,
    ts_TensorHandle other,
    double scalar,
    ts_Error* error
) {
    try {
        if (!tensor || !other) {
            set_error(error, 1, "Null tensor handle");
            return;
        }

        // Use .data() to bypass autograd: tensor.data += scalar * other
        tensor->tensor.data().add_(other->tensor, scalar);
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
    }
}

// Comparison operations
ts_TensorHandle ts_tensor_eq(
    ts_TensorHandle a,
    ts_TensorHandle b,
    ts_Error* error
) {
    try {
        if (!a || !b) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = a->tensor.eq(b->tensor);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

ts_TensorHandle ts_tensor_ne(
    ts_TensorHandle a,
    ts_TensorHandle b,
    ts_Error* error
) {
    try {
        if (!a || !b) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = a->tensor.ne(b->tensor);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

ts_TensorHandle ts_tensor_lt(
    ts_TensorHandle a,
    ts_TensorHandle b,
    ts_Error* error
) {
    try {
        if (!a || !b) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = a->tensor.lt(b->tensor);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

ts_TensorHandle ts_tensor_le(
    ts_TensorHandle a,
    ts_TensorHandle b,
    ts_Error* error
) {
    try {
        if (!a || !b) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = a->tensor.le(b->tensor);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

ts_TensorHandle ts_tensor_gt(
    ts_TensorHandle a,
    ts_TensorHandle b,
    ts_Error* error
) {
    try {
        if (!a || !b) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = a->tensor.gt(b->tensor);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

ts_TensorHandle ts_tensor_ge(
    ts_TensorHandle a,
    ts_TensorHandle b,
    ts_Error* error
) {
    try {
        if (!a || !b) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = a->tensor.ge(b->tensor);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

// Random tensor creation (uniform [0, 1))
ts_TensorHandle ts_tensor_rand(
    const int64_t* shape,
    size_t ndim,
    ts_DType dtype,
    ts_DeviceType device,
    int device_index,
    ts_Error* error
) {
    try {
        std::vector<int64_t> shape_vec(shape, shape + ndim);
        auto options = torch::TensorOptions()
            .dtype(dtype_to_scalar_type(dtype))
            .device(make_device(device, device_index));

        auto tensor = torch::rand(shape_vec, options);
        auto* handle = new ts_Tensor(std::move(tensor));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

// Variance operation
ts_TensorHandle ts_tensor_var(
    ts_TensorHandle tensor,
    int64_t dim,
    int unbiased,
    int keepdim,
    ts_Error* error
) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = tensor->tensor.var(dim, unbiased != 0, keepdim != 0);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

// Conv2d operation
ts_TensorHandle ts_tensor_conv2d(
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
) {
    try {
        if (!input || !weight) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        torch::Tensor bias_tensor;
        if (bias) {
            bias_tensor = bias->tensor;
        }

        auto result = torch::conv2d(
            input->tensor,
            weight->tensor,
            bias_tensor,
            {stride_h, stride_w},
            {padding_h, padding_w},
            {dilation_h, dilation_w},
            groups
        );

        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

// MaxPool2d operation
ts_TensorHandle ts_tensor_max_pool2d(
    ts_TensorHandle input,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_h,
    int64_t padding_w,
    ts_Error* error
) {
    try {
        if (!input) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = torch::max_pool2d(
            input->tensor,
            {kernel_h, kernel_w},
            {stride_h, stride_w},
            {padding_h, padding_w}
        );

        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

// AvgPool2d operation
ts_TensorHandle ts_tensor_avg_pool2d(
    ts_TensorHandle input,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_h,
    int64_t padding_w,
    ts_Error* error
) {
    try {
        if (!input) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = torch::avg_pool2d(
            input->tensor,
            {kernel_h, kernel_w},
            {stride_h, stride_w},
            {padding_h, padding_w}
        );

        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

// Dropout operation
ts_TensorHandle ts_tensor_dropout(
    ts_TensorHandle input,
    double p,
    int training,
    ts_Error* error
) {
    try {
        if (!input) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = torch::dropout(input->tensor, p, training != 0);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

// BatchNorm2d operation
ts_TensorHandle ts_tensor_batch_norm(
    ts_TensorHandle input,
    ts_TensorHandle weight,
    ts_TensorHandle bias,
    ts_TensorHandle running_mean,
    ts_TensorHandle running_var,
    int training,
    double momentum,
    double eps,
    ts_Error* error
) {
    try {
        if (!input) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        torch::Tensor weight_t = weight ? weight->tensor : torch::Tensor();
        torch::Tensor bias_t = bias ? bias->tensor : torch::Tensor();
        torch::Tensor running_mean_t = running_mean ? running_mean->tensor : torch::Tensor();
        torch::Tensor running_var_t = running_var ? running_var->tensor : torch::Tensor();

        auto result = torch::batch_norm(
            input->tensor,
            weight_t,
            bias_t,
            running_mean_t,
            running_var_t,
            training != 0,
            momentum,
            eps,
            true  // cudnn_enabled
        );

        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

// LayerNorm operation
ts_TensorHandle ts_tensor_layer_norm(
    ts_TensorHandle input,
    const int64_t* normalized_shape,
    size_t normalized_shape_len,
    ts_TensorHandle weight,
    ts_TensorHandle bias,
    double eps,
    ts_Error* error
) {
    try {
        if (!input) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        std::vector<int64_t> norm_shape(normalized_shape, normalized_shape + normalized_shape_len);
        torch::Tensor weight_t = weight ? weight->tensor : torch::Tensor();
        torch::Tensor bias_t = bias ? bias->tensor : torch::Tensor();

        auto result = torch::layer_norm(
            input->tensor,
            norm_shape,
            weight_t,
            bias_t,
            eps
        );

        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

// Index select operation
ts_TensorHandle ts_tensor_index_select(
    ts_TensorHandle tensor,
    int64_t dim,
    ts_TensorHandle index,
    ts_Error* error
) {
    try {
        if (!tensor || !index) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = torch::index_select(tensor->tensor, dim, index->tensor);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

// Argmax operation
ts_TensorHandle ts_tensor_argmax(
    ts_TensorHandle tensor,
    int64_t dim,
    int keepdim,
    ts_Error* error
) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = tensor->tensor.argmax(dim, keepdim != 0);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

// Narrow operation - returns a view (zero-copy slice)
ts_TensorHandle ts_tensor_narrow(
    ts_TensorHandle tensor,
    int64_t dim,
    int64_t start,
    int64_t length,
    ts_Error* error
) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        // narrow returns a view - no data copy!
        auto result = tensor->tensor.narrow(dim, start, length);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

// Element-wise minimum of two tensors
ts_TensorHandle ts_tensor_minimum(
    ts_TensorHandle a,
    ts_TensorHandle b,
    ts_Error* error
) {
    try {
        if (!a || !b) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = torch::minimum(a->tensor, b->tensor);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

// Element-wise maximum of two tensors
ts_TensorHandle ts_tensor_maximum(
    ts_TensorHandle a,
    ts_TensorHandle b,
    ts_Error* error
) {
    try {
        if (!a || !b) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = torch::maximum(a->tensor, b->tensor);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

// Clamp tensor values to range [min, max]
ts_TensorHandle ts_tensor_clamp(
    ts_TensorHandle tensor,
    double min_val,
    double max_val,
    ts_Error* error
) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = torch::clamp(tensor->tensor, min_val, max_val);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

// Clamp tensor values to minimum
ts_TensorHandle ts_tensor_clamp_min(
    ts_TensorHandle tensor,
    double min_val,
    ts_Error* error
) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = torch::clamp_min(tensor->tensor, min_val);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

// Clamp tensor values to maximum
ts_TensorHandle ts_tensor_clamp_max(
    ts_TensorHandle tensor,
    double max_val,
    ts_Error* error
) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = torch::clamp_max(tensor->tensor, max_val);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}
