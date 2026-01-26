#ifndef TS_TORCH_INTERNAL_H
#define TS_TORCH_INTERNAL_H

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

// Thread-local scope stack declaration (defined in common.cpp)
extern thread_local std::vector<std::unique_ptr<ts_Scope>> g_scope_stack;

// Helper function to convert ts_DType to torch::ScalarType
inline torch::ScalarType dtype_to_scalar_type(ts_DType dtype) {
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
inline ts_DType scalar_type_to_dtype(torch::ScalarType scalar_type) {
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
inline torch::Device make_device(ts_DeviceType device_type, int device_index) {
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
inline ts_DeviceType device_to_device_type(const torch::Device& device) {
    switch (device.type()) {
        case torch::kCPU: return TS_DEVICE_CPU;
        case torch::kCUDA: return TS_DEVICE_CUDA;
        case torch::kMPS: return TS_DEVICE_MPS;
        default: return TS_DEVICE_CPU;
    }
}

// Helper function to set error
inline void set_error(ts_Error* error, int code, const char* message) {
    if (error) {
        error->code = code;
        std::strncpy(error->message, message, sizeof(error->message) - 1);
        error->message[sizeof(error->message) - 1] = '\0';
    }
}

// Helper function to register tensor in current scope
inline void register_in_scope(ts_TensorHandle handle) {
    if (!g_scope_stack.empty()) {
        g_scope_stack.back()->tensors.insert(handle);
    }
}

#endif // TS_TORCH_INTERNAL_H
