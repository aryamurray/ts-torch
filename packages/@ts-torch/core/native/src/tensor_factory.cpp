#include "ts_torch/internal.h"

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

// ============================================================================
// Int32 shape overloads (fast path - avoids BigInt conversion overhead)
// ============================================================================

// Maximum number of dimensions we support for validation
static constexpr size_t MAX_TENSOR_NDIM = 16;

// Helper to validate and convert int32 shape to int64 vector
static inline bool validate_and_convert_shape_i32(
    const int32_t* shape,
    size_t ndim,
    std::vector<int64_t>& out_shape,
    ts_Error* error
) {
    if (ndim > MAX_TENSOR_NDIM) {
        set_error(error, 1, "Too many dimensions (max 16)");
        return false;
    }
    out_shape.resize(ndim);
    for (size_t i = 0; i < ndim; i++) {
        if (shape[i] < 0) {
            set_error(error, 1, "Negative dimension not allowed");
            return false;
        }
        out_shape[i] = static_cast<int64_t>(shape[i]);
    }
    return true;
}

ts_TensorHandle ts_tensor_zeros_i32(
    const int32_t* shape,
    size_t ndim,
    ts_DType dtype,
    ts_DeviceType device,
    int device_index,
    ts_Error* error
) {
    try {
        std::vector<int64_t> shape_vec;
        if (!validate_and_convert_shape_i32(shape, ndim, shape_vec, error)) {
            return nullptr;
        }
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

ts_TensorHandle ts_tensor_ones_i32(
    const int32_t* shape,
    size_t ndim,
    ts_DType dtype,
    ts_DeviceType device,
    int device_index,
    ts_Error* error
) {
    try {
        std::vector<int64_t> shape_vec;
        if (!validate_and_convert_shape_i32(shape, ndim, shape_vec, error)) {
            return nullptr;
        }
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

ts_TensorHandle ts_tensor_randn_i32(
    const int32_t* shape,
    size_t ndim,
    ts_DType dtype,
    ts_DeviceType device,
    int device_index,
    ts_Error* error
) {
    try {
        std::vector<int64_t> shape_vec;
        if (!validate_and_convert_shape_i32(shape, ndim, shape_vec, error)) {
            return nullptr;
        }
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

ts_TensorHandle ts_tensor_rand_i32(
    const int32_t* shape,
    size_t ndim,
    ts_DType dtype,
    ts_DeviceType device,
    int device_index,
    ts_Error* error
) {
    try {
        std::vector<int64_t> shape_vec;
        if (!validate_and_convert_shape_i32(shape, ndim, shape_vec, error)) {
            return nullptr;
        }
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

ts_TensorHandle ts_tensor_empty_i32(
    const int32_t* shape,
    size_t ndim,
    ts_DType dtype,
    ts_DeviceType device,
    int device_index,
    ts_Error* error
) {
    try {
        std::vector<int64_t> shape_vec;
        if (!validate_and_convert_shape_i32(shape, ndim, shape_vec, error)) {
            return nullptr;
        }
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

// ============================================================================
// Buffer-based tensor creation
// ============================================================================

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
        auto target_device = make_device(device, device_index);

        // Calculate number of elements
        size_t numel = 1;
        for (auto dim : shape_vec) numel *= dim;

        auto options = torch::TensorOptions()
            .dtype(scalar_type)
            .device(target_device);

        // Create tensor directly on target device
        torch::Tensor tensor = torch::empty(shape_vec, options);

        if (device == TS_DEVICE_CPU) {
            // Direct memcpy for CPU tensors - avoids from_blob + clone overhead
            std::memcpy(tensor.data_ptr(), data, numel * tensor.element_size());
        } else {
            // For GPU: create a view of the source data, then copy to device tensor
            auto cpu_view = torch::from_blob(
                const_cast<void*>(data),
                shape_vec,
                torch::TensorOptions().dtype(scalar_type)
            );
            tensor.copy_(cpu_view);
        }

        auto* handle = new ts_Tensor(std::move(tensor));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}
