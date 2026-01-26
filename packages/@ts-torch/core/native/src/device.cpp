#include "ts_torch/internal.h"

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
