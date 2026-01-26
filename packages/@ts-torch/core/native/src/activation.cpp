#include "ts_torch/internal.h"

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
