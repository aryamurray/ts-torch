#include "ts_torch/internal.h"

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
