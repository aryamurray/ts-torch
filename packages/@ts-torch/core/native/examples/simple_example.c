/*
 * Simple example demonstrating ts-torch C API usage
 *
 * This example shows:
 * - Tensor creation
 * - Basic operations
 * - Scope-based memory management
 * - Error handling
 */

#include <ts_torch.h>
#include <stdio.h>
#include <stdlib.h>

void print_tensor_info(ts_TensorHandle tensor, const char* name) {
    ts_Error error = {0};

    int64_t ndim = ts_tensor_ndim(tensor, &error);
    int64_t numel = ts_tensor_numel(tensor, &error);
    ts_DType dtype = ts_tensor_dtype(tensor, &error);

    printf("%s: ndim=%lld, numel=%lld, dtype=%d\n", name, ndim, numel, dtype);

    // Get and print shape
    int64_t shape[8];
    size_t actual_ndim;
    ts_tensor_shape(tensor, shape, &actual_ndim, &error);

    printf("  Shape: [");
    for (size_t i = 0; i < actual_ndim; i++) {
        printf("%lld%s", shape[i], i < actual_ndim - 1 ? ", " : "");
    }
    printf("]\n");
}

void example_basic_operations(void) {
    printf("\n=== Basic Operations Example ===\n");

    ts_Error error = {0};
    ts_ScopeHandle scope = ts_scope_begin();

    // Create two tensors
    int64_t shape[] = {2, 3};
    ts_TensorHandle a = ts_tensor_ones(
        shape, 2,
        TS_DTYPE_FLOAT32,
        TS_DEVICE_CPU, 0,
        &error
    );

    ts_TensorHandle b = ts_tensor_ones(
        shape, 2,
        TS_DTYPE_FLOAT32,
        TS_DEVICE_CPU, 0,
        &error
    );

    if (ts_error_occurred(&error)) {
        printf("Error creating tensors: %s\n", error.message);
        ts_scope_end(scope);
        return;
    }

    print_tensor_info(a, "Tensor a");
    print_tensor_info(b, "Tensor b");

    // Perform operations
    ts_TensorHandle sum = ts_tensor_add(a, b, &error);
    ts_TensorHandle product = ts_tensor_mul(a, b, &error);

    if (ts_error_occurred(&error)) {
        printf("Error in operations: %s\n", error.message);
        ts_scope_end(scope);
        return;
    }

    print_tensor_info(sum, "Sum (a + b)");
    print_tensor_info(product, "Product (a * b)");

    // Scope automatically cleans up all tensors
    ts_scope_end(scope);
    printf("Tensors cleaned up automatically\n");
}

void example_matrix_multiplication(void) {
    printf("\n=== Matrix Multiplication Example ===\n");

    ts_Error error = {0};
    ts_ScopeHandle scope = ts_scope_begin();

    // Create matrices
    int64_t shape_a[] = {3, 4};
    int64_t shape_b[] = {4, 2};

    ts_TensorHandle a = ts_tensor_randn(
        shape_a, 2,
        TS_DTYPE_FLOAT32,
        TS_DEVICE_CPU, 0,
        &error
    );

    ts_TensorHandle b = ts_tensor_randn(
        shape_b, 2,
        TS_DTYPE_FLOAT32,
        TS_DEVICE_CPU, 0,
        &error
    );

    if (ts_error_occurred(&error)) {
        printf("Error creating matrices: %s\n", error.message);
        ts_scope_end(scope);
        return;
    }

    print_tensor_info(a, "Matrix A");
    print_tensor_info(b, "Matrix B");

    // Matrix multiplication: (3x4) @ (4x2) = (3x2)
    ts_TensorHandle c = ts_tensor_matmul(a, b, &error);

    if (ts_error_occurred(&error)) {
        printf("Error in matmul: %s\n", error.message);
        ts_scope_end(scope);
        return;
    }

    print_tensor_info(c, "Result C = A @ B");

    ts_scope_end(scope);
}

void example_activations(void) {
    printf("\n=== Activation Functions Example ===\n");

    ts_Error error = {0};
    ts_ScopeHandle scope = ts_scope_begin();

    // Create a tensor with values in range [-2, 2]
    int64_t shape[] = {2, 4};
    ts_TensorHandle x = ts_tensor_randn(
        shape, 2,
        TS_DTYPE_FLOAT32,
        TS_DEVICE_CPU, 0,
        &error
    );

    if (ts_error_occurred(&error)) {
        printf("Error creating tensor: %s\n", error.message);
        ts_scope_end(scope);
        return;
    }

    print_tensor_info(x, "Input");

    // Apply various activation functions
    ts_TensorHandle relu_out = ts_tensor_relu(x, &error);
    ts_TensorHandle sigmoid_out = ts_tensor_sigmoid(x, &error);
    ts_TensorHandle tanh_out = ts_tensor_tanh(x, &error);

    if (ts_error_occurred(&error)) {
        printf("Error in activations: %s\n", error.message);
        ts_scope_end(scope);
        return;
    }

    print_tensor_info(relu_out, "ReLU(x)");
    print_tensor_info(sigmoid_out, "Sigmoid(x)");
    print_tensor_info(tanh_out, "Tanh(x)");

    // Softmax along last dimension
    ts_TensorHandle softmax_out = ts_tensor_softmax(x, 1, &error);

    if (ts_error_occurred(&error)) {
        printf("Error in softmax: %s\n", error.message);
        ts_scope_end(scope);
        return;
    }

    print_tensor_info(softmax_out, "Softmax(x, dim=1)");

    ts_scope_end(scope);
}

void example_reduction_operations(void) {
    printf("\n=== Reduction Operations Example ===\n");

    ts_Error error = {0};
    ts_ScopeHandle scope = ts_scope_begin();

    // Create a 3x4 tensor
    int64_t shape[] = {3, 4};
    ts_TensorHandle x = ts_tensor_ones(
        shape, 2,
        TS_DTYPE_FLOAT32,
        TS_DEVICE_CPU, 0,
        &error
    );

    if (ts_error_occurred(&error)) {
        printf("Error creating tensor: %s\n", error.message);
        ts_scope_end(scope);
        return;
    }

    print_tensor_info(x, "Input");

    // Total sum
    ts_TensorHandle total_sum = ts_tensor_sum(x, &error);
    print_tensor_info(total_sum, "Sum (all elements)");

    // Sum along dimension 0
    ts_TensorHandle sum_dim0 = ts_tensor_sum_dim(x, 0, 0, &error);
    print_tensor_info(sum_dim0, "Sum along dim 0");

    // Sum along dimension 1, keeping dimension
    ts_TensorHandle sum_dim1_keep = ts_tensor_sum_dim(x, 1, 1, &error);
    print_tensor_info(sum_dim1_keep, "Sum along dim 1 (keepdim)");

    // Mean
    ts_TensorHandle mean = ts_tensor_mean(x, &error);
    print_tensor_info(mean, "Mean");

    if (ts_error_occurred(&error)) {
        printf("Error in reductions: %s\n", error.message);
    }

    ts_scope_end(scope);
}

void example_device_query(void) {
    printf("\n=== Device Information ===\n");

    printf("ts-torch version: %s\n", ts_version());

    int cuda_available = ts_cuda_is_available();
    printf("CUDA available: %s\n", cuda_available ? "Yes" : "No");

    if (cuda_available) {
        int device_count = ts_cuda_device_count();
        printf("CUDA device count: %d\n", device_count);
    }
}

void example_autograd(void) {
    printf("\n=== Autograd Example ===\n");

    ts_Error error = {0};
    ts_ScopeHandle scope = ts_scope_begin();

    // Create input tensor with gradient tracking
    int64_t shape[] = {2, 2};
    ts_TensorHandle x = ts_tensor_ones(
        shape, 2,
        TS_DTYPE_FLOAT32,
        TS_DEVICE_CPU, 0,
        &error
    );

    ts_tensor_set_requires_grad(x, 1, &error);

    if (ts_error_occurred(&error)) {
        printf("Error setting requires_grad: %s\n", error.message);
        ts_scope_end(scope);
        return;
    }

    printf("Created tensor with requires_grad=true\n");
    print_tensor_info(x, "x");

    // Forward pass: y = x * 2
    int64_t scalar_shape[] = {1};
    ts_TensorHandle two = ts_tensor_ones(scalar_shape, 1, TS_DTYPE_FLOAT32, TS_DEVICE_CPU, 0, &error);

    // In a real implementation, we'd need scalar broadcasting
    // For now, this demonstrates the API structure
    ts_TensorHandle y = ts_tensor_mul(x, x, &error);  // y = x^2
    ts_TensorHandle z = ts_tensor_sum(y, &error);      // z = sum(x^2)

    print_tensor_info(z, "z = sum(x^2)");

    // Backward pass
    printf("Computing gradients...\n");
    ts_tensor_backward(z, &error);

    if (ts_error_occurred(&error)) {
        printf("Error in backward: %s\n", error.message);
        ts_scope_end(scope);
        return;
    }

    // Get gradient
    ts_TensorHandle grad = ts_tensor_grad(x, &error);

    if (ts_error_occurred(&error)) {
        printf("Error getting gradient: %s\n", error.message);
        ts_scope_end(scope);
        return;
    }

    if (grad) {
        print_tensor_info(grad, "gradient of z w.r.t. x");
    } else {
        printf("No gradient computed\n");
    }

    ts_scope_end(scope);
}

int main(void) {
    printf("===================================\n");
    printf("   ts-torch C API Examples\n");
    printf("===================================\n");

    example_device_query();
    example_basic_operations();
    example_matrix_multiplication();
    example_activations();
    example_reduction_operations();
    example_autograd();

    printf("\n===================================\n");
    printf("   All examples completed!\n");
    printf("===================================\n");

    return 0;
}
