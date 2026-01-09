/*
 * Basic unit tests for ts-torch C API
 *
 * This is a simple test suite without external dependencies.
 * For production, consider using a proper C testing framework like Unity or Check.
 */

#include <ts_torch.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Test counter
static int tests_passed = 0;
static int tests_failed = 0;

// Test macros
#define TEST(name) \
    static void test_##name(void); \
    static void run_test_##name(void) { \
        printf("Running test: %s ... ", #name); \
        test_##name(); \
        printf("PASSED\n"); \
        tests_passed++; \
    } \
    static void test_##name(void)

#define ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            printf("FAILED\n"); \
            printf("  Assertion failed: %s\n", message); \
            printf("  at %s:%d\n", __FILE__, __LINE__); \
            tests_failed++; \
            return; \
        } \
    } while(0)

#define ASSERT_EQ(a, b, message) \
    ASSERT((a) == (b), message)

#define ASSERT_NE(a, b, message) \
    ASSERT((a) != (b), message)

#define ASSERT_NULL(ptr, message) \
    ASSERT((ptr) == NULL, message)

#define ASSERT_NOT_NULL(ptr, message) \
    ASSERT((ptr) != NULL, message)

#define ASSERT_NO_ERROR(error) \
    ASSERT(!ts_error_occurred(&error), error.message)

// Test cases

TEST(version) {
    const char* version = ts_version();
    ASSERT_NOT_NULL(version, "Version should not be NULL");
    ASSERT(strlen(version) > 0, "Version should not be empty");
}

TEST(error_handling) {
    ts_Error error = {0};

    // Initially no error
    ASSERT_EQ(ts_error_occurred(&error), 0, "Should have no error initially");

    // Clear error
    ts_error_clear(&error);
    ASSERT_EQ(error.code, 0, "Error code should be 0 after clear");
    ASSERT_EQ(error.message[0], '\0', "Error message should be empty after clear");
}

TEST(tensor_zeros) {
    ts_Error error = {0};
    int64_t shape[] = {2, 3};

    ts_TensorHandle tensor = ts_tensor_zeros(
        shape, 2,
        TS_DTYPE_FLOAT32,
        TS_DEVICE_CPU, 0,
        &error
    );

    ASSERT_NO_ERROR(error);
    ASSERT_NOT_NULL(tensor, "Tensor should not be NULL");

    // Check properties
    int64_t ndim = ts_tensor_ndim(tensor, &error);
    ASSERT_NO_ERROR(error);
    ASSERT_EQ(ndim, 2, "Tensor should have 2 dimensions");

    int64_t numel = ts_tensor_numel(tensor, &error);
    ASSERT_NO_ERROR(error);
    ASSERT_EQ(numel, 6, "Tensor should have 6 elements");

    ts_DType dtype = ts_tensor_dtype(tensor, &error);
    ASSERT_NO_ERROR(error);
    ASSERT_EQ(dtype, TS_DTYPE_FLOAT32, "Tensor should be FLOAT32");

    ts_tensor_delete(tensor);
}

TEST(tensor_ones) {
    ts_Error error = {0};
    int64_t shape[] = {5};

    ts_TensorHandle tensor = ts_tensor_ones(
        shape, 1,
        TS_DTYPE_FLOAT32,
        TS_DEVICE_CPU, 0,
        &error
    );

    ASSERT_NO_ERROR(error);
    ASSERT_NOT_NULL(tensor, "Tensor should not be NULL");

    int64_t numel = ts_tensor_numel(tensor, &error);
    ASSERT_EQ(numel, 5, "Tensor should have 5 elements");

    ts_tensor_delete(tensor);
}

TEST(tensor_shape) {
    ts_Error error = {0};
    int64_t shape[] = {2, 3, 4};

    ts_TensorHandle tensor = ts_tensor_zeros(
        shape, 3,
        TS_DTYPE_FLOAT32,
        TS_DEVICE_CPU, 0,
        &error
    );

    ASSERT_NO_ERROR(error);

    // Get shape back
    int64_t result_shape[8];
    size_t result_ndim;
    ts_tensor_shape(tensor, result_shape, &result_ndim, &error);

    ASSERT_NO_ERROR(error);
    ASSERT_EQ(result_ndim, 3, "Should have 3 dimensions");
    ASSERT_EQ(result_shape[0], 2, "First dimension should be 2");
    ASSERT_EQ(result_shape[1], 3, "Second dimension should be 3");
    ASSERT_EQ(result_shape[2], 4, "Third dimension should be 4");

    ts_tensor_delete(tensor);
}

TEST(tensor_add) {
    ts_Error error = {0};
    int64_t shape[] = {2, 2};

    ts_TensorHandle a = ts_tensor_ones(shape, 2, TS_DTYPE_FLOAT32, TS_DEVICE_CPU, 0, &error);
    ASSERT_NO_ERROR(error);

    ts_TensorHandle b = ts_tensor_ones(shape, 2, TS_DTYPE_FLOAT32, TS_DEVICE_CPU, 0, &error);
    ASSERT_NO_ERROR(error);

    ts_TensorHandle c = ts_tensor_add(a, b, &error);
    ASSERT_NO_ERROR(error);
    ASSERT_NOT_NULL(c, "Result should not be NULL");

    int64_t numel = ts_tensor_numel(c, &error);
    ASSERT_EQ(numel, 4, "Result should have 4 elements");

    ts_tensor_delete(c);
    ts_tensor_delete(b);
    ts_tensor_delete(a);
}

TEST(tensor_mul) {
    ts_Error error = {0};
    int64_t shape[] = {3, 3};

    ts_TensorHandle a = ts_tensor_ones(shape, 2, TS_DTYPE_FLOAT32, TS_DEVICE_CPU, 0, &error);
    ts_TensorHandle b = ts_tensor_ones(shape, 2, TS_DTYPE_FLOAT32, TS_DEVICE_CPU, 0, &error);
    ts_TensorHandle c = ts_tensor_mul(a, b, &error);

    ASSERT_NO_ERROR(error);
    ASSERT_NOT_NULL(c, "Result should not be NULL");

    ts_tensor_delete(c);
    ts_tensor_delete(b);
    ts_tensor_delete(a);
}

TEST(tensor_matmul) {
    ts_Error error = {0};
    int64_t shape_a[] = {2, 3};
    int64_t shape_b[] = {3, 4};

    ts_TensorHandle a = ts_tensor_ones(shape_a, 2, TS_DTYPE_FLOAT32, TS_DEVICE_CPU, 0, &error);
    ts_TensorHandle b = ts_tensor_ones(shape_b, 2, TS_DTYPE_FLOAT32, TS_DEVICE_CPU, 0, &error);
    ts_TensorHandle c = ts_tensor_matmul(a, b, &error);

    ASSERT_NO_ERROR(error);
    ASSERT_NOT_NULL(c, "Result should not be NULL");

    // Result should be 2x4
    int64_t result_shape[8];
    size_t result_ndim;
    ts_tensor_shape(c, result_shape, &result_ndim, &error);

    ASSERT_EQ(result_ndim, 2, "Result should have 2 dimensions");
    ASSERT_EQ(result_shape[0], 2, "First dimension should be 2");
    ASSERT_EQ(result_shape[1], 4, "Second dimension should be 4");

    ts_tensor_delete(c);
    ts_tensor_delete(b);
    ts_tensor_delete(a);
}

TEST(tensor_transpose) {
    ts_Error error = {0};
    int64_t shape[] = {2, 3};

    ts_TensorHandle a = ts_tensor_zeros(shape, 2, TS_DTYPE_FLOAT32, TS_DEVICE_CPU, 0, &error);
    ts_TensorHandle b = ts_tensor_transpose(a, 0, 1, &error);

    ASSERT_NO_ERROR(error);
    ASSERT_NOT_NULL(b, "Transposed tensor should not be NULL");

    // Check transposed shape
    int64_t result_shape[8];
    size_t result_ndim;
    ts_tensor_shape(b, result_shape, &result_ndim, &error);

    ASSERT_EQ(result_shape[0], 3, "First dimension should be 3");
    ASSERT_EQ(result_shape[1], 2, "Second dimension should be 2");

    ts_tensor_delete(b);
    ts_tensor_delete(a);
}

TEST(tensor_reshape) {
    ts_Error error = {0};
    int64_t shape[] = {2, 3};
    int64_t new_shape[] = {6};

    ts_TensorHandle a = ts_tensor_zeros(shape, 2, TS_DTYPE_FLOAT32, TS_DEVICE_CPU, 0, &error);
    ts_TensorHandle b = ts_tensor_reshape(a, new_shape, 1, &error);

    ASSERT_NO_ERROR(error);
    ASSERT_NOT_NULL(b, "Reshaped tensor should not be NULL");

    int64_t ndim = ts_tensor_ndim(b, &error);
    ASSERT_EQ(ndim, 1, "Reshaped tensor should have 1 dimension");

    int64_t numel = ts_tensor_numel(b, &error);
    ASSERT_EQ(numel, 6, "Reshaped tensor should have 6 elements");

    ts_tensor_delete(b);
    ts_tensor_delete(a);
}

TEST(tensor_sum) {
    ts_Error error = {0};
    int64_t shape[] = {2, 3};

    ts_TensorHandle a = ts_tensor_ones(shape, 2, TS_DTYPE_FLOAT32, TS_DEVICE_CPU, 0, &error);
    ts_TensorHandle sum = ts_tensor_sum(a, &error);

    ASSERT_NO_ERROR(error);
    ASSERT_NOT_NULL(sum, "Sum should not be NULL");

    // Sum of 6 ones should give numel of 1 (scalar)
    int64_t numel = ts_tensor_numel(sum, &error);
    ASSERT_EQ(numel, 1, "Sum result should be a scalar");

    ts_tensor_delete(sum);
    ts_tensor_delete(a);
}

TEST(tensor_relu) {
    ts_Error error = {0};
    int64_t shape[] = {2, 2};

    ts_TensorHandle a = ts_tensor_randn(shape, 2, TS_DTYPE_FLOAT32, TS_DEVICE_CPU, 0, &error);
    ts_TensorHandle b = ts_tensor_relu(a, &error);

    ASSERT_NO_ERROR(error);
    ASSERT_NOT_NULL(b, "ReLU result should not be NULL");

    ts_tensor_delete(b);
    ts_tensor_delete(a);
}

TEST(tensor_sigmoid) {
    ts_Error error = {0};
    int64_t shape[] = {5};

    ts_TensorHandle a = ts_tensor_randn(shape, 1, TS_DTYPE_FLOAT32, TS_DEVICE_CPU, 0, &error);
    ts_TensorHandle b = ts_tensor_sigmoid(a, &error);

    ASSERT_NO_ERROR(error);
    ASSERT_NOT_NULL(b, "Sigmoid result should not be NULL");

    ts_tensor_delete(b);
    ts_tensor_delete(a);
}

TEST(tensor_clone) {
    ts_Error error = {0};
    int64_t shape[] = {2, 3};

    ts_TensorHandle a = ts_tensor_ones(shape, 2, TS_DTYPE_FLOAT32, TS_DEVICE_CPU, 0, &error);
    ts_TensorHandle b = ts_tensor_clone(a, &error);

    ASSERT_NO_ERROR(error);
    ASSERT_NOT_NULL(b, "Clone should not be NULL");
    ASSERT_NE(a, b, "Clone should be a different handle");

    // Check same shape
    int64_t numel_a = ts_tensor_numel(a, &error);
    int64_t numel_b = ts_tensor_numel(b, &error);
    ASSERT_EQ(numel_a, numel_b, "Clone should have same number of elements");

    ts_tensor_delete(b);
    ts_tensor_delete(a);
}

TEST(tensor_detach) {
    ts_Error error = {0};
    int64_t shape[] = {2, 2};

    ts_TensorHandle a = ts_tensor_ones(shape, 2, TS_DTYPE_FLOAT32, TS_DEVICE_CPU, 0, &error);
    ts_tensor_set_requires_grad(a, 1, &error);

    ts_TensorHandle b = ts_tensor_detach(a, &error);

    ASSERT_NO_ERROR(error);
    ASSERT_NOT_NULL(b, "Detached tensor should not be NULL");

    // Detached tensor should not require gradients
    int requires_grad = ts_tensor_requires_grad(b, &error);
    ASSERT_EQ(requires_grad, 0, "Detached tensor should not require gradients");

    ts_tensor_delete(b);
    ts_tensor_delete(a);
}

TEST(tensor_requires_grad) {
    ts_Error error = {0};
    int64_t shape[] = {3, 3};

    ts_TensorHandle a = ts_tensor_ones(shape, 2, TS_DTYPE_FLOAT32, TS_DEVICE_CPU, 0, &error);

    // Initially should not require grad
    int requires_grad = ts_tensor_requires_grad(a, &error);
    ASSERT_NO_ERROR(error);
    ASSERT_EQ(requires_grad, 0, "Tensor should not require grad by default");

    // Set requires grad
    ts_tensor_set_requires_grad(a, 1, &error);
    ASSERT_NO_ERROR(error);

    requires_grad = ts_tensor_requires_grad(a, &error);
    ASSERT_EQ(requires_grad, 1, "Tensor should require grad after setting");

    ts_tensor_delete(a);
}

TEST(scope_management) {
    ts_Error error = {0};
    ts_ScopeHandle scope = ts_scope_begin();

    ASSERT_NOT_NULL(scope, "Scope should not be NULL");

    int64_t shape[] = {2, 2};
    ts_TensorHandle a = ts_tensor_ones(shape, 2, TS_DTYPE_FLOAT32, TS_DEVICE_CPU, 0, &error);
    ASSERT_NO_ERROR(error);

    ts_scope_register_tensor(scope, a);

    // End scope - should clean up tensor
    ts_scope_end(scope);

    // Note: We can't verify cleanup directly, but scope_end shouldn't crash
}

TEST(scope_escape) {
    ts_Error error = {0};
    ts_ScopeHandle scope = ts_scope_begin();

    int64_t shape[] = {2, 2};
    ts_TensorHandle a = ts_tensor_ones(shape, 2, TS_DTYPE_FLOAT32, TS_DEVICE_CPU, 0, &error);

    ts_scope_register_tensor(scope, a);
    ts_TensorHandle escaped = ts_scope_escape_tensor(scope, a);

    ASSERT_EQ(a, escaped, "Escaped tensor should be the same handle");

    ts_scope_end(scope);

    // Tensor should still be valid after scope ends
    int64_t numel = ts_tensor_numel(escaped, &error);
    ASSERT_NO_ERROR(error);
    ASSERT_EQ(numel, 4, "Escaped tensor should still be accessible");

    ts_tensor_delete(escaped);
}

TEST(device_cpu) {
    ts_Error error = {0};
    int64_t shape[] = {5};

    ts_TensorHandle a = ts_tensor_ones(shape, 1, TS_DTYPE_FLOAT32, TS_DEVICE_CPU, 0, &error);
    ASSERT_NO_ERROR(error);

    ts_DeviceType device_type = ts_tensor_device_type(a, &error);
    ASSERT_NO_ERROR(error);
    ASSERT_EQ(device_type, TS_DEVICE_CPU, "Tensor should be on CPU");

    ts_tensor_delete(a);
}

TEST(cuda_availability) {
    int available = ts_cuda_is_available();
    // Just check that the function works, don't assert value
    // since CUDA may or may not be available
    printf("(CUDA available: %s) ", available ? "yes" : "no");

    if (available) {
        int count = ts_cuda_device_count();
        ASSERT(count > 0, "Should have at least one CUDA device if CUDA is available");
    }
}

// Test runner
typedef void (*test_func_t)(void);

typedef struct {
    const char* name;
    test_func_t func;
} test_entry_t;

#define TEST_ENTRY(name) { #name, run_test_##name }

static test_entry_t all_tests[] = {
    TEST_ENTRY(version),
    TEST_ENTRY(error_handling),
    TEST_ENTRY(tensor_zeros),
    TEST_ENTRY(tensor_ones),
    TEST_ENTRY(tensor_shape),
    TEST_ENTRY(tensor_add),
    TEST_ENTRY(tensor_mul),
    TEST_ENTRY(tensor_matmul),
    TEST_ENTRY(tensor_transpose),
    TEST_ENTRY(tensor_reshape),
    TEST_ENTRY(tensor_sum),
    TEST_ENTRY(tensor_relu),
    TEST_ENTRY(tensor_sigmoid),
    TEST_ENTRY(tensor_clone),
    TEST_ENTRY(tensor_detach),
    TEST_ENTRY(tensor_requires_grad),
    TEST_ENTRY(scope_management),
    TEST_ENTRY(scope_escape),
    TEST_ENTRY(device_cpu),
    TEST_ENTRY(cuda_availability),
};

int main(void) {
    printf("===================================\n");
    printf("   ts-torch C API Test Suite\n");
    printf("===================================\n\n");

    printf("Version: %s\n", ts_version());
    printf("CUDA available: %s\n\n", ts_cuda_is_available() ? "Yes" : "No");

    size_t num_tests = sizeof(all_tests) / sizeof(all_tests[0]);

    for (size_t i = 0; i < num_tests; i++) {
        all_tests[i].func();
    }

    printf("\n===================================\n");
    printf("Test Results:\n");
    printf("  Passed: %d\n", tests_passed);
    printf("  Failed: %d\n", tests_failed);
    printf("  Total:  %d\n", tests_passed + tests_failed);
    printf("===================================\n");

    return tests_failed > 0 ? 1 : 0;
}
