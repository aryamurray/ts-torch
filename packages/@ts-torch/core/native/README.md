# ts-torch Native Library

C shim layer that wraps the LibTorch C++ API for FFI (Foreign Function Interface) access from TypeScript/JavaScript.

## Overview

This library provides a C API wrapper around LibTorch, enabling:
- Zero-copy tensor operations via FFI
- Automatic memory management through scope-based RAII
- Full gradient computation support (autograd)
- Multi-device support (CPU, CUDA, MPS)
- Type-safe error handling

## Architecture

### Key Components

1. **Opaque Handles**: Type-safe pointers to C++ objects
   - `ts_TensorHandle`: Wraps `torch::Tensor`
   - `ts_ModuleHandle`: Wraps `torch::nn::Module`
   - `ts_OptimizerHandle`: Wraps `torch::optim::Optimizer`
   - `ts_ScopeHandle`: Manages automatic cleanup

2. **Error Handling**: Thread-safe error propagation
   - `ts_Error` struct with error code and message
   - All functions accept optional error parameter
   - No exceptions cross FFI boundary

3. **Memory Management**:
   - Scope-based automatic cleanup (`ts_scope_begin`/`ts_scope_end`)
   - Manual cleanup with `ts_tensor_delete`
   - Escape mechanism for returning values from scopes

4. **Data Types**: Comprehensive dtype support
   - Float: FLOAT32, FLOAT64, FLOAT16, BFLOAT16
   - Integer: INT8, INT16, INT32, INT64, UINT8
   - Boolean: BOOL

5. **Devices**: Multi-backend support
   - CPU (always available)
   - CUDA (NVIDIA GPUs)
   - MPS (Apple Silicon)

## Building

### Prerequisites

- CMake 3.18 or higher
- C++17 compatible compiler
  - GCC 7+ or Clang 5+ (Linux/macOS)
  - MSVC 2019+ (Windows)
- LibTorch (download from https://pytorch.org/)

### Download LibTorch

#### Linux/macOS
```bash
# CPU version
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-latest.zip
unzip libtorch-cxx11-abi-shared-with-deps-latest.zip

# CUDA version (example for CUDA 12.1)
wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-latest.zip
unzip libtorch-cxx11-abi-shared-with-deps-latest.zip
```

#### Windows
```powershell
# Download from https://pytorch.org/get-started/locally/
# Select: Windows, LibTorch, C++/Java, CPU or CUDA
# Extract the zip file
```

### Build Instructions

#### Linux/macOS
```bash
cd packages/@ts-torch/core/native
mkdir build
cd build

# Configure with LibTorch path
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch -DCMAKE_BUILD_TYPE=Release ..

# Build
cmake --build . -j$(nproc)

# Install (optional)
sudo cmake --install .
```

#### Windows (Visual Studio)
```powershell
cd packages\@ts-torch\core\native
mkdir build
cd build

# Configure (adjust paths as needed)
cmake -DCMAKE_PREFIX_PATH=C:\path\to\libtorch -DCMAKE_BUILD_TYPE=Release ..

# Build
cmake --build . --config Release -j

# The DLL and LIB files will be in build\Release\
```

#### Windows (MinGW)
```bash
cd packages/@ts-torch/core/native
mkdir build
cd build

cmake -G "MinGW Makefiles" -DCMAKE_PREFIX_PATH=C:/path/to/libtorch -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j
```

### Build Options

- `CMAKE_PREFIX_PATH`: Path to LibTorch installation (required)
- `CMAKE_BUILD_TYPE`: Build type (Debug, Release, RelWithDebInfo)
- `CMAKE_INSTALL_PREFIX`: Installation prefix (default: /usr/local)

Example with custom install prefix:
```bash
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch \
      -DCMAKE_INSTALL_PREFIX=/opt/ts-torch \
      -DCMAKE_BUILD_TYPE=Release ..
```

## API Usage Examples

### Basic Tensor Creation

```c
#include <ts_torch.h>
#include <stdio.h>

int main() {
    ts_Error error = {0};

    // Create a 2x3 tensor of zeros
    int64_t shape[] = {2, 3};
    ts_TensorHandle tensor = ts_tensor_zeros(
        shape, 2,
        TS_DTYPE_FLOAT32,
        TS_DEVICE_CPU, 0,
        &error
    );

    if (ts_error_occurred(&error)) {
        printf("Error: %s\n", error.message);
        return 1;
    }

    // Check properties
    int64_t ndim = ts_tensor_ndim(tensor, &error);
    int64_t numel = ts_tensor_numel(tensor, &error);
    printf("Shape: [%ld, %ld], Elements: %ld\n", shape[0], shape[1], numel);

    // Cleanup
    ts_tensor_delete(tensor);
    return 0;
}
```

### Scope-Based Memory Management

```c
#include <ts_torch.h>

ts_TensorHandle compute_sum(void) {
    ts_Error error = {0};
    ts_ScopeHandle scope = ts_scope_begin();

    // All tensors created in this scope are automatically tracked
    int64_t shape[] = {10};
    ts_TensorHandle a = ts_tensor_ones(shape, 1, TS_DTYPE_FLOAT32, TS_DEVICE_CPU, 0, &error);
    ts_TensorHandle b = ts_tensor_ones(shape, 1, TS_DTYPE_FLOAT32, TS_DEVICE_CPU, 0, &error);

    // Compute result
    ts_TensorHandle result = ts_tensor_add(a, b, &error);

    // Escape result from scope (won't be deleted)
    ts_scope_escape_tensor(scope, result);

    // End scope - deletes a and b, but not result
    ts_scope_end(scope);

    return result;
}
```

### Autograd Example

```c
#include <ts_torch.h>

void gradient_example(void) {
    ts_Error error = {0};
    int64_t shape[] = {2, 2};

    // Create tensor with gradient tracking
    ts_TensorHandle x = ts_tensor_ones(shape, 2, TS_DTYPE_FLOAT32, TS_DEVICE_CPU, 0, &error);
    ts_tensor_set_requires_grad(x, 1, &error);

    // Forward pass: y = x^2 * 3 (element-wise)
    ts_TensorHandle x_squared = ts_tensor_mul(x, x, &error);

    int64_t three_shape[] = {1};
    ts_TensorHandle three = ts_tensor_ones(three_shape, 1, TS_DTYPE_FLOAT32, TS_DEVICE_CPU, 0, &error);
    // (Would need scalar multiplication, simplified here)

    ts_TensorHandle y = ts_tensor_mul(x_squared, three, &error);

    // Backward pass
    ts_tensor_backward(y, &error);

    // Get gradient
    ts_TensorHandle grad = ts_tensor_grad(x, &error);

    // Cleanup
    ts_tensor_delete(grad);
    ts_tensor_delete(y);
    ts_tensor_delete(three);
    ts_tensor_delete(x_squared);
    ts_tensor_delete(x);
}
```

### CUDA Example

```c
#include <ts_torch.h>
#include <stdio.h>

void cuda_example(void) {
    if (!ts_cuda_is_available()) {
        printf("CUDA not available\n");
        return;
    }

    printf("CUDA devices: %d\n", ts_cuda_device_count());

    ts_Error error = {0};
    int64_t shape[] = {1000, 1000};

    // Create tensors on GPU
    ts_TensorHandle a = ts_tensor_randn(shape, 2, TS_DTYPE_FLOAT32, TS_DEVICE_CUDA, 0, &error);
    ts_TensorHandle b = ts_tensor_randn(shape, 2, TS_DTYPE_FLOAT32, TS_DEVICE_CUDA, 0, &error);

    // Matrix multiplication on GPU
    ts_TensorHandle c = ts_tensor_matmul(a, b, &error);

    // Move result back to CPU
    ts_TensorHandle c_cpu = ts_tensor_cpu(c, &error);

    // Cleanup
    ts_tensor_delete(c_cpu);
    ts_tensor_delete(c);
    ts_tensor_delete(b);
    ts_tensor_delete(a);
}
```

## API Reference

### Tensor Creation

- `ts_tensor_zeros` - Create tensor filled with zeros
- `ts_tensor_ones` - Create tensor filled with ones
- `ts_tensor_randn` - Create tensor with random normal values
- `ts_tensor_empty` - Create uninitialized tensor
- `ts_tensor_from_buffer` - Create tensor from existing memory

### Tensor Properties

- `ts_tensor_ndim` - Get number of dimensions
- `ts_tensor_size` - Get size of specific dimension
- `ts_tensor_shape` - Get full shape array
- `ts_tensor_dtype` - Get data type
- `ts_tensor_numel` - Get total number of elements
- `ts_tensor_device_type` - Get device type (CPU/CUDA/MPS)
- `ts_tensor_device_index` - Get device index

### Tensor Operations

- `ts_tensor_add`, `ts_tensor_sub`, `ts_tensor_mul`, `ts_tensor_div` - Element-wise arithmetic
- `ts_tensor_matmul` - Matrix multiplication
- `ts_tensor_transpose` - Transpose two dimensions
- `ts_tensor_reshape` - Change shape
- `ts_tensor_sum`, `ts_tensor_mean` - Reduction operations

### Activation Functions

- `ts_tensor_relu` - Rectified Linear Unit
- `ts_tensor_sigmoid` - Sigmoid activation
- `ts_tensor_softmax` - Softmax along dimension
- `ts_tensor_tanh` - Hyperbolic tangent

### Autograd

- `ts_tensor_backward` - Compute gradients
- `ts_tensor_grad` - Get gradient tensor
- `ts_tensor_set_requires_grad` - Enable/disable gradient tracking
- `ts_tensor_requires_grad` - Check if gradient tracking is enabled
- `ts_tensor_is_leaf` - Check if tensor is a leaf node

### Memory Management

- `ts_tensor_delete` - Manually delete tensor
- `ts_tensor_clone` - Create deep copy
- `ts_tensor_detach` - Detach from computation graph
- `ts_scope_begin` - Begin automatic cleanup scope
- `ts_scope_end` - End scope and cleanup
- `ts_scope_escape_tensor` - Prevent tensor from being deleted

### Device Operations

- `ts_cuda_is_available` - Check CUDA availability
- `ts_cuda_device_count` - Get number of CUDA devices
- `ts_tensor_to_device` - Move tensor to device
- `ts_tensor_cpu` - Move tensor to CPU
- `ts_tensor_cuda` - Move tensor to CUDA

## Thread Safety

- Each thread has its own scope stack (thread-local storage)
- LibTorch operations are generally thread-safe
- Error handling is thread-safe (per-call error parameter)
- Concurrent access to same tensor requires external synchronization

## Performance Considerations

1. **Memory Layout**: Tensors are row-major (C-contiguous)
2. **Zero-Copy**: Use `ts_tensor_data_ptr` for direct memory access
3. **Async Operations**: CUDA operations are asynchronous by default
4. **Scope Overhead**: Minimal, use scopes for convenience

## Error Codes

- `0` - No error
- `1` - Generic error (check message for details)

Always check `ts_error_occurred(&error)` after operations that can fail.

## License

MIT License - See LICENSE file for details

## Related Projects

- [LibTorch](https://pytorch.org/cppdocs/) - PyTorch C++ API
- [ts-torch](https://github.com/yourusername/ts-torch) - TypeScript bindings

## Contributing

See main repository for contribution guidelines.
