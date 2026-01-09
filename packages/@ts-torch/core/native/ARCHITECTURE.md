# ts-torch Native Library Architecture

## Overview

The ts-torch native library is a C shim layer that wraps LibTorch (PyTorch C++ API) to provide a Foreign Function Interface (FFI) suitable for consumption by TypeScript/JavaScript via Node.js native addons or WebAssembly.

## Design Principles

1. **C ABI Compatibility**: Pure C interface for maximum portability across languages and platforms
2. **Zero-Copy Operations**: Direct memory access where possible to avoid unnecessary copies
3. **RAII Memory Management**: Scope-based automatic cleanup with escape hatches
4. **Thread Safety**: Thread-local storage for scope stacks, safe error handling
5. **Error Propagation**: No exceptions across FFI boundary, explicit error parameters
6. **Type Safety**: Opaque pointers prevent ABI issues

## Architecture Layers

```
┌─────────────────────────────────────────────────────────┐
│  TypeScript/JavaScript Layer (Future)                   │
│  - Type-safe tensor operations                          │
│  - Automatic memory management                          │
│  - Promise-based async operations                       │
└─────────────────────────────────────────────────────────┘
                          │
                          │ FFI (node-ffi-napi or N-API)
                          ▼
┌─────────────────────────────────────────────────────────┐
│  C Shim Layer (ts_torch) - THIS LIBRARY                 │
│  - C API functions (ts_tensor_*, ts_scope_*, etc.)     │
│  - Opaque handle types                                  │
│  - Error handling (ts_Error struct)                    │
│  - Memory management (scope-based RAII)                │
└─────────────────────────────────────────────────────────┘
                          │
                          │ reinterpret_cast
                          ▼
┌─────────────────────────────────────────────────────────┐
│  LibTorch C++ API                                        │
│  - torch::Tensor                                        │
│  - torch::nn::Module                                    │
│  - torch::optim::Optimizer                              │
│  - Autograd engine                                      │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Hardware Backends                                       │
│  - CPU (always available)                               │
│  - CUDA (NVIDIA GPUs)                                   │
│  - MPS (Apple Silicon)                                  │
└─────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Opaque Handle Types

```c
typedef struct ts_Tensor* ts_TensorHandle;
typedef struct ts_Module* ts_ModuleHandle;
typedef struct ts_Optimizer* ts_OptimizerHandle;
typedef struct ts_Scope* ts_ScopeHandle;
```

**Purpose**: Hide C++ implementation details behind type-safe pointers.

**Implementation**:

```cpp
struct ts_Tensor {
    torch::Tensor tensor;
    explicit ts_Tensor(torch::Tensor t) : tensor(std::move(t)) {}
};
```

**Benefits**:

- ABI stability (C++ std::vector layout doesn't matter to C caller)
- Type safety (can't accidentally pass wrong handle type)
- Forward compatibility (can change internal layout without breaking API)

### 2. Error Handling

```c
typedef struct {
    int code;
    char message[256];
} ts_Error;
```

**Design Rationale**:

- No exceptions across FFI boundary (C doesn't have exceptions)
- Fixed-size struct for easy FFI marshalling
- Per-call error parameter for thread safety
- Non-zero code indicates error occurred

**Usage Pattern**:

```c
ts_Error error = {0};
ts_TensorHandle tensor = ts_tensor_zeros(shape, 2, ..., &error);
if (ts_error_occurred(&error)) {
    printf("Error: %s\n", error.message);
    return;
}
```

### 3. Memory Management

#### Manual Management

```c
ts_TensorHandle tensor = ts_tensor_zeros(...);
// Use tensor
ts_tensor_delete(tensor);  // Manual cleanup
```

#### Scope-Based RAII

```c
ts_ScopeHandle scope = ts_scope_begin();

ts_TensorHandle a = ts_tensor_ones(...);  // Automatically registered
ts_TensorHandle b = ts_tensor_ones(...);  // Automatically registered
ts_TensorHandle c = ts_tensor_add(a, b, ...);  // Automatically registered

ts_TensorHandle result = ts_scope_escape_tensor(scope, c);  // Won't be deleted

ts_scope_end(scope);  // Deletes a, b, c but not result
```

**Implementation**:

```cpp
thread_local std::vector<std::unique_ptr<ts_Scope>> g_scope_stack;

struct ts_Scope {
    std::unordered_set<ts_TensorHandle> tensors;
    std::unordered_set<ts_TensorHandle> escaped;
};
```

**Benefits**:

- Prevents memory leaks in complex operations
- Optional (can still use manual management)
- Thread-safe (thread-local storage)
- Minimal overhead (pointer set operations)

### 4. Data Type System

```c
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
```

**Mapping to LibTorch**:

```cpp
torch::ScalarType dtype_to_scalar_type(ts_DType dtype) {
    switch (dtype) {
        case TS_DTYPE_FLOAT32: return torch::kFloat32;
        case TS_DTYPE_FLOAT64: return torch::kFloat64;
        // ...
    }
}
```

### 5. Device Management

```c
typedef enum {
    TS_DEVICE_CPU = 0,
    TS_DEVICE_CUDA = 1,
    TS_DEVICE_MPS = 2
} ts_DeviceType;
```

**Device Operations**:

- `ts_cuda_is_available()` - Runtime CUDA detection
- `ts_cuda_device_count()` - Query number of GPUs
- `ts_tensor_to_device()` - Move tensor between devices
- `ts_tensor_cpu()`, `ts_tensor_cuda()` - Convenience wrappers

## API Categories

### Tensor Creation

- `ts_tensor_zeros`, `ts_tensor_ones`, `ts_tensor_randn`
- `ts_tensor_empty`, `ts_tensor_from_buffer`

### Tensor Properties

- `ts_tensor_ndim`, `ts_tensor_shape`, `ts_tensor_size`
- `ts_tensor_dtype`, `ts_tensor_numel`
- `ts_tensor_device_type`, `ts_tensor_device_index`

### Tensor Operations

- **Arithmetic**: `ts_tensor_add`, `ts_tensor_sub`, `ts_tensor_mul`, `ts_tensor_div`
- **Linear Algebra**: `ts_tensor_matmul`, `ts_tensor_transpose`
- **Shape**: `ts_tensor_reshape`
- **Reduction**: `ts_tensor_sum`, `ts_tensor_mean`
- **Activation**: `ts_tensor_relu`, `ts_tensor_sigmoid`, `ts_tensor_softmax`, `ts_tensor_tanh`
- **Comparison**: `ts_tensor_eq`, `ts_tensor_ne`, `ts_tensor_lt`, `ts_tensor_le`, `ts_tensor_gt`, `ts_tensor_ge`

### Autograd

- `ts_tensor_backward()` - Compute gradients
- `ts_tensor_grad()` - Get gradient tensor
- `ts_tensor_set_requires_grad()` - Enable gradient tracking
- `ts_tensor_requires_grad()` - Query gradient tracking status
- `ts_tensor_is_leaf()` - Check if leaf node in computation graph

### Memory Management

- `ts_tensor_delete()` - Manual cleanup
- `ts_tensor_clone()` - Deep copy
- `ts_tensor_detach()` - Detach from autograd graph
- `ts_scope_begin()`, `ts_scope_end()` - Scope management
- `ts_scope_register_tensor()`, `ts_scope_escape_tensor()` - Scope control

## Thread Safety

### Thread-Safe Components

- Error handling (per-call error parameter)
- LibTorch operations (internally thread-safe)
- Scope stack (thread-local storage)

### Not Thread-Safe

- Concurrent access to same tensor (requires external synchronization)
- Concurrent modification of tensor data

### Best Practices

```c
// Each thread has its own scope stack
void worker_thread() {
    ts_ScopeHandle scope = ts_scope_begin();
    // ... operations ...
    ts_scope_end(scope);
}

// Sharing tensors between threads requires synchronization
pthread_mutex_t tensor_lock;
ts_TensorHandle shared_tensor;

void thread_a() {
    pthread_mutex_lock(&tensor_lock);
    ts_tensor_add(shared_tensor, ...);
    pthread_mutex_unlock(&tensor_lock);
}
```

## Performance Considerations

### Zero-Copy Operations

```c
// Direct memory access (no copy)
void* ptr = ts_tensor_data_ptr(tensor, &error);
float* data = (float*)ptr;

// Copy to buffer (explicit copy)
float buffer[100];
ts_tensor_copy_to_buffer(tensor, buffer, sizeof(buffer), &error);
```

### Async CUDA Operations

- CUDA operations are asynchronous by default
- Use CUDA streams (future enhancement) for explicit control
- CPU-GPU transfers may be synchronous (depends on flags)

### Memory Layout

- Tensors are row-major (C-contiguous) by default
- Use `contiguous()` in C++ layer to ensure layout
- Direct data access requires contiguous memory

### Optimization Tips

1. **Minimize Device Transfers**: Keep tensors on GPU when possible
2. **Use In-Place Operations**: Future enhancement for `ts_tensor_add_`
3. **Batch Operations**: Process multiple samples together
4. **Scope Overhead**: Minimal, but manual management can be slightly faster
5. **Autograd Overhead**: Only track gradients when needed

## Error Handling Patterns

### Basic Pattern

```c
ts_Error error = {0};
ts_TensorHandle tensor = ts_tensor_zeros(..., &error);
if (ts_error_occurred(&error)) {
    fprintf(stderr, "Error: %s\n", error.message);
    return NULL;
}
```

### Cleanup on Error

```c
ts_Error error = {0};
ts_TensorHandle a = NULL, b = NULL, c = NULL;

a = ts_tensor_ones(..., &error);
if (ts_error_occurred(&error)) goto cleanup;

b = ts_tensor_ones(..., &error);
if (ts_error_occurred(&error)) goto cleanup;

c = ts_tensor_add(a, b, &error);
if (ts_error_occurred(&error)) goto cleanup;

cleanup:
    if (a) ts_tensor_delete(a);
    if (b) ts_tensor_delete(b);
    if (c) ts_tensor_delete(c);
    return ts_error_occurred(&error) ? NULL : c;
```

### Scope-Based (Recommended)

```c
ts_Error error = {0};
ts_ScopeHandle scope = ts_scope_begin();

ts_TensorHandle a = ts_tensor_ones(..., &error);
if (ts_error_occurred(&error)) {
    ts_scope_end(scope);
    return NULL;
}

ts_TensorHandle result = ts_tensor_add(a, ..., &error);
result = ts_scope_escape_tensor(scope, result);
ts_scope_end(scope);

return result;
```

## Build System

### CMake Structure

```
native/
├── CMakeLists.txt           # Main build configuration
├── cmake/
│   └── ts_torch-config.cmake.in  # Package config template
├── include/
│   └── ts_torch.h          # Public C API
├── src/
│   └── ts_torch.cpp        # Implementation
├── examples/
│   ├── CMakeLists.txt
│   └── simple_example.c
└── tests/
    ├── CMakeLists.txt
    └── test_basic.c
```

### Dependencies

- CMake 3.18+
- C++17 compiler
- LibTorch (downloaded separately)

### Output Artifacts

- **Linux**: `libts_torch.so`
- **macOS**: `libts_torch.dylib`
- **Windows**: `ts_torch.dll`, `ts_torch.lib`

## Future Enhancements

### API Extensions

- [ ] Module API (`ts_module_*` functions)
- [ ] Optimizer API (`ts_optimizer_*` functions)
- [ ] Custom operators
- [ ] JIT/TorchScript integration
- [ ] Distributed training support
- [ ] Quantization support

### Performance

- [ ] CUDA stream management
- [ ] In-place operations (`ts_tensor_add_`, etc.)
- [ ] Memory pool management
- [ ] Async operations with callbacks

### Developer Experience

- [ ] Better error messages with stack traces
- [ ] Debug mode with tensor validation
- [ ] Performance profiling hooks
- [ ] Memory leak detection tools

### Platforms

- [ ] WebAssembly build
- [ ] Mobile (Android/iOS) support
- [ ] ARM optimizations

## Integration with TypeScript

### Future Node.js Addon

```typescript
// TypeScript wrapper will use node-ffi-napi or N-API
import { Library } from 'ffi-napi'

const libts_torch = Library('libts_torch', {
  ts_tensor_zeros: ['pointer', ['pointer', 'size_t', 'int', 'int', 'int', 'pointer']],
  // ... other functions
})

class Tensor {
  private handle: Buffer

  constructor(shape: number[], dtype: DType = DType.Float32) {
    const error = Buffer.alloc(260) // ts_Error size
    this.handle = libts_torch.ts_tensor_zeros(
      Buffer.from(new BigInt64Array(shape).buffer),
      shape.length,
      dtype,
      DeviceType.CPU,
      0,
      error,
    )
  }

  // ... methods
}
```

### Design Goals for TS Wrapper

1. **Type Safety**: Full TypeScript types with generics
2. **Memory Safety**: Automatic disposal with finalizers
3. **API Parity**: Match PyTorch API where possible
4. **Performance**: Zero-copy where feasible
5. **Async**: Promise-based for long-running ops

## License

MIT License

## Contributing

See main repository for contribution guidelines.
