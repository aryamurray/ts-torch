# ts-torch FFI Bindings

Bun FFI bindings that connect TypeScript to the native C library for tensor operations.

## Architecture

### Files

- **`symbols.ts`** (592 lines) - Complete FFI symbol definitions for all C functions
- **`loader.ts`** (206 lines) - Platform-specific library loading and resolution
- **`error.ts`** (218 lines) - Error handling utilities and TorchError class
- **`index.ts`** (47 lines) - Public API exports

### Key Components

#### 1. Symbol Definitions (`symbols.ts`)

Defines all C function signatures with proper Bun FFI types:

```typescript
export const FFI_SYMBOLS = {
  ts_tensor_zeros: {
    args: ["ptr", "i32", "i32", "bool", "ptr"],
    returns: "ptr",
  },
  // ... 40+ more functions
};
```

Categories:

- Tensor creation (zeros, ones, randn, from_buffer, empty)
- Tensor properties (ndim, size, shape, dtype, numel, requires_grad)
- Memory management (delete, clone, detach, data_ptr, copy_to_buffer)
- Operations (add, sub, mul, div, matmul, transpose, reshape)
- Reductions (sum, mean, max, min)
- Activations (relu, sigmoid, softmax, tanh)
- Autograd (backward, grad, set_requires_grad, zero_grad)
- Scope management (scope_begin, scope_end, register, escape)
- Device management (CUDA availability, device transfers)

#### 2. Library Loader (`loader.ts`)

Handles platform detection and library resolution:

```typescript
// Get platform-specific package name
const { packageName, libraryName } = getPlatformPackage();
// => { packageName: "@ts-torch/darwin-arm64", libraryName: "libts_torch" }

// Resolve library path
const path = getLibraryPath();
// => "/path/to/@ts-torch/darwin-arm64/libts_torch.dylib"

// Load library (cached)
const lib = getLib();
const result = lib.symbols.ts_tensor_zeros(...);
```

Resolution order:

1. `TS_TORCH_LIB` environment variable
2. Platform-specific npm package
3. Local development paths (workspace monorepo)

#### 3. Error Handling (`error.ts`)

Manages C error struct (260 bytes: 4-byte code + 256-byte message):

```typescript
// Manual error handling
const err = createError();
const handle = lib.symbols.ts_tensor_zeros(ptr(shape), 2, 0, false, err);
checkError(err); // Throws TorchError if code != 0

// Automatic error handling
const handle = withError((err) => lib.symbols.ts_tensor_zeros(ptr(shape), 2, 0, false, err));
```

Error codes:

- `OK = 0` - Success
- `NULL_POINTER = 1` - Null pointer encountered
- `INVALID_SHAPE = 2` - Invalid tensor shape
- `INVALID_DTYPE = 3` - Invalid data type
- `DIMENSION_MISMATCH = 4` - Tensor dimensions don't match
- `OUT_OF_MEMORY = 5` - Memory allocation failed
- `CUDA_ERROR = 6` - CUDA operation failed
- `GRAD_ERROR = 7` - Gradient computation error
- `SCOPE_ERROR = 8` - Memory scope error
- `UNKNOWN = 99` - Unknown error

## Usage Examples

### Basic Tensor Creation

```typescript
import { getLib, createError, checkError, ptr } from "@ts-torch/core/ffi";

const lib = getLib();
const err = createError();

// Create shape array
const shape = new BigInt64Array([2, 3]);

// Create tensor
const handle = lib.symbols.ts_tensor_zeros(
  ptr(shape), // shape pointer
  2, // ndim
  0, // dtype (0=f32, 1=f64, 2=i32, 3=i64)
  false, // requires_grad
  err,
);

checkError(err);
```

### With Automatic Error Handling

```typescript
import { getLib, withError, ptr } from "@ts-torch/core/ffi";

const lib = getLib();
const shape = new BigInt64Array([2, 3]);

const handle = withError((err) => lib.symbols.ts_tensor_zeros(ptr(shape), 2, 0, false, err));
```

### Tensor Operations

```typescript
import { getLib, withError } from "@ts-torch/core/ffi";

const lib = getLib();

// Create two tensors
const a = withError((err) => lib.symbols.ts_tensor_ones(ptr(shape), 2, 0, false, err));
const b = withError((err) => lib.symbols.ts_tensor_ones(ptr(shape), 2, 0, false, err));

// Add tensors
const result = withError((err) => lib.symbols.ts_tensor_add(a, b, err));

// Cleanup
lib.symbols.ts_tensor_delete(a);
lib.symbols.ts_tensor_delete(b);
lib.symbols.ts_tensor_delete(result);
```

### Scope Management (Automatic Cleanup)

```typescript
import { getLib, withError } from "@ts-torch/core/ffi";

const lib = getLib();

// Begin scope
const scopeId = lib.symbols.ts_scope_begin();

try {
  // Create tensors (automatically registered with scope)
  const a = withError((err) => lib.symbols.ts_tensor_ones(ptr(shape), 2, 0, false, err));

  withError((err) => lib.symbols.ts_scope_register_tensor(a, err));

  // ... use tensors ...
} finally {
  // Cleanup all tensors in scope
  withError((err) => lib.symbols.ts_scope_end(scopeId, err));
}
```

### Reading Tensor Data

```typescript
import { getLib, withError, ptr } from '@ts-torch/core/ffi';

const lib = getLib();
const handle = /* ... create tensor ... */;

// Get number of elements
const numel = withError(err =>
  lib.symbols.ts_tensor_numel(handle, err)
);

// Allocate buffer
const buffer = new Float32Array(Number(numel));

// Copy data to buffer
withError(err =>
  lib.symbols.ts_tensor_copy_to_buffer(
    handle,
    ptr(buffer),
    BigInt(buffer.byteLength),
    err
  )
);

console.log(buffer); // Float32Array with tensor data
```

### CUDA Operations

```typescript
import { getLib } from "@ts-torch/core/ffi";

const lib = getLib();

// Check CUDA availability
if (lib.symbols.ts_cuda_is_available()) {
  const deviceCount = lib.symbols.ts_cuda_device_count();
  console.log(`CUDA available with ${deviceCount} devices`);

  // Move tensor to GPU
  const gpuHandle = withError((err) =>
    lib.symbols.ts_tensor_to_device(
      cpuHandle,
      1, // device_type: 1=CUDA
      0, // device_id: first GPU
      err,
    ),
  );
}
```

## FFI Type Mappings

| C Type    | FFIType | TypeScript |
| --------- | ------- | ---------- |
| `void*`   | `ptr`   | `Pointer`  |
| `int32_t` | `i32`   | `number`   |
| `int64_t` | `i64`   | `bigint`   |
| `float`   | `f32`   | `number`   |
| `double`  | `f64`   | `number`   |
| `bool`    | `bool`  | `boolean`  |
| `void`    | `void`  | `void`     |

## Platform Support

Supported platforms (detected via `process.platform` and `process.arch`):

- macOS ARM64 → `@ts-torch/darwin-arm64`
- macOS x64 → `@ts-torch/darwin-x64`
- Linux x64 → `@ts-torch/linux-x64`
- Linux ARM64 → `@ts-torch/linux-arm64`
- Windows x64 → `@ts-torch/win32-x64`
- Windows ARM64 → `@ts-torch/win32-arm64`

## Error Handling Best Practices

1. **Always check errors** after FFI calls that take error pointer
2. **Use `withError()`** for cleaner error handling
3. **Validate inputs** before FFI calls (shapes, dtypes)
4. **Check null pointers** for returned handles
5. **Clean up resources** using `ts_tensor_delete()` or scopes

## References

- [Bun FFI Documentation](https://bun.sh/docs/api/ffi)
- C Library API (see `native/` directory)
- Platform Packages (see `packages/@ts-torch/*-*/`)
