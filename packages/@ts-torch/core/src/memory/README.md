# Memory Management System

Automatic memory management for ts-torch using scoped cleanup and tensor pooling.

## Overview

The memory management system provides two key features:

1. **Scoped Memory Management** - Automatic cleanup using `torch.run()` (similar to PyTorch's `torch.no_grad()`)
2. **Tensor Pooling** - Object reuse for reduced allocation overhead

## Quick Start

```typescript
import { run } from '@ts-torch/core/memory'
import { zeros, ones } from '@ts-torch/core'

// All tensors auto-freed when scope exits
const result = run(() => {
  const a = zeros([100, 100]) // Auto-freed
  const b = ones([100, 100]) // Auto-freed
  const c = a.add(b) // Auto-freed
  return c.escape() // Escaped - not freed
})
// a, b are freed; result persists
```

## API Reference

### Scope Management

#### `run<T>(fn: () => T): T`

Execute a function with scoped memory management. All tensors created within the scope are automatically freed when it exits.

**Parameters:**

- `fn` - Function to execute within the scope

**Returns:**

- Result of the function

**Example:**

```typescript
const tensor = run(() => {
  const x = zeros([10, 10])
  const y = ones([10, 10])
  const result = x.add(y)
  return result.escape()
})
```

#### `runAsync<T>(fn: () => Promise<T>): Promise<T>`

Async version of `run()` for async operations.

**Example:**

```typescript
const result = await runAsync(async () => {
  const data = await fetchData()
  const tensor = fromBuffer(data)
  const processed = await processAsync(tensor)
  return processed.escape()
})
```

#### `escapeTensor<T>(tensor: T): T`

Escape a tensor from the current scope to prevent it from being freed.

**Throws:**

- Error if not currently in a scope

**Example:**

```typescript
run(() => {
  const temp = zeros([10, 10])
  const keep = ones([10, 10])
  escapeTensor(keep)
  // temp will be freed, keep will persist
})
```

#### `inScope(): boolean`

Check if currently inside a scope.

**Example:**

```typescript
console.log(inScope()) // false
run(() => {
  console.log(inScope()) // true
})
```

#### `scopeDepth(): number`

Get current scope depth (0 if not in a scope).

**Example:**

```typescript
run(() => {
  console.log(scopeDepth()) // 1
  run(() => {
    console.log(scopeDepth()) // 2
  })
})
```

#### `scopeTensorCount(): number`

Get number of tensors tracked in current scope.

**Example:**

```typescript
run(() => {
  zeros([10, 10])
  ones([10, 10])
  console.log(scopeTensorCount()) // 2
})
```

### Tensor Pooling

#### `class TensorPool`

Pool for reusing tensor allocations.

**Constructor:**

```typescript
new TensorPool(maxPoolSize?: number)
```

**Methods:**

##### `acquire<S, D>(shape: S, dtype: D): Tensor | null`

Acquire a tensor from the pool if available.

**Returns:**

- Pooled tensor or `null` if none available

**Example:**

```typescript
const pool = new TensorPool()
const tensor = pool.acquire([10, 10], 'float32') ?? zeros([10, 10])
```

##### `release(tensor: Tensor): void`

Release a tensor back to the pool for reuse.

**Example:**

```typescript
pool.release(tensor)
```

##### `stats(): PoolStats`

Get pool statistics including hit rate.

**Returns:**

```typescript
{
  size: number // Total cached tensors
  hitCount: number // Number of hits
  missCount: number // Number of misses
  hitRate: number // Hit rate (0-1)
  pools: Map<string, number> // Per-key sizes
}
```

##### `clear(): void`

Clear all cached tensors.

##### `clearKey(shape: Shape, dtype: DType): void`

Clear tensors for a specific (shape, dtype) key.

##### `prune(targetSize?: number): void`

Reduce pool size to target (default: half current size).

**Example:**

```typescript
if (pool.stats().size > 1000) {
  pool.prune(500) // Reduce to 500 tensors
}
```

#### `globalTensorPool`

Global tensor pool instance for convenience.

**Example:**

```typescript
import { globalTensorPool } from '@ts-torch/core/memory'

const tensor = globalTensorPool.acquire([10, 10], 'float32') ?? zeros([10, 10])
globalTensorPool.release(tensor)
```

## Usage Patterns

### Training Loop

```typescript
for (let epoch = 0; epoch < 10; epoch++) {
  run(() => {
    for (let batch = 0; batch < batches.length; batch++) {
      const loss = run(() => {
        const input = getBatch(batch)
        const output = model.forward(input)
        const loss = criterion(output, target)

        optimizer.step(loss)
        return loss.item() // Escape scalar value
      })
      // input, output freed here
    }
  })
  // All batch tensors freed here
}
```

### Inference

```typescript
const results = inputs.map((input) => {
  return run(() => {
    const processed = preprocess(input)
    const output = model.forward(processed)
    return output.argmax().item()
  })
  // All intermediate tensors freed
})
```

### With Pooling

```typescript
const pool = new TensorPool()

for (let i = 0; i < 1000; i++) {
  run(() => {
    const grad = pool.acquire([256, 256], 'float32') ?? zeros([256, 256])

    // Use gradient...
    optimizer.step(grad)

    pool.release(grad)
  })
}

console.log('Pool hit rate:', pool.stats().hitRate)
```

### Nested Scopes

```typescript
const result = run(() => {
  const x = randn([100, 100])

  const intermediate = run(() => {
    const y = randn([100, 100])
    const z = x.matmul(y)
    return z.escape()
  })
  // y freed, z persists

  return x.add(intermediate).escape()
})
// x and intermediate freed, result persists
```

### Error Handling

```typescript
try {
  run(() => {
    const tensor = zeros([100, 100])
    throw new Error('Something failed!')
    // tensor still gets cleaned up
  })
} catch (error) {
  console.log('Error caught, memory still cleaned up')
}
```

## Implementation Details

### Scope Stack

Scopes are implemented using a closure-based stack:

```typescript
let currentScope: ScopeContext | null = null

interface ScopeContext {
  id: number
  tensors: Set<Tensor>
  parent: ScopeContext | null
}
```

When entering a scope:

1. Call native `ts_scope_begin()` to get scope ID
2. Create JS scope context
3. Execute user function
4. Call native `ts_scope_end()` to free tensors
5. Restore parent scope

### Native Integration

The scope system syncs with the C++ backend:

- `ts_scope_begin()` - Creates native scope, returns ID
- `ts_scope_end(id)` - Frees all registered tensors in scope
- `ts_scope_register_tensor(handle)` - Register tensor with current scope
- `ts_scope_escape_tensor(handle)` - Remove tensor from scope

### Tensor Pool

The pool uses a `Map<string, Tensor[]>` keyed by `"${dtype}:${shape}"`:

```typescript
private pools: Map<string, Tensor[]> = new Map();
private makeKey(shape: Shape, dtype: DType): string {
  return `${dtype}:[${shape.join(",")}]`;
}
```

Tensors are stored in LIFO (stack) order for cache locality.

## Performance Considerations

### Scope Overhead

- Minimal overhead: just function call + stack push/pop
- Native scope operations are O(1)
- Recommended for any non-trivial computation

### Pool Benefits

- Reduces allocation overhead (typically 10-30% in training loops)
- Most effective for:
  - Repeated same-shaped tensors
  - High-frequency allocation/deallocation
  - Training loops with fixed batch sizes

### Pool Tuning

```typescript
// Small pool for memory-constrained environments
const smallPool = new TensorPool(8)

// Large pool for high-throughput scenarios
const largePool = new TensorPool(128)

// Monitor and prune based on memory pressure
setInterval(() => {
  if (memoryUsage() > threshold) {
    pool.prune()
  }
}, 60000)
```

## Best Practices

1. **Use scopes in loops** - Prevent memory leaks from intermediate tensors
2. **Escape sparingly** - Only escape what you need to keep
3. **Pool hot paths** - Use pooling for high-frequency allocations
4. **Monitor pool stats** - Check hit rates to validate pooling benefit
5. **Nested scopes for control** - Use nested scopes for fine-grained memory management
6. **Error safety** - Scopes ensure cleanup even with exceptions

## Debugging

### Check scope state

```typescript
import { inScope, scopeDepth, scopeTensorCount } from '@ts-torch/core/memory'

console.log('In scope:', inScope())
console.log('Scope depth:', scopeDepth())
console.log('Tensors tracked:', scopeTensorCount())
```

### Monitor pool performance

```typescript
const stats = pool.stats()
console.log(`Pool size: ${stats.size}`)
console.log(`Hit rate: ${(stats.hitRate * 100).toFixed(1)}%`)
console.log(`Efficiency: ${stats.hitCount}/${stats.hitCount + stats.missCount}`)

for (const [key, count] of stats.pools) {
  console.log(`  ${key}: ${count} tensors`)
}
```

## Comparison with Other Frameworks

### PyTorch

PyTorch uses reference counting with Python GC:

```python
# PyTorch - GC-based
x = torch.zeros(10, 10)
y = torch.ones(10, 10)
z = x + y
# x, y freed when out of scope
```

ts-torch scopes are similar to context managers:

```typescript
// ts-torch - scope-based
run(() => {
  const x = zeros([10, 10])
  const y = ones([10, 10])
  const z = x.add(y)
  // x, y freed when scope exits
})
```

### JAX

JAX uses functional programming with immutability:

```python
# JAX - functional
x = jnp.zeros((10, 10))
y = jnp.ones((10, 10))
z = x + y  # Creates new array
```

ts-torch scopes provide similar memory safety with imperative syntax.

### TensorFlow

TensorFlow 2.x uses eager execution with Python GC, similar to PyTorch. ts-torch provides more explicit control.

## See Also

- [Tensor API](../tensor/README.md)
- [FFI Documentation](../ffi/README.md)
- [Examples](./examples.ts)
