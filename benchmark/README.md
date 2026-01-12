# ts-torch Benchmarks

Performance benchmarks for the ts-torch library, including FFI overhead measurements.

## Quick Start

```bash
# Run all benchmarks
bun run bench

# Run specific category
bun run bench:core    # Tensor operations
bun run bench:nn      # Neural network modules
bun run bench:optim   # Optimizers

# Run FFI overhead tests
bun run bench:ffi

# Save results to JSON
bun run bench:json
```

## CLI Options

```
bun run benchmark/index.ts [options]

Options:
  --category <name>   Filter by category (core, nn, optim)
  --filter <pattern>  Filter benchmarks by name pattern
  --json              Output results to JSON file
  --time <ms>         Time per benchmark in ms (default: 1000)
  --no-warmup         Skip warmup phase
  --help, -h          Show help message
```

## Benchmark Categories

### Core (`benchmark/core/`)

- **ffi-overhead.bench.ts** - Measures FFI call overhead isolated from computation
- **tensor-creation.bench.ts** - Factory functions (zeros, ones, randn, etc.)
- **element-wise.bench.ts** - add, sub, mul, div operations
- **matrix-ops.bench.ts** - matmul, transpose, reshape
- **activations.bench.ts** - relu, sigmoid, softmax, etc.
- **reductions.bench.ts** - sum, mean operations

### NN Modules (`benchmark/nn/`)

- **linear.bench.ts** - Linear layer forward pass
- **conv2d.bench.ts** - Conv2d layer
- **pooling.bench.ts** - MaxPool2d, AvgPool2d, AdaptiveAvgPool2d
- **normalization.bench.ts** - BatchNorm, LayerNorm

### Optimizers (`benchmark/optim/`)

- **optimizers.bench.ts** - SGD, Adam, RMSprop step performance
- **training-loop.bench.ts** - Full forward + backward + step

## Understanding FFI Overhead

The FFI overhead benchmarks help you understand:

1. **Fixed call overhead** - Time spent on each FFI call regardless of tensor size
2. **Data transfer costs** - How overhead scales with tensor size
3. **Crossover point** - When computation dominates over FFI overhead

Key metrics from `bench:ffi`:

```
FFI Overhead Analysis:
  Estimated FFI overhead: ~5-10μs per call
  Estimated compute: ~0.001-0.01ns per element
```

Small tensors (< 100 elements) are FFI-dominated, while large tensors (> 10K elements) are compute-dominated.

## Output Format

### Console Output

```
ts-torch Benchmark Suite
========================

[core] FFI Overhead
  createError()            │   2.34M ops/sec │    0.43μs │  ±3.1%
  tensor zeros [1]         │    156K ops/sec │    6.41μs │  ±4.2%
  add [1024, 1024]         │     478 ops/sec │    2.09ms │  ±0.9%

Summary
-------
Total benchmarks: 47
Total time: 52.3s
```

### JSON Output

Results are saved to `benchmark/results/` with timestamps:

```json
{
  "timestamp": "2024-01-12T10:30:00Z",
  "platform": { "os": "win32", "arch": "x64", "bunVersion": "1.3.5" },
  "suites": [
    {
      "name": "FFI Overhead",
      "category": "core",
      "benchmarks": [
        { "name": "createError()", "opsPerSec": 2340000, "meanUs": 0.43, ... }
      ]
    }
  ]
}
```

## Adding New Benchmarks

Create a new `.bench.ts` file in the appropriate category:

```typescript
import { Bench } from 'tinybench'
import { torch, run } from '@ts-torch/core'
import type { BenchmarkSuite, BenchmarkConfig } from '../lib/types.js'

export const suite: BenchmarkSuite = {
  name: 'My Benchmarks',
  category: 'core', // or 'nn', 'optim'

  async run(config: BenchmarkConfig) {
    const bench = new Bench({
      time: config.time ?? 1000,
      warmup: config.warmup ?? true,
    })

    bench.add('my operation', () => {
      run(() => {
        // Benchmark code here
        // Use run() for proper memory management
        const t = torch.zeros([100, 100] as const)
        return t.add(t)
      })
    })

    await bench.run()
    return bench
  },
}

export default suite
```

## Tips

1. **Always use `run()` scope** - Ensures tensors are properly cleaned up
2. **Include tensor creation in benchmark** - Unless measuring forward-only
3. **Test multiple sizes** - FFI overhead varies with tensor size
4. **Use `--time` for accuracy** - Longer runs reduce variance
