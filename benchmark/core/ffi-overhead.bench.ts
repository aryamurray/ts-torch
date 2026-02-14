/**
 * FFI Overhead Benchmarks
 *
 * Measures the overhead of FFI calls to isolate it from computation time.
 * Key overhead sources:
 * - BigInt64Array shape allocation
 * - Tensor creation/deletion
 * - Library function calls
 */

import { Bench } from 'tinybench'
import { device, run } from '@ts-torch/core'
import type { BenchmarkSuite, BenchmarkConfig } from '../lib/types.js'

const cpu = device.cpu()
import { estimateFfiOverhead } from '../lib/utils.js'

export const suite: BenchmarkSuite = {
  name: 'FFI Overhead',
  category: 'core',

  async run(config: BenchmarkConfig) {
    const bench = new Bench({
      time: config.time ?? 1000,
      warmup: config.warmup ?? true,
    })

    // 1. Pure JS allocations (baseline)
    bench.add('ArrayBuffer(260) allocation', () => {
      new ArrayBuffer(260)
    })

    bench.add('BigInt64Array([2]) allocation', () => {
      new BigInt64Array([2n, 3n])
    })

    bench.add('BigInt64Array([4]) allocation', () => {
      new BigInt64Array([2n, 3n, 4n, 5n])
    })

    // 2. Tensor creation at different sizes (to see FFI vs compute)
    bench.add('tensor zeros [1] (minimal)', () => {
      run(() => {
        cpu.zeros([1] as const)
      })
    })

    bench.add('tensor zeros [2, 2] (4 elements)', () => {
      run(() => {
        cpu.zeros([2, 2] as const)
      })
    })

    bench.add('tensor zeros [8, 8] (64 elements)', () => {
      run(() => {
        cpu.zeros([8, 8] as const)
      })
    })

    bench.add('tensor zeros [32, 32] (1K elements)', () => {
      run(() => {
        cpu.zeros([32, 32] as const)
      })
    })

    bench.add('tensor zeros [128, 128] (16K elements)', () => {
      run(() => {
        cpu.zeros([128, 128] as const)
      })
    })

    // 3. Operation overhead: add small vs large tensors
    bench.add('add [2, 2] (FFI-dominated)', () => {
      run(() => {
        const a = cpu.ones([2, 2] as const)
        const b = cpu.ones([2, 2] as const)
        return a.add(b)
      })
    })

    bench.add('add [1024, 1024] (compute-dominated)', () => {
      run(() => {
        const a = cpu.ones([1024, 1024] as const)
        const b = cpu.ones([1024, 1024] as const)
        return a.add(b)
      })
    })

    // 4. Multiple FFI calls in sequence
    bench.add('10x sequential add [100, 100]', () => {
      run(() => {
        let t = cpu.ones([100, 100] as const)
        for (let i = 0; i < 10; i++) {
          t = t.add(cpu.ones([100, 100] as const))
        }
        return t
      })
    })

    // 5. Tensor creation methods comparison
    bench.add('zeros [256, 256]', () => {
      run(() => {
        cpu.zeros([256, 256] as const)
      })
    })

    bench.add('ones [256, 256]', () => {
      run(() => {
        cpu.ones([256, 256] as const)
      })
    })

    bench.add('empty [256, 256]', () => {
      run(() => {
        cpu.empty([256, 256] as const)
      })
    })

    bench.add('randn [256, 256]', () => {
      run(() => {
        cpu.randn([256, 256] as const)
      })
    })

    await bench.run()

    // Calculate and log FFI overhead estimate
    const tasks = bench.tasks
    const small = tasks.find((t) => t.name.includes('[2, 2]') && t.name.includes('add'))
    const large = tasks.find((t) => t.name.includes('[1024, 1024]') && t.name.includes('add'))

    if (small?.result && large?.result) {
      const overhead = estimateFfiOverhead(
        small.result.mean * 1000, // ms to us
        large.result.mean * 1000,
        4, // 2x2 = 4 elements
        1024 * 1024, // 1024x1024 elements
      )
      console.log('')
      console.log('  FFI Overhead Analysis:')
      console.log(`    Estimated FFI overhead: ${overhead.ffiOverheadUs.toFixed(1)}μs per call`)
      console.log(`    Estimated compute: ${overhead.computePerElementNs.toFixed(3)}ns per element`)
    }

    return bench
  },
}

export default suite
