#!/usr/bin/env bun
/**
 * Benchmark CLI entry point
 *
 * Usage:
 *   bun run benchmark/index.ts           # Run all benchmarks
 *   bun run benchmark/index.ts --category core
 *   bun run benchmark/index.ts --filter ffi
 *   bun run benchmark/index.ts --json
 */

import { resolve } from 'node:path'
import { BenchmarkRunner } from './lib/runner.js'
import type { BenchmarkConfig } from './lib/types.js'

/**
 * Parse command line arguments
 */
function parseArgs(): BenchmarkConfig {
  const args = process.argv.slice(2)
  const config: BenchmarkConfig = {
    time: 1000,
    warmup: true,
    output: {
      console: true,
      json: false,
    },
  }

  for (let i = 0; i < args.length; i++) {
    const arg = args[i]

    if (arg === '--category' && args[i + 1]) {
      config.category = args[++i]
    } else if (arg === '--filter' && args[i + 1]) {
      config.filter = args[++i]
    } else if (arg === '--json') {
      config.output!.json = true
    } else if (arg === '--time' && args[i + 1]) {
      config.time = parseInt(args[++i]!, 10)
    } else if (arg === '--no-warmup') {
      config.warmup = false
    } else if (arg === '--help' || arg === '-h') {
      printHelp()
      process.exit(0)
    }
  }

  return config
}

/**
 * Print help message
 */
function printHelp(): void {
  console.log(`
ts-torch Benchmark Suite

Usage:
  bun run benchmark/index.ts [options]

Options:
  --category <name>   Filter by category (core, nn, optim)
  --filter <pattern>  Filter benchmarks by name pattern
  --json              Output results to JSON file
  --time <ms>         Time per benchmark in ms (default: 1000)
  --no-warmup         Skip warmup phase
  --help, -h          Show this help message

Examples:
  bun run benchmark/index.ts                    # Run all benchmarks
  bun run benchmark/index.ts --category core    # Run core benchmarks only
  bun run benchmark/index.ts --filter ffi       # Run FFI-related benchmarks
  bun run benchmark/index.ts --json             # Save results to JSON
`)
}

/**
 * Main entry point
 */
async function main(): Promise<void> {
  const config = parseArgs()
  const runner = new BenchmarkRunner(config)

  // Discover benchmarks from the benchmark directory
  const benchmarkDir = resolve(import.meta.dirname, '.')
  await runner.discover(benchmarkDir)

  // Run benchmarks
  await runner.run()
}

main().catch((err) => {
  console.error('Benchmark failed:', err)
  process.exit(1)
})
