/**
 * Benchmark system type definitions
 */

import type { Bench, Task } from 'tinybench'

/**
 * Configuration for benchmark runs
 */
export interface BenchmarkConfig {
  /** Time in ms per benchmark (default: 1000) */
  time?: number
  /** Fixed number of iterations (overrides time) */
  iterations?: number
  /** Whether to warm up before measuring (default: true) */
  warmup?: boolean
  /** Tensor sizes to test */
  sizes?: Array<[number, number]>
  /** Output configuration */
  output?: OutputConfig
  /** Filter benchmarks by name pattern */
  filter?: string
  /** Filter by category */
  category?: string
}

/**
 * Output configuration
 */
export interface OutputConfig {
  /** Output to console (default: true) */
  console?: boolean
  /** Output to JSON file */
  json?: boolean
  /** Path for JSON output */
  jsonPath?: string
}

/**
 * Individual benchmark result
 */
export interface BenchmarkResult {
  /** Benchmark name */
  name: string
  /** Operations per second */
  opsPerSec: number
  /** Mean time per operation in microseconds */
  meanUs: number
  /** Standard deviation */
  stdDev: number
  /** Minimum time */
  minUs: number
  /** Maximum time */
  maxUs: number
  /** 75th percentile */
  p75Us: number
  /** 99th percentile */
  p99Us: number
  /** Number of samples */
  samples: number
  /** Relative margin of error (percentage) */
  rme: number
}

/**
 * Suite results
 */
export interface SuiteResult {
  /** Suite name */
  name: string
  /** Category (core, nn, optim, etc.) */
  category: string
  /** Individual benchmark results */
  benchmarks: BenchmarkResult[]
  /** Total time to run suite in ms */
  duration: number
}

/**
 * Full benchmark results
 */
export interface BenchmarkResults {
  /** ISO timestamp */
  timestamp: string
  /** Platform information */
  platform: PlatformInfo
  /** Suite results */
  suites: SuiteResult[]
  /** Total duration in ms */
  totalDuration: number
}

/**
 * Platform information
 */
export interface PlatformInfo {
  os: string
  arch: string
  nodeVersion: string
  bunVersion?: string
}

/**
 * Benchmark suite definition
 */
export interface BenchmarkSuite {
  /** Suite name */
  name: string
  /** Category (core, nn, optim, etc.) */
  category: string
  /** Run the benchmark suite */
  run(config: BenchmarkConfig): Promise<Bench>
}

/**
 * Extract results from tinybench Task
 */
export function extractTaskResult(task: Task): BenchmarkResult | null {
  const result = task.result
  if (!result) return null

  const hz = result.hz
  const meanUs = result.mean * 1000 // ms to us

  return {
    name: task.name,
    opsPerSec: Math.round(hz),
    meanUs: meanUs,
    stdDev: (result.sd ?? 0) * 1000, // ms to us
    minUs: (result.min ?? meanUs) * 1000,
    maxUs: (result.max ?? meanUs) * 1000,
    p75Us: (result.p75 ?? meanUs) * 1000,
    p99Us: (result.p99 ?? meanUs) * 1000,
    samples: result.samples?.length ?? 0,
    rme: result.rme ?? 0,
  }
}
