/**
 * Benchmark utility functions
 */

import { run } from '@ts-torch/core'

/**
 * Format time value for display
 * @param us - Time in microseconds
 */
export function formatTime(us: number): string {
  if (us < 1) {
    return `${(us * 1000).toFixed(1)}ns`
  }
  if (us < 1000) {
    return `${us.toFixed(1)}μs`
  }
  if (us < 1000000) {
    return `${(us / 1000).toFixed(2)}ms`
  }
  return `${(us / 1000000).toFixed(2)}s`
}

/**
 * Format ops/sec for display
 * @param ops - Operations per second
 */
export function formatOps(ops: number): string {
  if (ops >= 1000000) {
    return `${(ops / 1000000).toFixed(2)}M`
  }
  if (ops >= 1000) {
    return `${(ops / 1000).toFixed(2)}K`
  }
  return ops.toFixed(0)
}

/**
 * Pad string to width
 */
export function pad(str: string, width: number, align: 'left' | 'right' = 'left'): string {
  if (str.length >= width) return str
  const padding = ' '.repeat(width - str.length)
  return align === 'left' ? str + padding : padding + str
}

/**
 * Run function in a memory scope (convenience wrapper)
 */
export function scoped<T>(fn: () => T): T {
  return run(fn)
}

/**
 * Calculate FFI overhead estimate from small vs large tensor timings
 * Assumes: time = ffiOverhead + computePerElement * elements
 */
export function estimateFfiOverhead(
  smallTime: number,
  largeTime: number,
  smallElements: number,
  largeElements: number,
): { ffiOverheadUs: number; computePerElementNs: number } {
  const computePerElement = (largeTime - smallTime) / (largeElements - smallElements)
  const ffiOverhead = smallTime - computePerElement * smallElements

  return {
    ffiOverheadUs: Math.max(0, ffiOverhead),
    computePerElementNs: computePerElement * 1000, // us to ns
  }
}

/**
 * Standard tensor sizes for benchmarks
 */
export const STANDARD_SIZES: Array<[number, number]> = [
  [8, 8], // 64 elements - FFI overhead dominates
  [32, 32], // 1,024 elements
  [128, 128], // 16,384 elements
  [512, 512], // 262,144 elements
  [1024, 1024], // 1,048,576 elements - compute dominates
]

/**
 * Calculate number of elements in a shape
 */
export function numel(shape: readonly number[]): number {
  return shape.reduce((a, b) => a * b, 1)
}

/**
 * Sleep for ms milliseconds
 */
export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}
