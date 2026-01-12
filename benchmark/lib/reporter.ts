/**
 * Benchmark result reporter
 * Outputs results to console and/or JSON files
 */

import { writeFileSync, mkdirSync, existsSync } from 'node:fs'
import { join } from 'node:path'
import type { BenchmarkResults, SuiteResult, BenchmarkResult, OutputConfig } from './types.js'
import { formatTime, formatOps, pad } from './utils.js'

/**
 * Reporter for benchmark results
 */
export class Reporter {
  private startTime: number = 0
  private totalSuites: number = 0

  constructor(private config: OutputConfig = {}) {}

  /**
   * Called when benchmark run starts
   */
  start(numSuites: number): void {
    this.startTime = Date.now()
    this.totalSuites = numSuites

    if (this.config.console !== false) {
      console.log('')
      console.log('ts-torch Benchmark Suite')
      console.log('========================')
      console.log('')
    }
  }

  /**
   * Called when a suite starts
   */
  suiteStart(name: string, category: string): void {
    if (this.config.console !== false) {
      console.log(`[${category}] ${name}`)
    }
  }

  /**
   * Called when a suite ends
   */
  suiteEnd(result: SuiteResult): void {
    if (this.config.console !== false) {
      for (const bench of result.benchmarks) {
        this.printBenchmark(bench)
      }
      console.log('')
    }
  }

  /**
   * Print a single benchmark result
   */
  private printBenchmark(bench: BenchmarkResult): void {
    const name = pad(bench.name, 35)
    const ops = pad(`${formatOps(bench.opsPerSec)} ops/sec`, 16, 'right')
    const time = pad(formatTime(bench.meanUs), 12, 'right')
    const rme = `±${bench.rme.toFixed(1)}%`

    console.log(`  ${name} │ ${ops} │ ${time} │ ${rme}`)
  }

  /**
   * Called when all benchmarks complete
   */
  finish(results: BenchmarkResults): void {
    const duration = Date.now() - this.startTime

    if (this.config.console !== false) {
      this.printSummary(results, duration)
    }

    if (this.config.json) {
      this.writeJson(results)
    }
  }

  /**
   * Print summary statistics
   */
  private printSummary(results: BenchmarkResults, duration: number): void {
    const allBenchmarks = results.suites.flatMap((s) => s.benchmarks)
    const totalBenchmarks = allBenchmarks.length

    if (totalBenchmarks === 0) {
      console.log('No benchmarks were run.')
      return
    }

    const fastest = allBenchmarks.reduce((a, b) => (a.opsPerSec > b.opsPerSec ? a : b))
    const slowest = allBenchmarks.reduce((a, b) => (a.opsPerSec < b.opsPerSec ? a : b))

    console.log('Summary')
    console.log('-------')
    console.log(`Total benchmarks: ${totalBenchmarks}`)
    console.log(`Total time: ${(duration / 1000).toFixed(1)}s`)
    console.log(`Fastest: ${fastest.name} @ ${formatOps(fastest.opsPerSec)} ops/sec`)
    console.log(`Slowest: ${slowest.name} @ ${formatOps(slowest.opsPerSec)} ops/sec`)
    console.log('')
  }

  /**
   * Write results to JSON file
   */
  private writeJson(results: BenchmarkResults): void {
    const outputDir = this.config.jsonPath ?? './benchmark/results'

    if (!existsSync(outputDir)) {
      mkdirSync(outputDir, { recursive: true })
    }

    const timestamp = new Date().toISOString().replace(/[:.]/g, '-')
    const filename = `benchmark-${timestamp}.json`
    const filepath = join(outputDir, filename)

    writeFileSync(filepath, JSON.stringify(results, null, 2))
    console.log(`Results written to: ${filepath}`)
  }
}
