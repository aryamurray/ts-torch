/**
 * Benchmark runner
 * Discovers and executes benchmark suites
 */

import { join } from 'node:path'
import { readdirSync, statSync } from 'node:fs'
import type { BenchmarkConfig, BenchmarkResults, BenchmarkSuite, SuiteResult, PlatformInfo } from './types.js'
import { extractTaskResult } from './types.js'
import { Reporter } from './reporter.js'

/**
 * Benchmark runner
 */
export class BenchmarkRunner {
  private suites: BenchmarkSuite[] = []
  private reporter: Reporter

  constructor(private config: BenchmarkConfig = {}) {
    this.reporter = new Reporter(config.output)
  }

  /**
   * Discover benchmark files in directory
   */
  async discover(baseDir: string): Promise<void> {
    const categories = ['core', 'nn', 'optim']

    for (const category of categories) {
      const categoryDir = join(baseDir, category)

      try {
        const stat = statSync(categoryDir)
        if (!stat.isDirectory()) continue
      } catch {
        continue
      }

      const files = readdirSync(categoryDir).filter((f) => f.endsWith('.bench.ts'))

      for (const file of files) {
        const filepath = join(categoryDir, file)
        try {
          const module = await import(filepath)
          if (module.suite) {
            this.suites.push(module.suite)
          }
        } catch (err) {
          console.error(`Failed to load ${filepath}:`, err)
        }
      }
    }
  }

  /**
   * Add a suite directly
   */
  addSuite(suite: BenchmarkSuite): void {
    this.suites.push(suite)
  }

  /**
   * Check if a suite should run based on filters
   */
  private shouldRun(suite: BenchmarkSuite): boolean {
    // Category filter
    if (this.config.category && suite.category !== this.config.category) {
      return false
    }

    // Name filter
    if (this.config.filter) {
      const pattern = this.config.filter.toLowerCase()
      if (!suite.name.toLowerCase().includes(pattern)) {
        return false
      }
    }

    return true
  }

  /**
   * Run all discovered benchmarks
   */
  async run(): Promise<BenchmarkResults> {
    const startTime = Date.now()
    const suitesToRun = this.suites.filter((s) => this.shouldRun(s))

    this.reporter.start(suitesToRun.length)

    const results: BenchmarkResults = {
      timestamp: new Date().toISOString(),
      platform: this.getPlatformInfo(),
      suites: [],
      totalDuration: 0,
    }

    for (const suite of suitesToRun) {
      this.reporter.suiteStart(suite.name, suite.category)

      const suiteStart = Date.now()
      const bench = await suite.run(this.config)
      const suiteDuration = Date.now() - suiteStart

      const suiteResult: SuiteResult = {
        name: suite.name,
        category: suite.category,
        benchmarks: bench.tasks
          .map((t) => extractTaskResult(t))
          .filter((r): r is NonNullable<typeof r> => r !== null),
        duration: suiteDuration,
      }

      results.suites.push(suiteResult)
      this.reporter.suiteEnd(suiteResult)
    }

    results.totalDuration = Date.now() - startTime
    this.reporter.finish(results)

    return results
  }

  /**
   * Get platform information
   */
  private getPlatformInfo(): PlatformInfo {
    return {
      os: process.platform,
      arch: process.arch,
      nodeVersion: process.version,
      bunVersion: typeof Bun !== 'undefined' ? Bun.version : undefined,
    }
  }
}
