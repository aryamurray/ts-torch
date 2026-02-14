/**
 * Global configuration for ts-torch runtime.
 *
 * @module config
 */

import type { NativeModule } from './ffi/loader.js'

/**
 * Configuration options for the ts-torch runtime.
 *
 * @example
 * ```ts
 * import { torch } from '@ts-torch/core'
 *
 * torch.config({ numThreads: 1, seed: 42 })
 * ```
 */
export interface TorchConfig {
  /**
   * Number of threads for intra-op parallelism (LibTorch, OpenMP, MKL, OpenBLAS).
   *
   * Controls how many CPU threads are used for individual tensor operations
   * like matrix multiplication. Does not affect JavaScript's single-threaded
   * event loop — only native C++ operations.
   *
   * - `1` (default): Single-threaded. Best for small tensors, RL workloads,
   *   and avoiding thread over-subscription with Node.js.
   * - `0`: Auto-detect (uses all available CPU cores). Use for large batch
   *   training where BLAS parallelism helps.
   * - `N`: Use exactly N threads.
   *
   * @default 1
   */
  numThreads?: number

  /**
   * Global random seed for reproducible tensor operations (randn, rand, dropout, etc.).
   *
   * Sets the seed for LibTorch's default CPU random number generator.
   * When set, operations like `randn()` and `rand()` produce deterministic results.
   *
   * Leave undefined for non-deterministic (random) behavior.
   *
   * @default undefined
   *
   * @example
   * ```ts
   * torch.config({ seed: 42 })
   * const a = device.cpu().randn([3, 3])  // always the same
   * ```
   */
  seed?: number
}

/** Resolved config state — numThreads is always set, seed may be undefined */
interface ResolvedConfig {
  numThreads: number
  seed: number | undefined
}

const DEFAULT_CONFIG: ResolvedConfig = {
  numThreads: 1,
  seed: undefined,
}

/** Internal mutable config state */
let _config: ResolvedConfig = { ...DEFAULT_CONFIG }

/** Whether the native lib has been initialized with our config */
let _applied = false

/**
 * Apply current config to a loaded native module.
 * Called by `getLib()` on first load, and by `config()` if the lib is already loaded.
 *
 * @internal
 */
export function applyConfig(lib: NativeModule): void {
  lib.ts_set_num_threads(_config.numThreads)
  if (_config.seed !== undefined) {
    lib.ts_manual_seed(_config.seed)
  }
  _applied = true
}

/**
 * Update a single config key from outside (e.g. setNumThreads compat shim).
 *
 * @internal
 */
export function updateConfigKey<K extends keyof TorchConfig>(key: K, value: TorchConfig[K]): void {
  ;(_config as any)[key] = value
}

/**
 * Set global configuration options, or read the current config.
 *
 * @param opts - Configuration values to merge. Omit to read current config.
 * @returns A frozen snapshot of the current config when called with no arguments.
 *
 * @example
 * ```ts
 * // Set config (can be called multiple times, last write wins)
 * torch.config({ numThreads: 1, seed: 42 })
 *
 * // Read current config
 * const cfg = torch.config()
 * console.log(cfg.numThreads) // 1
 * ```
 */
function config(opts: TorchConfig): void
function config(): Readonly<ResolvedConfig>
function config(
  opts?: TorchConfig,
): void | Readonly<ResolvedConfig> {
  if (opts === undefined) {
    return Object.freeze({ ..._config })
  }

  if (opts.numThreads !== undefined) {
    _config.numThreads = opts.numThreads
  }
  if (opts.seed !== undefined) {
    _config.seed = opts.seed
  }

  // If native lib is already loaded, apply immediately
  if (_applied) {
    // Dynamic import avoided — use getLib which is cached
    const { getLib } = require('./ffi/loader.js') as { getLib: () => NativeModule }
    const lib = getLib()
    applyConfig(lib)
  }
}

/**
 * The `torch` namespace provides global configuration for ts-torch.
 *
 * @example
 * ```ts
 * import { torch } from '@ts-torch/core'
 *
 * // Configure before any tensor ops
 * torch.config({ numThreads: 1, seed: 42 })
 *
 * // Read current config
 * const cfg = torch.config()
 * ```
 */
export const torch = {
  config,
} as {
  config(opts: TorchConfig): void
  config(): Readonly<ResolvedConfig>
  config(
    opts?: TorchConfig,
  ): void | Readonly<ResolvedConfig>
}
