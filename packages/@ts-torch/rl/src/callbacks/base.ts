/**
 * Training Callbacks
 *
 * Callbacks allow you to customize training behavior without modifying algorithm code.
 * They hook into various stages of the training loop.
 *
 * @example
 * ```ts
 * class MyCallback extends BaseCallback {
 *   onStep(): boolean {
 *     if (this.numTimesteps % 1000 === 0) {
 *       console.log(`Step ${this.numTimesteps}`)
 *     }
 *     return true  // Continue training
 *   }
 * }
 *
 * await ppo.learn({
 *   totalTimesteps: 100_000,
 *   callback: new MyCallback(),
 * })
 * ```
 */

// ==================== Types ====================

/**
 * Local variables passed to callbacks
 */
export interface CallbackLocals {
  /** Current observations */
  observations?: Float32Array
  /** Current actions */
  actions?: Float32Array
  /** Current rewards */
  rewards?: Float32Array
  /** Current done flags */
  dones?: Uint8Array
  /** Current values (on-policy) */
  values?: Float32Array
  /** Current log probabilities (on-policy) */
  logProbs?: Float32Array
  /** Any additional data */
  [key: string]: unknown
}

/**
 * Base interface for training algorithms (needed by callbacks)
 */
export interface BaseAlgorithmRef {
  /** Current number of timesteps */
  numTimesteps: number
  /** Number of environments */
  nEnvs: number
  /** Metrics logger for recording training data */
  metricsLogger?: MetricsLogger | null
}

/**
 * Metrics logger interface for recording training metrics
 * Note: This is different from the unified Logger in @ts-torch/core
 */
export interface MetricsLogger {
  record(key: string, value: number | string): void
  dump(step: number): void
}

// ==================== Implementation ====================

/**
 * Base class for training callbacks
 *
 * Override the lifecycle methods to customize behavior.
 * Return false from onStep() to stop training.
 */
export abstract class BaseCallback {
  /** Number of times callback was called */
  nCalls: number = 0

  /** Number of timesteps so far */
  numTimesteps: number = 0

  /** Reference to the algorithm (set during init) */
  model: BaseAlgorithmRef | null = null

  /** Parent callback (for nested callbacks) */
  parent: BaseCallback | null = null

  /** Metrics logger from the algorithm */
  logger: MetricsLogger | null = null

  /** Verbosity level */
  verbose: number = 0

  constructor(verbose: number = 0) {
    this.verbose = verbose
  }

  /**
   * Initialize the callback with algorithm reference
   * Called once at the start of training
   */
  initCallback(model: BaseAlgorithmRef): void {
    this.model = model
    this.logger = model.metricsLogger ?? null
    this._initCallback()
  }

  /**
   * Override for custom initialization
   */
  protected _initCallback(): void {}

  /**
   * Called at the start of training
   */
  onTrainingStart(locals: CallbackLocals): void {
    this._onTrainingStart(locals)
  }

  /**
   * Override for custom training start behavior
   */
  protected _onTrainingStart(_locals: CallbackLocals): void {}

  /**
   * Called at the start of a rollout
   */
  onRolloutStart(): void {
    this._onRolloutStart()
  }

  /**
   * Override for custom rollout start behavior
   */
  protected _onRolloutStart(): void {}

  /**
   * Called after each step
   * @returns False to stop training, true to continue
   */
  onStep(): boolean {
    this.nCalls++
    this.numTimesteps = this.model?.numTimesteps ?? 0
    return this._onStep()
  }

  /**
   * Override for custom step behavior
   * @returns False to stop training
   */
  protected _onStep(): boolean {
    return true
  }

  /**
   * Called at the end of a rollout
   */
  onRolloutEnd(): void {
    this._onRolloutEnd()
  }

  /**
   * Override for custom rollout end behavior
   */
  protected _onRolloutEnd(): void {}

  /**
   * Called at the end of training
   */
  onTrainingEnd(): void {
    this._onTrainingEnd()
  }

  /**
   * Override for custom training end behavior
   */
  protected _onTrainingEnd(): void {}

  /**
   * Update child callback references
   */
  updateLocals(_locals: CallbackLocals): void {
    // Can be overridden to pass data to child callbacks
  }
}

/**
 * Callback that wraps a list of callbacks
 * All callbacks are called in order
 */
export class CallbackList extends BaseCallback {
  callbacks: BaseCallback[]

  constructor(callbacks: BaseCallback[]) {
    super()
    this.callbacks = callbacks
  }

  protected _initCallback(): void {
    for (const callback of this.callbacks) {
      callback.parent = this
      callback.initCallback(this.model!)
    }
  }

  protected _onTrainingStart(_locals: CallbackLocals): void {
    for (const callback of this.callbacks) {
      callback.onTrainingStart(_locals)
    }
  }

  protected _onRolloutStart(): void {
    for (const callback of this.callbacks) {
      callback.onRolloutStart()
    }
  }

  protected _onStep(): boolean {
    for (const callback of this.callbacks) {
      if (!callback.onStep()) {
        return false
      }
    }
    return true
  }

  protected _onRolloutEnd(): void {
    for (const callback of this.callbacks) {
      callback.onRolloutEnd()
    }
  }

  protected _onTrainingEnd(): void {
    for (const callback of this.callbacks) {
      callback.onTrainingEnd()
    }
  }
}

/**
 * Simple callback that stops training when a condition is met
 */
export class StopTrainingCallback extends BaseCallback {
  private shouldStop: boolean = false

  constructor() {
    super()
  }

  /**
   * Call this to stop training
   */
  stop(): void {
    this.shouldStop = true
  }

  protected _onStep(): boolean {
    return !this.shouldStop
  }
}

/**
 * Callback that tracks episode rewards and lengths
 */
export class EpisodeTrackingCallback extends BaseCallback {
  /** Episode rewards for completed episodes */
  episodeRewards: number[] = []
  /** Episode lengths for completed episodes */
  episodeLengths: number[] = []
  /** Number of completed episodes */
  numEpisodes: number = 0

  protected _onStep(): boolean {
    // Episode tracking would be handled by extracting info from env
    // This is a placeholder - actual implementation depends on how
    // episode info is passed through
    return true
  }

  /**
   * Record a completed episode
   */
  recordEpisode(reward: number, length: number): void {
    this.episodeRewards.push(reward)
    this.episodeLengths.push(length)
    this.numEpisodes++
  }

  /**
   * Get mean reward over last n episodes
   */
  meanReward(n: number = 100): number {
    if (this.episodeRewards.length === 0) return 0
    const recentRewards = this.episodeRewards.slice(-n)
    return recentRewards.reduce((a, b) => a + b, 0) / recentRewards.length
  }

  /**
   * Get mean episode length over last n episodes
   */
  meanLength(n: number = 100): number {
    if (this.episodeLengths.length === 0) return 0
    const recentLengths = this.episodeLengths.slice(-n)
    return recentLengths.reduce((a, b) => a + b, 0) / recentLengths.length
  }
}

// ==================== Factory ====================

/**
 * Create a callback list from an array of callbacks
 */
export function callbackList(callbacks: BaseCallback[]): CallbackList {
  return new CallbackList(callbacks)
}

/**
 * Convert a callback or array of callbacks to a single callback
 */
export function maybeCallback(
  callback: BaseCallback | BaseCallback[] | undefined | null,
): BaseCallback | null {
  if (!callback) return null
  if (Array.isArray(callback)) {
    return callback.length > 0 ? new CallbackList(callback) : null
  }
  return callback
}
