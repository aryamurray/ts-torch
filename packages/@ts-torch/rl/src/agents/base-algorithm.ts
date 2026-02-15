/**
 * Base Algorithm Class
 *
 * Abstract base class for all RL algorithms.
 * Provides common functionality for training, saving, loading, and inference.
 *
 * Subclasses implement:
 * - _setupModel(): Initialize networks, optimizers, buffers
 * - _train(): Perform gradient updates
 * - collectRollouts() or equivalent
 */

import type { DeviceType } from '@ts-torch/core'
import type { DeviceContext } from '@ts-torch/core'
import { Logger, verboseToLevel } from '@ts-torch/core'
import type { VecEnv } from '../vec-env/index.js'
import type {
  BaseCallback,
  MetricsLogger,
  Callbacks,
  TrainingStartData,
  TrainingEndData,
  RolloutStartData,
  RolloutEndData,
} from '../callbacks/index.js'
import { maybeCallback } from '../callbacks/index.js'

// ==================== Types ====================

/**
 * Learning rate can be a constant or a schedule function
 * Schedule receives progress_remaining (1.0 -> 0.0) and returns lr
 */
export type Schedule = number | ((progressRemaining: number) => number)

/**
 * Configuration for learn()
 *
 * @example
 * ```ts
 * await agent.learn({
 *   totalTimesteps: 100_000,
 *
 *   // Declarative callbacks (recommended)
 *   callbacks: {
 *     onEpisodeEnd: ({ episodeReward }) => {
 *       console.log(`Episode reward: ${episodeReward}`)
 *       if (episodeReward > 500) return false  // early stop
 *     },
 *     onTrainingEnd: ({ finalReward }) => {
 *       console.log(`Final reward: ${finalReward}`)
 *     }
 *   },
 *
 *   // Built-in behaviors
 *   checkpointFreq: 50_000,
 *   checkpointPath: './models/ppo',
 *   logInterval: 1000
 * })
 * ```
 */
export interface LearnConfig {
  /** Total number of timesteps to train */
  totalTimesteps: number

  // ===== Callbacks =====
  /**
   * Declarative callbacks (recommended).
   * Simple inline functions for common hooks.
   */
  callbacks?: Callbacks

  /**
   * Class-based callback (legacy/advanced).
   * Use for complex callback logic or callback composition.
   * @deprecated Prefer `callbacks` for most use cases
   */
  callback?: BaseCallback | BaseCallback[]

  // ===== Built-in Behaviors =====
  /** Reset timestep counter (default: true) */
  resetNumTimesteps?: boolean

  /** Log interval in timesteps (0 = no logging, default: 0) */
  logInterval?: number

  /** Show progress bar (default: false) */
  progressBar?: boolean

  /** Save checkpoint every N steps (0 = disabled) */
  checkpointFreq?: number

  /** Path to save checkpoints (required if checkpointFreq > 0) */
  checkpointPath?: string

  /** Evaluate every N steps (0 = disabled) */
  evalFreq?: number

  /** Separate environment for evaluation */
  evalEnv?: VecEnv

  /** Number of episodes per evaluation (default: 5) */
  evalEpisodes?: number

  /** Enable real-time TUI dashboard (requires @ts-torch/dashboard) */
  dashboard?: boolean
}

/**
 * Configuration for base algorithm
 */
export interface BaseAlgorithmConfig {
  /** Vectorized environment */
  env: VecEnv
  /** Device to run on */
  device: DeviceContext<DeviceType>
  /** Learning rate (constant or schedule) */
  learningRate: Schedule
  /** Verbosity level (0 = no output, 1 = info, 2 = debug) */
  verbose?: number
}

// ==================== Implementation ====================

/**
 * Abstract base class for RL algorithms
 */
export abstract class BaseAlgorithm {
  /** Vectorized environment */
  protected env: VecEnv

  /** Device context */
  protected device_: DeviceContext<DeviceType>

  /** Learning rate schedule */
  protected learningRate: Schedule

  /** Current number of timesteps */
  numTimesteps: number = 0

  /** Number of timesteps in current training run */
  protected numTimestepsAtStart: number = 0

  /** Total timesteps for current training run */
  protected totalTimesteps: number = 0

  /** Current iteration (rollout count) */
  protected iteration: number = 0

  /** Verbosity level (0=warn, 1=info, 2=debug) */
  protected verbose: number

  /** Metrics logger for recording training data */
  metricsLogger: MetricsLogger | null = null

  /** Current callback (legacy class-based) */
  protected callback: BaseCallback | null = null

  /** Declarative callbacks */
  protected callbacks: Callbacks | null = null

  /** Whether the model has been set up */
  protected isSetup: boolean = false

  /** Training start time for metrics */
  protected trainingStartTime: number = 0

  /** Episode tracking for callbacks */
  protected episodesCompleted: number = 0
  protected currentEpisodeRewards: Float32Array | null = null
  protected currentEpisodeLengths: Uint32Array | null = null

  constructor(config: BaseAlgorithmConfig) {
    this.env = config.env
    this.device_ = config.device
    this.learningRate = config.learningRate
    this.verbose = config.verbose ?? 0

    // Set Logger level based on verbose setting
    if (this.verbose > 0) {
      Logger.setLevel(verboseToLevel(this.verbose))
    }
  }

  /**
   * Get number of environments
   */
  get nEnvs(): number {
    return this.env.nEnvs
  }

  /**
   * Get the device
   */
  get device(): DeviceContext<DeviceType> {
    return this.device_
  }

  /**
   * Get current learning rate value
   */
  protected getCurrentLr(): number {
    if (typeof this.learningRate === 'number') {
      return this.learningRate
    }
    const progressRemaining = 1.0 - this.numTimesteps / this.totalTimesteps
    return this.learningRate(progressRemaining)
  }

  /**
   * Update learning rate based on schedule
   */
  protected updateLearningRate(): void {
    // Subclasses override to update optimizer LR
  }

  /**
   * Setup the model (networks, optimizers, buffers)
   * Called once before training starts
   */
  protected abstract _setupModel(): void

  /**
   * Collect experience (rollouts for on-policy, steps for off-policy)
   * @returns True if rollout was successful
   */
  protected abstract collectRollouts(): boolean

  /**
   * Perform gradient updates
   */
  protected abstract _train(): void

  /**
   * Main training loop
   *
   * @param config - Training configuration
   * @returns This algorithm instance for chaining
   */
  async learn(config: LearnConfig): Promise<this> {
    const { totalTimesteps, callback, resetNumTimesteps = true, logInterval = 0 } = config

    // Dashboard: dynamically create RL dashboard callbacks if requested
    let callbacks = config.callbacks
    if (config.dashboard && !callbacks) {
      try {
        const { createRLDashboardCallback } = await import('../dashboard-callback.js')
        callbacks = await createRLDashboardCallback()
      } catch (e) {
        Logger.warn(`Dashboard requested but failed to initialize: ${e instanceof Error ? e.message : e}`)
      }
    }

    // Setup if not done
    if (!this.isSetup) {
      this._setupModel()
      this.isSetup = true
    }

    // Reset timesteps if requested
    if (resetNumTimesteps) {
      this.numTimesteps = 0
      this.iteration = 0
      this.episodesCompleted = 0
    }

    this.numTimestepsAtStart = this.numTimesteps
    this.totalTimesteps = totalTimesteps
    this.trainingStartTime = Date.now()

    // Setup declarative callbacks
    this.callbacks = callbacks ?? null

    // Setup legacy class-based callback
    this.callback = maybeCallback(callback)
    if (this.callback) {
      this.callback.initCallback(this)
    }

    // Initialize episode tracking
    this.currentEpisodeRewards = new Float32Array(this.nEnvs)
    this.currentEpisodeLengths = new Uint32Array(this.nEnvs)

    // Training start callbacks
    const trainingStartData: TrainingStartData = {
      totalTimesteps,
      nEnvs: this.nEnvs,
      algorithm: this.constructor.name,
    }
    this.callbacks?.onTrainingStart?.(trainingStartData)
    this.callback?.onTrainingStart({})

    // Main training loop
    let shouldStop = false
    while (this.numTimesteps < totalTimesteps && !shouldStop) {
      // Rollout start callbacks
      const rolloutStartData: RolloutStartData = {
        timestep: this.numTimesteps,
        iteration: this.iteration,
      }
      this.callbacks?.onRolloutStart?.(rolloutStartData)
      this.callback?.onRolloutStart()

      // Collect rollout
      const continueTraining = this.collectRollouts()

      if (!continueTraining) {
        break
      }

      // Legacy step callback
      if (this.callback && !this.callback.onStep()) {
        break
      }

      this.iteration++

      // Rollout end callbacks
      const rolloutEndData: RolloutEndData = {
        timestep: this.numTimesteps,
        rolloutReward: 0, // TODO: compute from episode tracker
        rolloutLength: this.nEnvs * (this.iteration > 0 ? 1 : 0), // Simplified
        episodesCompleted: this.episodesCompleted,
      }
      this.callbacks?.onRolloutEnd?.(rolloutEndData)
      this.callback?.onRolloutEnd()

      // Update learning rate
      this.updateLearningRate()

      // Perform training
      this._train()

      // Logging
      if (logInterval > 0 && this.iteration % logInterval === 0) {
        this.logProgress()
      }
    }

    // Training end callbacks
    const trainingEndData: TrainingEndData = {
      totalTimesteps: this.numTimesteps,
      totalEpisodes: this.episodesCompleted,
      totalTime: Date.now() - this.trainingStartTime,
      finalReward: 0, // TODO: compute from episode tracker
    }
    this.callbacks?.onTrainingEnd?.(trainingEndData)
    this.callback?.onTrainingEnd()

    return this
  }

  /**
   * Log training progress
   */
  protected logProgress(): void {
    const progress = (this.numTimesteps / this.totalTimesteps) * 100
    Logger.info(`Timesteps: ${this.numTimesteps}/${this.totalTimesteps} (${progress.toFixed(1)}%)`)
  }

  /**
   * Get action from observation (inference)
   *
   * @param observation - Single observation
   * @param deterministic - Use deterministic action selection
   * @returns Action(s)
   */
  abstract predict(observation: Float32Array, deterministic?: boolean): number | Float32Array

  /**
   * Save the algorithm to a file
   */
  abstract save(path: string): Promise<void>

  /**
   * Get all trainable parameters
   */
  abstract parameters(): any[]

  /**
   * Set to training mode
   */
  abstract train(): void

  /**
   * Set to evaluation mode
   */
  abstract eval(): void
}
