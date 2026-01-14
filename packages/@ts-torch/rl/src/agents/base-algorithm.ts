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
import type { VecEnv } from '../vec-env/index.js'
import type { BaseCallback, Logger } from '../callbacks/index.js'
import { maybeCallback } from '../callbacks/index.js'

// ==================== Types ====================

/**
 * Learning rate can be a constant or a schedule function
 * Schedule receives progress_remaining (1.0 -> 0.0) and returns lr
 */
export type Schedule = number | ((progressRemaining: number) => number)

/**
 * Configuration for learn()
 */
export interface LearnConfig {
  /** Total number of timesteps to train */
  totalTimesteps: number
  /** Callback or array of callbacks */
  callback?: BaseCallback | BaseCallback[]
  /** Reset timestep counter (default: true) */
  resetNumTimesteps?: boolean
  /** Log interval in timesteps (0 = no logging) */
  logInterval?: number
  /** Show progress bar (default: false) */
  progressBar?: boolean
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

  /** Verbosity level */
  protected verbose: number

  /** Logger for metrics */
  logger: Logger | null = null

  /** Current callback */
  protected callback: BaseCallback | null = null

  /** Whether the model has been set up */
  protected isSetup: boolean = false

  constructor(config: BaseAlgorithmConfig) {
    this.env = config.env
    this.device_ = config.device
    this.learningRate = config.learningRate
    this.verbose = config.verbose ?? 0
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
    const progressRemaining = 1.0 - (this.numTimesteps / this.totalTimesteps)
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
    const {
      totalTimesteps,
      callback,
      resetNumTimesteps = true,
      logInterval = 0,
    } = config

    // Setup if not done
    if (!this.isSetup) {
      this._setupModel()
      this.isSetup = true
    }

    // Reset timesteps if requested
    if (resetNumTimesteps) {
      this.numTimesteps = 0
      this.iteration = 0
    }

    this.numTimestepsAtStart = this.numTimesteps
    this.totalTimesteps = totalTimesteps

    // Setup callback
    this.callback = maybeCallback(callback)
    if (this.callback) {
      this.callback.initCallback(this)
    }

    // Training start callback
    this.callback?.onTrainingStart({})

    // Main training loop
    while (this.numTimesteps < totalTimesteps) {
      // Collect rollout
      this.callback?.onRolloutStart()
      const continueTraining = this.collectRollouts()

      if (!continueTraining) {
        break
      }

      // Step callback
      if (this.callback && !this.callback.onStep()) {
        break
      }

      this.iteration++
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

    // Training end callback
    this.callback?.onTrainingEnd()

    return this
  }

  /**
   * Log training progress
   */
  protected logProgress(): void {
    const progress = (this.numTimesteps / this.totalTimesteps) * 100
    if (this.verbose > 0) {
      console.log(
        `Timesteps: ${this.numTimesteps}/${this.totalTimesteps} (${progress.toFixed(1)}%)`,
      )
    }
  }

  /**
   * Get action from observation (inference)
   *
   * @param observation - Single observation
   * @param deterministic - Use deterministic action selection
   * @returns Action(s)
   */
  abstract predict(
    observation: Float32Array,
    deterministic?: boolean,
  ): number | Float32Array

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
