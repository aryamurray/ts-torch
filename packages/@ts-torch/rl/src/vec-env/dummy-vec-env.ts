/**
 * DummyVecEnv - Sequential Vectorized Environment
 *
 * Runs multiple environment instances sequentially in the main thread.
 * Simple and easy to debug. Good for smaller numbers of environments.
 *
 * For parallel execution, use SubprocVecEnv (worker threads).
 *
 * @example
 * ```ts
 * import { DummyVecEnv } from './vec-env'
 * import { CartPole } from '../envs'
 *
 * const vecEnv = new DummyVecEnv(() => CartPole(), { nEnvs: 8 })
 *
 * const obs = vecEnv.reset()  // [8 * obsSize]
 * const actions = new Int32Array([0, 1, 0, 1, 0, 1, 0, 1])
 * const { observations, rewards, dones, infos } = vecEnv.step(actions)
 * ```
 */

import { FunctionalEnv } from '../environment.js'
import type { Space } from '../spaces/index.js'
import { discrete } from '../spaces/discrete.js'
import { box } from '../spaces/box.js'
import type { VecEnv, VecEnvStepResult, EnvInfo } from './base.js'

// ==================== Types ====================

/**
 * Configuration for DummyVecEnv
 */
export interface DummyVecEnvConfig {
  /** Number of parallel environments */
  nEnvs: number
  /** Reward dimensionality (default: 1) */
  rewardDim?: number
  /** 
   * Action space override.
   * If not provided, defaults to discrete(actionSpace) from the environment.
   * Use this for continuous action spaces.
   */
  actionSpace?: Space
}

// ==================== Implementation ====================

/**
 * Sequential vectorized environment
 *
 * Runs environments one at a time in the main thread.
 * Auto-resets environments when they reach done state.
 */
export class DummyVecEnv<S> implements VecEnv {
  private readonly envs: FunctionalEnv<S>[]
  private readonly nEnvs_: number
  private readonly obsSize: number
  private readonly actionDim_: number
  private readonly rewardDim_: number
  private readonly observationSpace_: Space
  private readonly actionSpace_: Space

  // Double-buffered observations for zero-copy returns
  private readonly obsBufferA: Float32Array
  private readonly obsBufferB: Float32Array
  private obsBufferCurrent: Float32Array
  private obsBufferPrev: Float32Array
  private readonly rewardBuffer: Float32Array
  private readonly doneBuffer: Uint8Array
  private readonly truncatedBuffer_: Uint8Array

  // Static empty infos for the common case (no episodes ended)
  private readonly emptyInfos: readonly EnvInfo[]

  // Episode tracking
  private readonly episodeLengths: number[]
  private readonly episodeRewards: number[]

  /**
   * Create a DummyVecEnv
   *
   * @param envFactory - Factory function that creates environment instances
   * @param config - Configuration options
   */
  constructor(
    envFactory: () => FunctionalEnv<S>,
    config: DummyVecEnvConfig,
  ) {
    this.nEnvs_ = config.nEnvs
    this.rewardDim_ = config.rewardDim ?? 1

    // Create environment instances using factory
    this.envs = []
    for (let i = 0; i < this.nEnvs_; i++) {
      this.envs.push(envFactory())
    }

    // Get sizes from first environment
    const firstEnv = this.envs[0]!
    this.obsSize = firstEnv.observationSize

    // Create observation space
    this.observationSpace_ = box({
      low: Array.from({ length: this.obsSize }, () => -Infinity),
      high: Array.from({ length: this.obsSize }, () => Infinity),
      shape: [this.obsSize],
    })

    // Set action space (use config override if provided, otherwise default to discrete)
    if (config.actionSpace) {
      this.actionSpace_ = config.actionSpace
      this.actionDim_ = config.actionSpace.type === 'discrete' 
        ? config.actionSpace.n 
        : config.actionSpace.shape.reduce((a, b) => a * b, 1)
    } else {
      // Default to discrete action space from environment
      this.actionDim_ = firstEnv.actionSpace ?? 1
      this.actionSpace_ = discrete(this.actionDim_)
    }

    // Double-buffered observations: swap each step for zero-copy returns
    this.obsBufferA = new Float32Array(this.nEnvs_ * this.obsSize)
    this.obsBufferB = new Float32Array(this.nEnvs_ * this.obsSize)
    this.obsBufferCurrent = this.obsBufferA
    this.obsBufferPrev = this.obsBufferB
    this.rewardBuffer = new Float32Array(this.nEnvs_ * this.rewardDim_)
    this.doneBuffer = new Uint8Array(this.nEnvs_)
    this.truncatedBuffer_ = new Uint8Array(this.nEnvs_)

    // Static empty infos array (reused when no episodes terminate)
    const emptyInfos: EnvInfo[] = []
    for (let i = 0; i < this.nEnvs_; i++) emptyInfos.push({})
    this.emptyInfos = Object.freeze(emptyInfos)

    // Episode tracking
    this.episodeLengths = Array.from({ length: this.nEnvs_ }, () => 0)
    this.episodeRewards = Array.from({ length: this.nEnvs_ }, () => 0)
  }

  /**
   * Number of environments
   */
  get nEnvs(): number {
    return this.nEnvs_
  }

  /**
   * Observation space
   */
  get observationSpace(): Space {
    return this.observationSpace_
  }

  /**
   * Action space
   */
  get actionSpace(): Space {
    return this.actionSpace_
  }

  /**
   * Observation size (flat)
   */
  get observationSize(): number {
    return this.obsSize
  }

  /**
   * Action dimension
   */
  get actionDim(): number {
    return this.actionDim_
  }

  /**
   * Reward dimensionality
   */
  get rewardDim(): number {
    return this.rewardDim_
  }

  /**
   * Reset all environments
   * @returns Initial observations [nEnvs * obsSize]
   */
  reset(): Float32Array {
    for (let i = 0; i < this.nEnvs_; i++) {
      const obs = this.envs[i]!.reset()
      this.obsBufferCurrent.set(obs, i * this.obsSize)
      this.episodeLengths[i] = 0
      this.episodeRewards[i] = 0
    }
    // Return a copy on reset (caller needs stable reference for lastObs)
    return new Float32Array(this.obsBufferCurrent)
  }

  /**
   * Step all environments
   *
   * @param actions - Actions for each env. For discrete spaces: [nEnvs]. 
   *                  For continuous spaces: [nEnvs * actionDim] (flattened).
   * @returns Step results
   */
  step(actions: Int32Array | Float32Array): VecEnvStepResult {
    const isDiscrete = this.actionSpace_.type === 'discrete'

    // Swap double-buffers: write new obs into prev buffer (which becomes current)
    const writeBuffer = this.obsBufferPrev
    this.obsBufferPrev = this.obsBufferCurrent
    this.obsBufferCurrent = writeBuffer

    let hasEpisodeEnd = false

    for (let i = 0; i < this.nEnvs_; i++) {
      const action = isDiscrete
        ? Math.round(actions[i]!)
        : actions[i]!
      const result = this.envs[i]!.step(action)

      // Update tracking
      this.episodeLengths[i]!++
      const stepReward = typeof result.reward === 'number'
        ? result.reward
        : result.reward[0]!
      this.episodeRewards[i]! += stepReward

      // Write observation directly into current buffer
      writeBuffer.set(result.observation, i * this.obsSize)

      // Copy reward
      if (typeof result.reward === 'number') {
        this.rewardBuffer[i * this.rewardDim_] = result.reward
      } else {
        this.rewardBuffer.set(result.reward, i * this.rewardDim_)
      }

      // Set done flag and capture truncation
      if (result.done) {
        this.doneBuffer[i] = 1
        this.truncatedBuffer_[i] = result.truncated ? 1 : 0
        hasEpisodeEnd = true
      } else {
        this.doneBuffer[i] = 0
      }
    }

    // Build infos only if episodes ended (sparse allocation)
    let infos: readonly EnvInfo[]
    if (hasEpisodeEnd) {
      const infoArray: EnvInfo[] = []
      for (let i = 0; i < this.nEnvs_; i++) {
        if (this.doneBuffer[i] === 1) {
          const wasTruncated = this.truncatedBuffer_[i] === 1
          infoArray.push({
            terminal: !wasTruncated,
            truncated: wasTruncated,
            terminalObservation: writeBuffer.slice(i * this.obsSize, (i + 1) * this.obsSize),
            episodeLength: this.episodeLengths[i],
            episodeReward: this.episodeRewards[i],
          })

          // Auto-reset
          const newObs = this.envs[i]!.reset()
          writeBuffer.set(newObs, i * this.obsSize)
          this.episodeLengths[i] = 0
          this.episodeRewards[i] = 0
        } else {
          infoArray.push({})
        }
      }
      infos = infoArray
    } else {
      infos = this.emptyInfos
    }

    // Return buffers directly — rollout buffer copies them in add() immediately
    return {
      observations: writeBuffer,
      rewards: this.rewardBuffer,
      dones: this.doneBuffer,
      infos: infos as EnvInfo[],
    }
  }

  /**
   * Step all environments, writing observations into a caller-provided buffer.
   *
   * Same as step() but writes observations into obsTarget instead of the
   * internal double-buffer. Used for shared-memory mode with RolloutBuffer.
   */
  stepInto(actions: Int32Array | Float32Array, obsTarget: Float32Array): VecEnvStepResult {
    const isDiscrete = this.actionSpace_.type === 'discrete'

    let hasEpisodeEnd = false

    for (let i = 0; i < this.nEnvs_; i++) {
      const action = isDiscrete
        ? Math.round(actions[i]!)
        : actions[i]!
      const result = this.envs[i]!.step(action)

      // Update tracking
      this.episodeLengths[i]!++
      const stepReward = typeof result.reward === 'number'
        ? result.reward
        : result.reward[0]!
      this.episodeRewards[i]! += stepReward

      // Write observation directly into caller-provided target
      obsTarget.set(result.observation, i * this.obsSize)

      // Copy reward
      if (typeof result.reward === 'number') {
        this.rewardBuffer[i * this.rewardDim_] = result.reward
      } else {
        this.rewardBuffer.set(result.reward, i * this.rewardDim_)
      }

      // Set done flag and capture truncation
      if (result.done) {
        this.doneBuffer[i] = 1
        this.truncatedBuffer_[i] = result.truncated ? 1 : 0
        hasEpisodeEnd = true
      } else {
        this.doneBuffer[i] = 0
      }
    }

    // Build infos only if episodes ended (sparse allocation)
    let infos: readonly EnvInfo[]
    if (hasEpisodeEnd) {
      const infoArray: EnvInfo[] = []
      for (let i = 0; i < this.nEnvs_; i++) {
        if (this.doneBuffer[i] === 1) {
          const wasTruncated = this.truncatedBuffer_[i] === 1
          infoArray.push({
            terminal: !wasTruncated,
            truncated: wasTruncated,
            terminalObservation: obsTarget.slice(i * this.obsSize, (i + 1) * this.obsSize),
            episodeLength: this.episodeLengths[i],
            episodeReward: this.episodeRewards[i],
          })

          // Auto-reset
          const newObs = this.envs[i]!.reset()
          obsTarget.set(newObs, i * this.obsSize)
          this.episodeLengths[i] = 0
          this.episodeRewards[i] = 0
        } else {
          infoArray.push({})
        }
      }
      infos = infoArray
    } else {
      infos = this.emptyInfos
    }

    return {
      observations: obsTarget,
      rewards: this.rewardBuffer,
      dones: this.doneBuffer,
      infos: infos as EnvInfo[],
    }
  }

  /**
   * Get current observations without stepping
   */
  getObservations(): Float32Array {
    for (let i = 0; i < this.nEnvs_; i++) {
      const obs = this.envs[i]!.observe()
      this.obsBufferCurrent.set(obs, i * this.obsSize)
    }
    return new Float32Array(this.obsBufferCurrent)
  }

  /**
   * Close all environments
   */
  close(): void {
    // No-op for DummyVecEnv (envs are garbage collected)
  }

  /**
   * Get underlying environment instances (for inspection/debugging)
   */
  getEnvs(): FunctionalEnv<S>[] {
    return this.envs
  }
}

// ==================== Factory ====================

/**
 * Create a DummyVecEnv (legacy API)
 *
 * @deprecated Use RL.vecEnv({ env, nEnvs }) instead
 * @param envFactory - Factory function that creates a new environment
 * @param config - Configuration options
 * @returns DummyVecEnv instance
 *
 * @example
 * ```ts
 * // Legacy API
 * const vecEnv = dummyVecEnv(() => CartPole(), { nEnvs: 8 })
 * 
 * // New declarative API (preferred)
 * const vecEnv = RL.vecEnv({ env: RL.envs.CartPole(), nEnvs: 8 })
 * ```
 */
export function dummyVecEnv<S>(
  envFactory: () => FunctionalEnv<S>,
  config: DummyVecEnvConfig,
): DummyVecEnv<S> {
  return new DummyVecEnv(envFactory, config)
}
