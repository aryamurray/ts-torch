/**
 * Rollout Buffer for On-Policy Algorithms
 *
 * Stores trajectories collected during rollouts for PPO/A2C training.
 * Computes returns and advantages using Generalized Advantage Estimation (GAE).
 * Uses native fused GAE when available for reduced overhead.
 *
 * Key Design:
 * - Pre-allocated TypedArrays for zero-allocation rollouts
 * - GAE computation for variance-reduced advantage estimates
 * - Generator-based minibatch iteration
 *
 * @example
 * ```ts
 * const buffer = new RolloutBuffer({
 *   bufferSize: 2048,
 *   nEnvs: 8,
 *   observationSize: 4,
 *   actionDim: 1,
 *   gamma: 0.99,
 *   gaeLambda: 0.95,
 * })
 *
 * // During rollout
 * buffer.add(obs, actions, rewards, dones, values, logProbs)
 *
 * // After rollout complete
 * buffer.computeReturnsAndAdvantage(lastValues, lastDones)
 *
 * // Training
 * for (const batch of buffer.get(64)) {
 *   // Use batch.observations, batch.actions, etc.
 * }
 * ```
 */

import { getLib } from '@ts-torch/core'

type NativeGaeFn = (
  rewards: Float32Array, values: Float32Array, episodeStarts: Uint8Array,
  lastValues: Float32Array, lastDones: Uint8Array,
  bufferSize: number, nEnvs: number, gamma: number, gaeLambda: number,
  advantagesOut: Float32Array, returnsOut: Float32Array,
) => void

// Lazily resolved native GAE function
let nativeGae: NativeGaeFn | null | undefined

function getNativeGae(): NativeGaeFn | null {
  if (nativeGae !== undefined) return nativeGae
  try {
    const lib = getLib()
    if (typeof lib.ts_compute_gae === 'function') {
      nativeGae = lib.ts_compute_gae as unknown as NativeGaeFn
    } else {
      nativeGae = null
    }
  } catch {
    nativeGae = null
  }
  return nativeGae
}

// ==================== Types ====================

/**
 * Configuration for RolloutBuffer
 */
export interface RolloutBufferConfig {
  /** Number of steps per environment per rollout */
  bufferSize: number
  /** Number of parallel environments */
  nEnvs: number
  /** Observation vector size */
  observationSize: number
  /** Action dimension (1 for discrete, >1 for continuous) */
  actionDim: number
  /** Discount factor */
  gamma: number
  /** GAE lambda parameter */
  gaeLambda: number
}

/**
 * A batch of samples from the rollout buffer
 */
export interface RolloutBufferSamples {
  /** Observations [batchSize, obsSize] as flat array */
  observations: Float32Array
  /** Actions [batchSize] or [batchSize, actionDim] as flat array */
  actions: Float32Array
  /** Old value estimates [batchSize] */
  oldValues: Float32Array
  /** Old log probabilities [batchSize] */
  oldLogProbs: Float32Array
  /** Advantage estimates [batchSize] */
  advantages: Float32Array
  /** Return targets [batchSize] */
  returns: Float32Array
  /** Batch size */
  batchSize: number
}

// ==================== Implementation ====================

/**
 * Rollout buffer for on-policy algorithms
 *
 * Stores complete trajectories and computes GAE advantages.
 */
export class RolloutBuffer {
  // Configuration
  private readonly bufferSize: number
  private readonly nEnvs: number
  private readonly observationSize: number
  private readonly actionDim: number
  private readonly gamma: number
  private readonly gaeLambda: number

  // Pre-allocated storage (shape: [bufferSize * nEnvs, ...])
  private readonly observations: Float32Array
  private readonly actions: Float32Array
  private readonly rewards: Float32Array
  private readonly episodeStarts: Uint8Array  // 1 if this step starts a new episode
  private readonly values: Float32Array
  private readonly logProbs: Float32Array

  // Computed during finalization
  private readonly advantages: Float32Array
  private readonly returns: Float32Array

  // Reusable batch buffers (lazily allocated on first get() call)
  private batchObs_: Float32Array | null = null
  private batchActions_: Float32Array | null = null
  private batchOldValues_: Float32Array | null = null
  private batchOldLogProbs_: Float32Array | null = null
  private batchAdvantages_: Float32Array | null = null
  private batchReturns_: Float32Array | null = null
  private lastBatchSize_: number = 0

  // State
  private position: number = 0
  private full: boolean = false
  private generatorReady: boolean = false

  /**
   * Create a new RolloutBuffer
   *
   * @param config - Buffer configuration
   */
  constructor(config: RolloutBufferConfig) {
    this.bufferSize = config.bufferSize
    this.nEnvs = config.nEnvs
    this.observationSize = config.observationSize
    this.actionDim = config.actionDim
    this.gamma = config.gamma
    this.gaeLambda = config.gaeLambda

    const totalSize = this.bufferSize * this.nEnvs

    // Allocate storage
    this.observations = new Float32Array(totalSize * this.observationSize)
    this.actions = new Float32Array(totalSize * this.actionDim)
    this.rewards = new Float32Array(totalSize)
    this.episodeStarts = new Uint8Array(totalSize)
    this.values = new Float32Array(totalSize)
    this.logProbs = new Float32Array(totalSize)

    // Computed arrays
    this.advantages = new Float32Array(totalSize)
    this.returns = new Float32Array(totalSize)
  }

  /**
   * Add a step of experience to the buffer
   *
   * @param obs - Observations [nEnvs * obsSize]
   * @param actions - Actions [nEnvs] or [nEnvs * actionDim]
   * @param rewards - Rewards [nEnvs]
   * @param episodeStarts - Whether each env started a new episode [nEnvs]
   * @param values - Value estimates [nEnvs]
   * @param logProbs - Log probabilities [nEnvs]
   */
  add(
    obs: Float32Array,
    actions: Float32Array | Int32Array,
    rewards: Float32Array,
    episodeStarts: Uint8Array,
    values: Float32Array,
    logProbs: Float32Array,
  ): void {
    if (this.full) {
      throw new Error('Rollout buffer is full. Call reset() before adding more data.')
    }

    const offset = this.position * this.nEnvs

    // Bulk copy observations [nEnvs * obsSize] using memcpy-backed set()
    this.observations.set(obs, offset * this.observationSize)

    // Bulk copy actions
    if (this.actionDim === 1) {
      // Discrete: actions may be Int32Array, need to copy element-wise into Float32Array
      for (let i = 0; i < this.nEnvs; i++) {
        this.actions[offset + i] = actions[i]!
      }
    } else {
      this.actions.set(actions as Float32Array, offset * this.actionDim)
    }

    // Bulk copy scalars using set()
    this.rewards.set(rewards, offset)
    this.episodeStarts.set(episodeStarts, offset)
    this.values.set(values, offset)
    this.logProbs.set(logProbs, offset)

    this.position++
    if (this.position === this.bufferSize) {
      this.full = true
    }
  }

  /**
   * Compute returns and advantages using GAE
   *
   * Must be called after the rollout is complete and before iterating.
   *
   * @param lastValues - Value estimates for the last observation [nEnvs]
   * @param lastDones - Done flags for the last step [nEnvs]
   */
  computeReturnsAndAdvantage(lastValues: Float32Array, lastDones: Uint8Array): void {
    if (!this.full) {
      throw new Error('Buffer not full. Cannot compute returns yet.')
    }

    // Try native fused GAE (single FFI call instead of nested JS loops)
    const gae = getNativeGae()
    if (gae) {
      try {
        gae(
          this.rewards, this.values, this.episodeStarts,
          lastValues, lastDones,
          this.bufferSize, this.nEnvs, this.gamma, this.gaeLambda,
          this.advantages, this.returns,
        )
        this.generatorReady = true
        return
      } catch {
        // Fall through to JS implementation
      }
    }

    // JS fallback: GAE computation (reverse iteration)
    const lastGaeLam = new Float32Array(this.nEnvs)

    for (let step = this.bufferSize - 1; step >= 0; step--) {
      const offset = step * this.nEnvs

      for (let envIdx = 0; envIdx < this.nEnvs; envIdx++) {
        const idx = offset + envIdx

        let nextNonTerminal: number
        let nextValue: number

        if (step === this.bufferSize - 1) {
          nextNonTerminal = 1.0 - lastDones[envIdx]!
          nextValue = lastValues[envIdx]!
        } else {
          const nextOffset = (step + 1) * this.nEnvs
          nextNonTerminal = 1.0 - this.episodeStarts[nextOffset + envIdx]!
          nextValue = this.values[nextOffset + envIdx]!
        }

        const delta =
          this.rewards[idx]! +
          this.gamma * nextValue * nextNonTerminal -
          this.values[idx]!

        lastGaeLam[envIdx] =
          delta + this.gamma * this.gaeLambda * nextNonTerminal * lastGaeLam[envIdx]!

        this.advantages[idx] = lastGaeLam[envIdx]!
      }
    }

    for (let i = 0; i < this.advantages.length; i++) {
      this.returns[i] = this.advantages[i]! + this.values[i]!
    }

    this.generatorReady = true
  }

  /**
   * Generate minibatches for training
   *
   * Shuffles all data and yields batches of the specified size.
   *
   * @param batchSize - Size of each minibatch (null = entire buffer)
   * @yields Minibatches of samples
   */
  *get(batchSize: number | null = null): Generator<RolloutBufferSamples> {
    if (!this.generatorReady) {
      throw new Error('Must call computeReturnsAndAdvantage() before iterating.')
    }

    const totalSize = this.bufferSize * this.nEnvs

    // Use full buffer if batchSize is null
    const actualBatchSize = batchSize ?? totalSize

    // Lazily allocate (or resize) reusable batch buffers
    if (this.lastBatchSize_ !== actualBatchSize) {
      this.batchObs_ = new Float32Array(actualBatchSize * this.observationSize)
      this.batchActions_ = new Float32Array(actualBatchSize * this.actionDim)
      this.batchOldValues_ = new Float32Array(actualBatchSize)
      this.batchOldLogProbs_ = new Float32Array(actualBatchSize)
      this.batchAdvantages_ = new Float32Array(actualBatchSize)
      this.batchReturns_ = new Float32Array(actualBatchSize)
      this.lastBatchSize_ = actualBatchSize
    }

    // Generate random permutation
    const indices = new Uint32Array(totalSize)
    for (let i = 0; i < totalSize; i++) {
      indices[i] = i
    }
    // Fisher-Yates shuffle
    for (let i = totalSize - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1))
      const temp = indices[i]!
      indices[i] = indices[j]!
      indices[j] = temp
    }

    const batchObs = this.batchObs_!
    const batchActions = this.batchActions_!
    const batchOldValues = this.batchOldValues_!
    const batchOldLogProbs = this.batchOldLogProbs_!
    const batchAdvantages = this.batchAdvantages_!
    const batchReturns = this.batchReturns_!
    const obsSize = this.observationSize
    const actionDim = this.actionDim

    // Generate batches (reuse pre-allocated buffers)
    let startIdx = 0
    while (startIdx < totalSize) {
      const endIdx = Math.min(startIdx + actualBatchSize, totalSize)
      const currentBatchSize = endIdx - startIdx

      // Copy data using shuffled indices
      for (let i = 0; i < currentBatchSize; i++) {
        const srcIdx = indices[startIdx + i]!

        // Copy observation using subarray + set for contiguous chunks
        batchObs.set(this.observations.subarray(srcIdx * obsSize, srcIdx * obsSize + obsSize), i * obsSize)

        // Copy action
        if (actionDim === 1) {
          batchActions[i] = this.actions[srcIdx]!
        } else {
          batchActions.set(
            this.actions.subarray(srcIdx * actionDim, srcIdx * actionDim + actionDim),
            i * actionDim,
          )
        }

        // Copy scalars
        batchOldValues[i] = this.values[srcIdx]!
        batchOldLogProbs[i] = this.logProbs[srcIdx]!
        batchAdvantages[i] = this.advantages[srcIdx]!
        batchReturns[i] = this.returns[srcIdx]!
      }

      yield {
        observations: currentBatchSize === actualBatchSize ? batchObs : batchObs.subarray(0, currentBatchSize * obsSize),
        actions: currentBatchSize === actualBatchSize ? batchActions : batchActions.subarray(0, currentBatchSize * actionDim),
        oldValues: currentBatchSize === actualBatchSize ? batchOldValues : batchOldValues.subarray(0, currentBatchSize),
        oldLogProbs: currentBatchSize === actualBatchSize ? batchOldLogProbs : batchOldLogProbs.subarray(0, currentBatchSize),
        advantages: currentBatchSize === actualBatchSize ? batchAdvantages : batchAdvantages.subarray(0, currentBatchSize),
        returns: currentBatchSize === actualBatchSize ? batchReturns : batchReturns.subarray(0, currentBatchSize),
        batchSize: currentBatchSize,
      }

      startIdx = endIdx
    }
  }

  /**
   * Reset the buffer for the next rollout
   */
  reset(): void {
    this.position = 0
    this.full = false
    this.generatorReady = false
  }

  /**
   * Whether the buffer is full
   */
  get isFull(): boolean {
    return this.full
  }

  /**
   * Current position in buffer
   */
  get currentPosition(): number {
    return this.position
  }

  /**
   * Total size of buffer (bufferSize * nEnvs)
   */
  get totalSize(): number {
    return this.bufferSize * this.nEnvs
  }

  /**
   * Get a writable view into the observations array for a given step.
   *
   * Used for shared-memory mode: the VecEnv writes observations directly
   * into the rollout buffer, eliminating one copy per step.
   *
   * @param step - Step index [0, bufferSize)
   * @returns Float32Array subarray view of length [nEnvs * observationSize]
   */
  getObsWriteTarget(step: number): Float32Array {
    const offset = step * this.nEnvs * this.observationSize
    return this.observations.subarray(offset, offset + this.nEnvs * this.observationSize)
  }

  /**
   * Add a step of experience WITHOUT copying observations.
   *
   * Used with shared-memory mode where obs were already written
   * via getObsWriteTarget(). Copies only actions, rewards, episodeStarts,
   * values, and logProbs.
   */
  addWithoutObs(
    actions: Float32Array | Int32Array,
    rewards: Float32Array,
    episodeStarts: Uint8Array,
    values: Float32Array,
    logProbs: Float32Array,
  ): void {
    if (this.full) {
      throw new Error('Rollout buffer is full. Call reset() before adding more data.')
    }

    const offset = this.position * this.nEnvs

    // Bulk copy actions
    if (this.actionDim === 1) {
      for (let i = 0; i < this.nEnvs; i++) {
        this.actions[offset + i] = actions[i]!
      }
    } else {
      this.actions.set(actions as Float32Array, offset * this.actionDim)
    }

    // Bulk copy scalars using set()
    this.rewards.set(rewards, offset)
    this.episodeStarts.set(episodeStarts, offset)
    this.values.set(values, offset)
    this.logProbs.set(logProbs, offset)

    this.position++
    if (this.position === this.bufferSize) {
      this.full = true
    }
  }
}
