/**
 * Rollout Buffer for On-Policy Algorithms
 *
 * Stores trajectories collected during rollouts for PPO/A2C training.
 * Computes returns and advantages using Generalized Advantage Estimation (GAE).
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

    // Copy observations [nEnvs, obsSize] -> flat
    for (let i = 0; i < this.nEnvs; i++) {
      const srcOffset = i * this.observationSize
      const dstOffset = (offset + i) * this.observationSize
      for (let j = 0; j < this.observationSize; j++) {
        this.observations[dstOffset + j] = obs[srcOffset + j]!
      }
    }

    // Copy actions
    if (this.actionDim === 1) {
      // Discrete: [nEnvs]
      for (let i = 0; i < this.nEnvs; i++) {
        this.actions[offset + i] = actions[i]!
      }
    } else {
      // Continuous: [nEnvs, actionDim]
      for (let i = 0; i < this.nEnvs; i++) {
        const srcOffset = i * this.actionDim
        const dstOffset = (offset + i) * this.actionDim
        for (let j = 0; j < this.actionDim; j++) {
          this.actions[dstOffset + j] = actions[srcOffset + j]!
        }
      }
    }

    // Copy scalars
    for (let i = 0; i < this.nEnvs; i++) {
      this.rewards[offset + i] = rewards[i]!
      this.episodeStarts[offset + i] = episodeStarts[i]!
      this.values[offset + i] = values[i]!
      this.logProbs[offset + i] = logProbs[i]!
    }

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

    // GAE computation (reverse iteration)
    // A_t = delta_t + gamma * lambda * (1 - done_{t+1}) * A_{t+1}
    // where delta_t = r_t + gamma * (1 - done_{t+1}) * V(s_{t+1}) - V(s_t)

    let lastGaeLam = new Float32Array(this.nEnvs)

    for (let step = this.bufferSize - 1; step >= 0; step--) {
      const offset = step * this.nEnvs

      for (let envIdx = 0; envIdx < this.nEnvs; envIdx++) {
        const idx = offset + envIdx

        // Determine if next step is terminal
        let nextNonTerminal: number
        let nextValue: number

        if (step === this.bufferSize - 1) {
          // Last step: use provided lastValues and lastDones
          nextNonTerminal = 1.0 - lastDones[envIdx]!
          nextValue = lastValues[envIdx]!
        } else {
          // Use next step's episode start to determine if this step was terminal
          const nextOffset = (step + 1) * this.nEnvs
          nextNonTerminal = 1.0 - this.episodeStarts[nextOffset + envIdx]!
          nextValue = this.values[nextOffset + envIdx]!
        }

        // TD error
        const delta =
          this.rewards[idx]! +
          this.gamma * nextValue * nextNonTerminal -
          this.values[idx]!

        // GAE
        lastGaeLam[envIdx] =
          delta + this.gamma * this.gaeLambda * nextNonTerminal * lastGaeLam[envIdx]!

        this.advantages[idx] = lastGaeLam[envIdx]!
      }
    }

    // Returns = advantages + values
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

    // Generate batches
    let startIdx = 0
    while (startIdx < totalSize) {
      const endIdx = Math.min(startIdx + actualBatchSize, totalSize)
      const currentBatchSize = endIdx - startIdx

      // Allocate batch arrays
      const batchObs = new Float32Array(currentBatchSize * this.observationSize)
      const batchActions = new Float32Array(currentBatchSize * this.actionDim)
      const batchOldValues = new Float32Array(currentBatchSize)
      const batchOldLogProbs = new Float32Array(currentBatchSize)
      const batchAdvantages = new Float32Array(currentBatchSize)
      const batchReturns = new Float32Array(currentBatchSize)

      // Copy data using shuffled indices
      for (let i = 0; i < currentBatchSize; i++) {
        const srcIdx = indices[startIdx + i]!

        // Copy observation
        const obsSrcOffset = srcIdx * this.observationSize
        const obsDstOffset = i * this.observationSize
        for (let j = 0; j < this.observationSize; j++) {
          batchObs[obsDstOffset + j] = this.observations[obsSrcOffset + j]!
        }

        // Copy action
        if (this.actionDim === 1) {
          batchActions[i] = this.actions[srcIdx]!
        } else {
          const actSrcOffset = srcIdx * this.actionDim
          const actDstOffset = i * this.actionDim
          for (let j = 0; j < this.actionDim; j++) {
            batchActions[actDstOffset + j] = this.actions[actSrcOffset + j]!
          }
        }

        // Copy scalars
        batchOldValues[i] = this.values[srcIdx]!
        batchOldLogProbs[i] = this.logProbs[srcIdx]!
        batchAdvantages[i] = this.advantages[srcIdx]!
        batchReturns[i] = this.returns[srcIdx]!
      }

      yield {
        observations: batchObs,
        actions: batchActions,
        oldValues: batchOldValues,
        oldLogProbs: batchOldLogProbs,
        advantages: batchAdvantages,
        returns: batchReturns,
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
}
