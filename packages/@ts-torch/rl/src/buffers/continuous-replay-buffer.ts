/**
 * Continuous Replay Buffer
 *
 * Experience replay buffer for continuous action spaces.
 * Used by SAC, TD3, DDPG and other off-policy continuous control algorithms.
 *
 * Similar to ReplayBuffer but stores continuous actions as Float32Array
 * instead of discrete actions as Int32Array.
 *
 * @example
 * ```ts
 * const buffer = new ContinuousReplayBuffer({
 *   capacity: 1_000_000,
 *   stateSize: 11,
 *   actionDim: 3,
 * })
 *
 * buffer.push({
 *   state: observation,
 *   action: new Float32Array([0.5, -0.3, 0.1]),
 *   reward: 1.0,
 *   nextState: nextObservation,
 *   done: false,
 * })
 * ```
 */

// ==================== Types ====================

/**
 * Transition with continuous action
 */
export interface ContinuousTransition {
  /** State observation */
  state: Float32Array
  /** Continuous action [actionDim] */
  action: Float32Array
  /** Reward (scalar) */
  reward: number
  /** Next state observation */
  nextState: Float32Array
  /** Whether episode ended */
  done: boolean
}

/**
 * Batch of continuous transitions
 */
export interface ContinuousTransitionBatch {
  /** States [batchSize, stateSize] as flat array */
  states: Float32Array
  /** Actions [batchSize, actionDim] as flat array */
  actions: Float32Array
  /** Rewards [batchSize] */
  rewards: Float32Array
  /** Next states [batchSize, stateSize] as flat array */
  nextStates: Float32Array
  /** Done flags [batchSize] */
  dones: Uint8Array
  /** Batch size */
  batchSize: number
  /** State dimensionality */
  stateSize: number
  /** Action dimensionality */
  actionDim: number
}

/**
 * Configuration for ContinuousReplayBuffer
 */
export interface ContinuousReplayBufferConfig {
  /** Maximum number of transitions */
  capacity: number
  /** State observation size */
  stateSize: number
  /** Continuous action dimension */
  actionDim: number
}

// ==================== Implementation ====================

/**
 * Replay buffer for continuous action spaces
 *
 * Uses pre-allocated TypedArrays for zero-allocation push.
 * Implements circular buffer that overwrites oldest when full.
 */
export class ContinuousReplayBuffer {
  // Pre-allocated storage
  private readonly states: Float32Array
  private readonly actions: Float32Array
  private readonly rewards: Float32Array
  private readonly nextStates: Float32Array
  private readonly dones: Uint8Array

  // Configuration
  private readonly capacity: number
  private readonly stateSize: number
  private readonly actionDim: number

  // State
  private head: number = 0
  private size_: number = 0

  constructor(config: ContinuousReplayBufferConfig) {
    this.capacity = config.capacity
    this.stateSize = config.stateSize
    this.actionDim = config.actionDim

    // Pre-allocate
    this.states = new Float32Array(config.capacity * config.stateSize)
    this.actions = new Float32Array(config.capacity * config.actionDim)
    this.rewards = new Float32Array(config.capacity)
    this.nextStates = new Float32Array(config.capacity * config.stateSize)
    this.dones = new Uint8Array(config.capacity)
  }

  /**
   * Add a transition to the buffer
   */
  push(transition: ContinuousTransition): void {
    const idx = this.head
    const stateOffset = idx * this.stateSize
    const actionOffset = idx * this.actionDim

    // Copy data
    this.states.set(transition.state, stateOffset)
    this.actions.set(transition.action, actionOffset)
    this.rewards[idx] = transition.reward
    this.nextStates.set(transition.nextState, stateOffset)
    this.dones[idx] = transition.done ? 1 : 0

    // Advance head
    this.head = (this.head + 1) % this.capacity
    if (this.size_ < this.capacity) {
      this.size_++
    }
  }

  /**
   * Sample a batch of transitions
   */
  sample(batchSize: number): ContinuousTransitionBatch {
    if (batchSize > this.size_) {
      throw new Error(`Cannot sample ${batchSize} from buffer with ${this.size_} transitions`)
    }

    // Allocate batch arrays
    const states = new Float32Array(batchSize * this.stateSize)
    const actions = new Float32Array(batchSize * this.actionDim)
    const rewards = new Float32Array(batchSize)
    const nextStates = new Float32Array(batchSize * this.stateSize)
    const dones = new Uint8Array(batchSize)

    // Random sampling
    for (let i = 0; i < batchSize; i++) {
      const srcIdx = Math.floor(Math.random() * this.size_)
      const srcStateOffset = srcIdx * this.stateSize
      const srcActionOffset = srcIdx * this.actionDim
      const dstStateOffset = i * this.stateSize
      const dstActionOffset = i * this.actionDim

      // Copy state
      for (let j = 0; j < this.stateSize; j++) {
        states[dstStateOffset + j] = this.states[srcStateOffset + j]!
      }

      // Copy action
      for (let j = 0; j < this.actionDim; j++) {
        actions[dstActionOffset + j] = this.actions[srcActionOffset + j]!
      }

      // Copy scalars
      rewards[i] = this.rewards[srcIdx]!
      dones[i] = this.dones[srcIdx]!

      // Copy next state
      for (let j = 0; j < this.stateSize; j++) {
        nextStates[dstStateOffset + j] = this.nextStates[srcStateOffset + j]!
      }
    }

    return {
      states,
      actions,
      rewards,
      nextStates,
      dones,
      batchSize,
      stateSize: this.stateSize,
      actionDim: this.actionDim,
    }
  }

  /**
   * Current buffer size
   */
  get size(): number {
    return this.size_
  }

  /**
   * Whether buffer is full
   */
  get isFull(): boolean {
    return this.size_ === this.capacity
  }

  /**
   * Maximum capacity
   */
  get maxCapacity(): number {
    return this.capacity
  }

  /**
   * Clear the buffer
   */
  clear(): void {
    this.head = 0
    this.size_ = 0
  }
}
