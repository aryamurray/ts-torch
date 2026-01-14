/**
 * High-Performance Replay Buffer
 *
 * Experience replay buffer using pre-allocated TypedArrays for zero-allocation
 * push operations. Critical for Bun/V8 performance by avoiding GC pressure.
 *
 * Features:
 * - Circular buffer with head pointer (overwrites oldest when full)
 * - Pre-allocated contiguous memory slabs for cache efficiency
 * - Zero allocation in push() - just copy data into slabs
 * - Optional Prioritized Experience Replay (PER) with O(log n) sampling
 *
 * @example
 * ```ts
 * // Uniform sampling (default)
 * const buffer = new ReplayBuffer(10000, 4, 1)
 *
 * // Prioritized sampling
 * const perBuffer = new ReplayBuffer(10000, 4, 1, {
 *   prioritized: true,
 *   alpha: 0.6,
 *   betaStart: 0.4,
 *   betaEnd: 1.0,
 *   betaFrames: 100000
 * })
 * ```
 */

import { SumTree } from './utils/sum-tree.js'

// ==================== Types ====================

/**
 * A single transition (state, action, reward, nextState, done)
 */
export interface Transition {
  /** State observation */
  state: Float32Array
  /** Discrete action taken */
  action: number
  /** Reward received (number for single-obj, Float32Array for multi-obj) */
  reward: Float32Array | number
  /** Next state observation */
  nextState: Float32Array
  /** Whether episode ended */
  done: boolean
}

/**
 * Batch of transitions for training
 */
export interface TransitionBatch {
  /** States [batchSize, stateSize] as flat array */
  states: Float32Array
  /** Actions [batchSize] */
  actions: Int32Array
  /** Rewards [batchSize, rewardDim] as flat array */
  rewards: Float32Array
  /** Next states [batchSize, stateSize] as flat array */
  nextStates: Float32Array
  /** Done flags [batchSize] */
  dones: Uint8Array
  /** Batch size */
  batchSize: number
  /** State dimensionality */
  stateSize: number
  /** Reward dimensionality */
  rewardDim: number

  // PER-specific fields (only present when prioritized=true)

  /** Sampled indices for priority updates */
  indices?: Int32Array
  /** Importance sampling weights (for loss correction) */
  weights?: Float32Array
}

/**
 * Configuration for Prioritized Experience Replay
 */
export interface PERConfig {
  /** Enable prioritized sampling (default: false) */
  prioritized?: boolean
  /** Priority exponent - how much prioritization to use (default: 0.6) */
  alpha?: number
  /** Initial importance sampling exponent (default: 0.4) */
  betaStart?: number
  /** Final importance sampling exponent (default: 1.0) */
  betaEnd?: number
  /** Number of frames to anneal beta from start to end (default: 100000) */
  betaFrames?: number
}

// ==================== Implementation ====================

/**
 * High-performance experience replay buffer
 *
 * Uses pre-allocated TypedArrays to avoid garbage collection pressure
 * during training. Implements a circular buffer that overwrites oldest
 * experiences when full.
 *
 * When prioritized=true, uses a SumTree for O(log n) proportional sampling
 * based on TD-error priorities.
 */
export class ReplayBuffer {
  // Pre-allocated memory slabs
  private readonly states: Float32Array
  private readonly actions: Int32Array
  private readonly rewards: Float32Array
  private readonly nextStates: Float32Array
  private readonly dones: Uint8Array

  // Buffer metadata
  private readonly capacity: number
  private readonly stateSize: number
  private readonly rewardDim: number

  // Circular buffer state
  private head: number = 0
  private size_: number = 0

  // PER components
  private readonly prioritized: boolean
  private readonly sumTree: SumTree | null
  private readonly alpha: number
  private readonly betaStart: number
  private readonly betaEnd: number
  private readonly betaFrames: number
  private frameCount: number = 0

  /** Default priority for new transitions */
  private maxPriority: number = 1.0

  /** Small constant to ensure non-zero priorities */
  private readonly priorityEps: number = 1e-6

  /**
   * Create a new replay buffer
   *
   * @param capacity - Maximum number of transitions to store
   * @param stateSize - Dimensionality of state observations
   * @param rewardDim - Dimensionality of reward vector (default: 1 for scalar rewards)
   * @param config - PER configuration (optional)
   */
  constructor(
    capacity: number,
    stateSize: number,
    rewardDim: number = 1,
    config: PERConfig = {},
  ) {
    this.capacity = capacity
    this.stateSize = stateSize
    this.rewardDim = rewardDim

    // Pre-allocate all memory upfront
    this.states = new Float32Array(capacity * stateSize)
    this.actions = new Int32Array(capacity)
    this.rewards = new Float32Array(capacity * rewardDim)
    this.nextStates = new Float32Array(capacity * stateSize)
    this.dones = new Uint8Array(capacity)

    // PER setup
    this.prioritized = config.prioritized ?? false
    this.alpha = config.alpha ?? 0.6
    this.betaStart = config.betaStart ?? 0.4
    this.betaEnd = config.betaEnd ?? 1.0
    this.betaFrames = config.betaFrames ?? 100000

    // Create SumTree only if prioritized
    this.sumTree = this.prioritized ? new SumTree(capacity) : null
  }

  /**
   * Add a transition to the buffer
   *
   * O(1) for uniform, O(log n) for prioritized (due to SumTree update)
   * Zero allocation for data storage (copies into pre-allocated slabs)
   *
   * @param transition - Transition to add
   * @param priority - Initial priority (optional, uses max priority if not provided)
   */
  push(transition: Transition, priority?: number): void {
    const idx = this.head
    const stateOffset = idx * this.stateSize
    const rewardOffset = idx * this.rewardDim

    // Copy state
    this.states.set(transition.state, stateOffset)

    // Copy action
    this.actions[idx] = transition.action

    // Copy reward (handle both scalar and vector)
    if (typeof transition.reward === 'number') {
      this.rewards[rewardOffset] = transition.reward
    } else {
      this.rewards.set(transition.reward, rewardOffset)
    }

    // Copy next state
    this.nextStates.set(transition.nextState, stateOffset)

    // Copy done flag
    this.dones[idx] = transition.done ? 1 : 0

    // Update priority in SumTree (if PER enabled)
    if (this.sumTree) {
      // Use provided priority or max priority for new transitions
      const p = priority ?? this.maxPriority
      const priorityAlpha = Math.pow(p + this.priorityEps, this.alpha)
      this.sumTree.update(idx, priorityAlpha)
    }

    // Advance head (circular)
    this.head = (this.head + 1) % this.capacity

    // Update size
    if (this.size_ < this.capacity) {
      this.size_++
    }
  }

  /**
   * Sample a batch of transitions
   *
   * When prioritized=false: Uniform random sampling
   * When prioritized=true: Proportional sampling based on priorities
   *
   * @param batchSize - Number of transitions to sample
   * @returns Batch of transitions (includes indices and weights for PER)
   * @throws Error if buffer has fewer transitions than requested
   */
  sample(batchSize: number): TransitionBatch {
    if (batchSize > this.size_) {
      throw new Error(`Cannot sample ${batchSize} transitions from buffer with ${this.size_} transitions`)
    }

    // Allocate batch arrays
    const states = new Float32Array(batchSize * this.stateSize)
    const actions = new Int32Array(batchSize)
    const rewards = new Float32Array(batchSize * this.rewardDim)
    const nextStates = new Float32Array(batchSize * this.stateSize)
    const dones = new Uint8Array(batchSize)

    // PER-specific arrays
    const indices = this.prioritized ? new Int32Array(batchSize) : undefined
    const weights = this.prioritized ? new Float32Array(batchSize) : undefined

    // Calculate current beta for importance sampling
    const beta = this.prioritized ? this.getCurrentBeta() : 0

    if (this.prioritized && this.sumTree) {
      // Prioritized sampling using SumTree
      const totalPriority = this.sumTree.total
      const minPriority = this.sumTree.min
      const segment = totalPriority / batchSize

      // Max weight for normalization (from min priority transition)
      const maxWeight = Math.pow(this.size_ * minPriority / totalPriority, -beta)

      for (let i = 0; i < batchSize; i++) {
        // Sample from segment [i*segment, (i+1)*segment)
        const a = segment * i
        const b = segment * (i + 1)
        const value = a + Math.random() * (b - a)

        const idx = this.sumTree.sample(value)
        indices![i] = idx

        // Calculate importance sampling weight
        const priority = this.sumTree.get(idx)
        const prob = priority / totalPriority
        const weight = Math.pow(this.size_ * prob, -beta) / maxWeight
        weights![i] = weight

        // Copy transition data
        this.copyTransition(idx, i, states, actions, rewards, nextStates, dones)
      }

      this.frameCount++
    } else {
      // Uniform sampling
      for (let i = 0; i < batchSize; i++) {
        const idx = Math.floor(Math.random() * this.size_)
        this.copyTransition(idx, i, states, actions, rewards, nextStates, dones)
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
      rewardDim: this.rewardDim,
      indices,
      weights,
    }
  }

  /**
   * Update priorities for sampled transitions
   *
   * Call this after computing TD-errors in training step.
   * Only has effect when prioritized=true.
   *
   * @param indices - Indices of sampled transitions
   * @param priorities - New priorities (typically abs(TD-error))
   */
  updatePriorities(indices: Int32Array, priorities: Float32Array): void {
    if (!this.sumTree) return

    for (let i = 0; i < indices.length; i++) {
      const idx = indices[i]!
      const priority = priorities[i]!

      // Track max priority for new transitions
      if (priority > this.maxPriority) {
        this.maxPriority = priority
      }

      // Update SumTree with priority^alpha
      const priorityAlpha = Math.pow(priority + this.priorityEps, this.alpha)
      this.sumTree.update(idx, priorityAlpha)
    }
  }

  /**
   * Sample individual Transition objects (less efficient, for compatibility)
   *
   * @param batchSize - Number of transitions to sample
   * @returns Array of Transition objects
   */
  sampleTransitions(batchSize: number): Transition[] {
    if (batchSize > this.size_) {
      throw new Error(`Cannot sample ${batchSize} transitions from buffer with ${this.size_} transitions`)
    }

    const transitions: Transition[] = []

    for (let i = 0; i < batchSize; i++) {
      const idx = Math.floor(Math.random() * this.size_)
      const stateOffset = idx * this.stateSize
      const rewardOffset = idx * this.rewardDim

      transitions.push({
        state: this.states.slice(stateOffset, stateOffset + this.stateSize),
        action: this.actions[idx]!,
        reward:
          this.rewardDim === 1
            ? this.rewards[rewardOffset]!
            : this.rewards.slice(rewardOffset, rewardOffset + this.rewardDim),
        nextState: this.nextStates.slice(stateOffset, stateOffset + this.stateSize),
        done: this.dones[idx] === 1,
      })
    }

    return transitions
  }

  /**
   * Get a specific transition by index
   *
   * @param index - Index in buffer (0 to size-1)
   * @returns Transition at index
   */
  get(index: number): Transition {
    if (index < 0 || index >= this.size_) {
      throw new Error(`Index ${index} out of bounds [0, ${this.size_})`)
    }

    const stateOffset = index * this.stateSize
    const rewardOffset = index * this.rewardDim

    return {
      state: this.states.slice(stateOffset, stateOffset + this.stateSize),
      action: this.actions[index]!,
      reward:
        this.rewardDim === 1
          ? this.rewards[rewardOffset]!
          : this.rewards.slice(rewardOffset, rewardOffset + this.rewardDim),
      nextState: this.nextStates.slice(stateOffset, stateOffset + this.stateSize),
      done: this.dones[index] === 1,
    }
  }

  /**
   * Current number of transitions in buffer
   */
  get size(): number {
    return this.size_
  }

  /**
   * Whether buffer is at capacity
   */
  get isFull(): boolean {
    return this.size_ === this.capacity
  }

  /**
   * Maximum capacity of buffer
   */
  get maxCapacity(): number {
    return this.capacity
  }

  /**
   * Whether this buffer uses prioritized sampling
   */
  get isPrioritized(): boolean {
    return this.prioritized
  }

  /**
   * Current beta value for importance sampling (only for PER)
   */
  get currentBeta(): number {
    return this.getCurrentBeta()
  }

  /**
   * Clear all transitions from buffer
   */
  clear(): void {
    this.head = 0
    this.size_ = 0
    this.frameCount = 0
    this.maxPriority = 1.0
    if (this.sumTree) {
      this.sumTree.clear()
    }
  }

  // ==================== Private Methods ====================

  /**
   * Copy transition data from source index to destination in batch arrays
   */
  private copyTransition(
    srcIdx: number,
    dstIdx: number,
    states: Float32Array,
    actions: Int32Array,
    rewards: Float32Array,
    nextStates: Float32Array,
    dones: Uint8Array,
  ): void {
    const srcStateOffset = srcIdx * this.stateSize
    const srcRewardOffset = srcIdx * this.rewardDim
    const dstStateOffset = dstIdx * this.stateSize
    const dstRewardOffset = dstIdx * this.rewardDim

    // Copy state
    for (let j = 0; j < this.stateSize; j++) {
      states[dstStateOffset + j] = this.states[srcStateOffset + j]!
    }

    // Copy action
    actions[dstIdx] = this.actions[srcIdx]!

    // Copy reward
    for (let j = 0; j < this.rewardDim; j++) {
      rewards[dstRewardOffset + j] = this.rewards[srcRewardOffset + j]!
    }

    // Copy next state
    for (let j = 0; j < this.stateSize; j++) {
      nextStates[dstStateOffset + j] = this.nextStates[srcStateOffset + j]!
    }

    // Copy done
    dones[dstIdx] = this.dones[srcIdx]!
  }

  /**
   * Calculate current beta value (linearly annealed)
   */
  private getCurrentBeta(): number {
    if (!this.prioritized) return 0

    const fraction = Math.min(1.0, this.frameCount / this.betaFrames)
    return this.betaStart + fraction * (this.betaEnd - this.betaStart)
  }
}
