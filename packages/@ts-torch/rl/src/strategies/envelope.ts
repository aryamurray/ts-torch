/**
 * Envelope Q-Learning Strategy for Multi-Objective RL
 *
 * Implements envelope Q-learning for Pareto optimization with multiple
 * objectives. Samples weight vectors from the simplex and uses them
 * to scalarize multi-objective Q-values.
 *
 * Key Concepts:
 * - Weight vectors sampled uniformly from (n-1)-simplex
 * - Q-network is conditioned on weights (concatenated to state)
 * - Bellman target uses envelope (max over weight-scalarized Q-values)
 *
 * Reference: "A Generalized Algorithm for Multi-Objective Reinforcement Learning
 *             and Policy Adaptation" (Yang et al., 2019)
 *
 * @example
 * ```ts
 * const strategy = new EnvelopeQStrategy({
 *   start: 1.0,
 *   end: 0.05,
 *   decay: 0.995,
 *   rewardDim: 3  // Three objectives
 * })
 *
 * const weights = strategy.sampleWeights()  // [0.3, 0.5, 0.2]
 * const action = strategy.selectAction(qValues, weights, numActions)
 * ```
 */

// ==================== Types ====================

/**
 * Configuration for envelope Q-learning
 */
export interface EnvelopeConfig {
  /** Initial epsilon for exploration */
  start: number
  /** Minimum epsilon */
  end: number
  /** Decay factor per step */
  decay: number
  /** Number of objectives (reward dimensions) */
  rewardDim: number
}

// ==================== Implementation ====================

/**
 * Envelope Q-learning strategy for multi-objective RL
 */
export class EnvelopeQStrategy {
  private epsilon: number
  private readonly endEpsilon: number
  private readonly decayFactor: number
  private readonly rewardDim: number

  // Cache for weight sampling
  private currentWeights: Float32Array

  constructor(config: EnvelopeConfig) {
    this.epsilon = config.start
    this.endEpsilon = config.end
    this.decayFactor = config.decay
    this.rewardDim = config.rewardDim
    this.currentWeights = new Float32Array(config.rewardDim)

    // Initialize with uniform weights
    const uniform = 1 / config.rewardDim
    for (let i = 0; i < config.rewardDim; i++) {
      this.currentWeights[i] = uniform
    }
  }

  /**
   * Sample a new weight vector uniformly from the (n-1)-simplex
   *
   * Uses the Dirichlet(1,1,...,1) distribution which is uniform on simplex.
   * Implementation: sample n exponential(1) values and normalize.
   *
   * @returns Weight vector that sums to 1
   */
  sampleWeights(): Float32Array {
    const weights = new Float32Array(this.rewardDim)
    let sum = 0

    // Sample from Exponential(1) = -ln(U) where U ~ Uniform(0,1)
    for (let i = 0; i < this.rewardDim; i++) {
      // Avoid log(0) by using 1 - U instead of U
      weights[i] = -Math.log(1 - Math.random() + 1e-10)
      sum += weights[i]!
    }

    // Normalize to sum to 1
    for (let i = 0; i < this.rewardDim; i++) {
      weights[i] = weights[i]! / sum
    }

    // Cache current weights
    this.currentWeights = weights

    return weights
  }

  /**
   * Get the current weight vector (without sampling new)
   */
  getWeights(): Float32Array {
    return this.currentWeights
  }

  /**
   * Select an action using epsilon-greedy with scalarized Q-values
   *
   * @param qValues - Multi-objective Q-values [numActions * rewardDim]
   * @param weights - Weight vector for scalarization
   * @param actionSpace - Number of available actions
   * @returns Selected action index
   */
  selectAction(qValues: Float32Array, weights: Float32Array, actionSpace: number): number {
    // Epsilon probability: random action
    if (Math.random() < this.epsilon) {
      return Math.floor(Math.random() * actionSpace)
    }

    // Otherwise: greedy action based on scalarized Q-values
    return this.selectGreedy(qValues, weights, actionSpace)
  }

  /**
   * Select the greedy action (argmax of scalarized Q-values)
   *
   * @param qValues - Multi-objective Q-values [numActions * rewardDim]
   * @param weights - Weight vector for scalarization
   * @param actionSpace - Number of available actions
   * @returns Action with highest scalarized Q-value
   */
  selectGreedy(qValues: Float32Array, weights: Float32Array, actionSpace: number): number {
    let bestAction = 0
    let bestValue = -Infinity

    for (let a = 0; a < actionSpace; a++) {
      const scalarQ = this.scalarize(qValues, weights, a, actionSpace)
      if (scalarQ > bestValue) {
        bestValue = scalarQ
        bestAction = a
      }
    }

    return bestAction
  }

  /**
   * Compute scalarized Q-value for a specific action
   *
   * @param qValues - Multi-objective Q-values [numActions * rewardDim]
   * @param weights - Weight vector
   * @param action - Action index
   * @param actionSpace - Total number of actions
   * @returns Weighted sum of Q-values for this action
   */
  scalarize(qValues: Float32Array, weights: Float32Array, action: number, _actionSpace: number): number {
    let sum = 0
    const offset = action * this.rewardDim

    for (let i = 0; i < this.rewardDim; i++) {
      sum += weights[i]! * qValues[offset + i]!
    }

    return sum
  }

  /**
   * Compute scalarized reward from reward vector
   *
   * @param reward - Multi-objective reward vector
   * @param weights - Weight vector
   * @returns Scalar reward
   */
  scalarizeReward(reward: Float32Array, weights: Float32Array): number {
    let sum = 0
    for (let i = 0; i < this.rewardDim; i++) {
      sum += weights[i]! * reward[i]!
    }
    return sum
  }

  /**
   * Decay epsilon by one step
   */
  step(): void {
    this.epsilon = Math.max(this.endEpsilon, this.epsilon * this.decayFactor)
  }

  /**
   * Reset epsilon to initial value
   */
  reset(startEpsilon?: number): void {
    this.epsilon = startEpsilon ?? 1.0
  }

  /**
   * Current epsilon value
   */
  get currentEpsilon(): number {
    return this.epsilon
  }

  /**
   * Number of objectives
   */
  get numObjectives(): number {
    return this.rewardDim
  }
}

/**
 * Factory function for envelope Q-learning strategy
 */
export function envelopeQ(config: EnvelopeConfig): EnvelopeQStrategy {
  return new EnvelopeQStrategy(config)
}
