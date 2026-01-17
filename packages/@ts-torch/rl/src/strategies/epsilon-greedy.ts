/**
 * Epsilon-Greedy Exploration Strategy
 *
 * Classic exploration strategy that selects random actions with probability
 * epsilon, and greedy (best Q-value) actions otherwise. Epsilon decays
 * over time to shift from exploration to exploitation.
 *
 * @example
 * ```ts
 * const strategy = new EpsilonGreedyStrategy({
 *   start: 1.0,    // 100% random initially
 *   end: 0.05,     // 5% random at convergence
 *   decay: 0.995   // Multiply epsilon by this each step
 * })
 *
 * // During training:
 * const action = strategy.selectAction(qValues, numActions)
 * strategy.step()  // Decay epsilon
 * ```
 */

// ==================== Types ====================

/**
 * Configuration for epsilon-greedy exploration
 */
export interface EpsilonGreedyConfig {
  /** Initial epsilon (probability of random action) */
  start: number
  /** Minimum epsilon (lower bound) */
  end: number
  /** Decay factor per step (multiplied each step) */
  decay: number
}

// ==================== Implementation ====================

/**
 * Epsilon-greedy exploration strategy
 */
export class EpsilonGreedyStrategy {
  private epsilon: number
  private readonly endEpsilon: number
  private readonly decayFactor: number

  constructor(config: EpsilonGreedyConfig) {
    this.epsilon = config.start
    this.endEpsilon = config.end
    this.decayFactor = config.decay
  }

  /**
   * Select an action using epsilon-greedy policy
   *
   * @param qValues - Q-values for each action (Float32Array or number[])
   * @param actionSpace - Number of available actions
   * @returns Selected action index
   */
  selectAction(qValues: Float32Array | number[], actionSpace: number): number {
    // Epsilon probability: random action
    if (Math.random() < this.epsilon) {
      return Math.floor(Math.random() * actionSpace)
    }

    // Otherwise: greedy action (argmax of Q-values)
    return this.argmax(qValues)
  }

  /**
   * Select a random action (for pure exploration)
   *
   * @param actionSpace - Number of available actions
   * @returns Random action index
   */
  selectRandom(actionSpace: number): number {
    return Math.floor(Math.random() * actionSpace)
  }

  /**
   * Select the greedy action (for evaluation/exploitation)
   *
   * @param qValues - Q-values for each action
   * @returns Action with highest Q-value
   */
  selectGreedy(qValues: Float32Array | number[]): number {
    return this.argmax(qValues)
  }

  /**
   * Decay epsilon by one step
   */
  step(): void {
    this.epsilon = Math.max(this.endEpsilon, this.epsilon * this.decayFactor)
  }

  /**
   * Reset epsilon to initial value
   *
   * @param startEpsilon - Value to reset to (optional, uses original start if not provided)
   */
  reset(startEpsilon?: number): void {
    this.epsilon = startEpsilon ?? 1.0
  }

  /**
   * Set epsilon to a specific value
   *
   * @param value - The epsilon value to set (clamped to [endEpsilon, 1.0])
   */
  setEpsilon(value: number): void {
    this.epsilon = Math.max(this.endEpsilon, Math.min(1.0, value))
  }

  /**
   * Current epsilon value
   */
  get currentEpsilon(): number {
    return this.epsilon
  }

  /**
   * Whether exploration is effectively done (epsilon at minimum)
   */
  get isConverged(): boolean {
    return this.epsilon <= this.endEpsilon + 1e-8
  }

  /**
   * Find index of maximum value (argmax)
   */
  private argmax(values: Float32Array | number[]): number {
    let maxIdx = 0
    let maxVal = values[0]!

    for (let i = 1; i < values.length; i++) {
      if (values[i]! > maxVal) {
        maxVal = values[i]!
        maxIdx = i
      }
    }

    return maxIdx
  }
}

/**
 * Factory function for epsilon-greedy strategy
 *
 * @param config - Strategy configuration
 * @returns EpsilonGreedyStrategy instance
 */
export function epsilonGreedy(config: EpsilonGreedyConfig): EpsilonGreedyStrategy {
  return new EpsilonGreedyStrategy(config)
}
