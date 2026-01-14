/**
 * Discrete Action/Observation Space
 *
 * Represents a finite set of discrete values {0, 1, ..., n-1}.
 * Used for environments with discrete actions (e.g., CartPole: left/right).
 *
 * @example
 * ```ts
 * const actionSpace = discrete(4)  // Actions: 0, 1, 2, 3
 * actionSpace.sample()  // Random action
 * actionSpace.contains(2)  // true
 * ```
 */

// ==================== Types ====================

/**
 * Discrete space - finite set of integers {0, 1, ..., n-1}
 */
export interface DiscreteSpace {
  readonly type: 'discrete'
  /** Number of discrete values */
  readonly n: number
  /** Shape of the space (always [1] for scalar) */
  readonly shape: readonly [1]

  /**
   * Sample a random value from the space
   * @returns Random integer in [0, n)
   */
  sample(): number

  /**
   * Check if a value is contained in the space
   * @param x - Value to check
   * @returns True if x is a valid value
   */
  contains(x: number): boolean
}

// ==================== Implementation ====================

class DiscreteSpaceImpl implements DiscreteSpace {
  readonly type = 'discrete' as const
  readonly shape = [1] as const

  constructor(readonly n: number) {
    if (!Number.isInteger(n) || n <= 0) {
      throw new Error(`Discrete space n must be a positive integer, got ${n}`)
    }
  }

  sample(): number {
    return Math.floor(Math.random() * this.n)
  }

  contains(x: number): boolean {
    return Number.isInteger(x) && x >= 0 && x < this.n
  }
}

// ==================== Factory ====================

/**
 * Create a discrete action/observation space
 *
 * @param n - Number of discrete values (actions will be 0 to n-1)
 * @returns Discrete space instance
 *
 * @example
 * ```ts
 * // CartPole: 2 actions (left, right)
 * const actionSpace = discrete(2)
 *
 * // Atari: 18 actions
 * const actionSpace = discrete(18)
 * ```
 */
export function discrete(n: number): DiscreteSpace {
  return new DiscreteSpaceImpl(n)
}
