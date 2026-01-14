/**
 * Box (Continuous) Action/Observation Space
 *
 * Represents a bounded continuous space with shape and bounds.
 * Used for environments with continuous actions (e.g., Pendulum: torque)
 * or continuous observations (e.g., joint positions, velocities).
 *
 * @example
 * ```ts
 * // Pendulum action: torque in [-2, 2]
 * const actionSpace = box({ low: [-2], high: [2], shape: [1] })
 *
 * // Observation: position and velocity
 * const obsSpace = box({
 *   low: [-1, -1, -8],
 *   high: [1, 1, 8],
 *   shape: [3]
 * })
 * ```
 */

// ==================== Types ====================

/**
 * Configuration for creating a Box space
 */
export interface BoxConfig {
  /** Lower bounds for each dimension */
  low: number[]
  /** Upper bounds for each dimension */
  high: number[]
  /** Shape of the space */
  shape: number[]
}

/**
 * Box space - continuous bounded space
 */
export interface BoxSpace {
  readonly type: 'box'
  /** Shape of the space */
  readonly shape: readonly number[]
  /** Lower bounds */
  readonly low: Float32Array
  /** Upper bounds */
  readonly high: Float32Array

  /**
   * Sample a random value uniformly from the space
   * @returns Random Float32Array within bounds
   */
  sample(): Float32Array

  /**
   * Check if a value is contained in the space
   * @param x - Value to check
   * @returns True if x is within bounds
   */
  contains(x: Float32Array | number[]): boolean

  /**
   * Clip a value to be within bounds
   * @param x - Value to clip
   * @returns Clipped value
   */
  clip(x: Float32Array | number[]): Float32Array
}

// ==================== Implementation ====================

class BoxSpaceImpl implements BoxSpace {
  readonly type = 'box' as const
  readonly shape: readonly number[]
  readonly low: Float32Array
  readonly high: Float32Array
  private readonly flatSize: number

  constructor(config: BoxConfig) {
    this.shape = Object.freeze([...config.shape])
    this.flatSize = config.shape.reduce((a, b) => a * b, 1)

    // Validate and create bounds
    if (config.low.length !== this.flatSize) {
      throw new Error(`Low bounds length (${config.low.length}) must match shape size (${this.flatSize})`)
    }
    if (config.high.length !== this.flatSize) {
      throw new Error(`High bounds length (${config.high.length}) must match shape size (${this.flatSize})`)
    }

    this.low = new Float32Array(config.low)
    this.high = new Float32Array(config.high)

    // Validate bounds
    for (let i = 0; i < this.flatSize; i++) {
      if (this.low[i]! > this.high[i]!) {
        throw new Error(`Low bound (${this.low[i]}) must be <= high bound (${this.high[i]}) at index ${i}`)
      }
    }
  }

  sample(): Float32Array {
    const result = new Float32Array(this.flatSize)
    for (let i = 0; i < this.flatSize; i++) {
      const lo = this.low[i]!
      const hi = this.high[i]!
      // Handle infinite bounds
      if (!Number.isFinite(lo) || !Number.isFinite(hi)) {
        // Sample from standard normal for unbounded dimensions
        result[i] = this.sampleNormal()
      } else {
        result[i] = lo + Math.random() * (hi - lo)
      }
    }
    return result
  }

  contains(x: Float32Array | number[]): boolean {
    if (x.length !== this.flatSize) {
      return false
    }
    for (let i = 0; i < this.flatSize; i++) {
      const val = x[i]!
      if (val < this.low[i]! || val > this.high[i]!) {
        return false
      }
    }
    return true
  }

  clip(x: Float32Array | number[]): Float32Array {
    const result = new Float32Array(this.flatSize)
    for (let i = 0; i < this.flatSize; i++) {
      result[i] = Math.max(this.low[i]!, Math.min(this.high[i]!, x[i]!))
    }
    return result
  }

  /**
   * Sample from standard normal distribution (Box-Muller transform)
   */
  private sampleNormal(): number {
    const u1 = Math.random()
    const u2 = Math.random()
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
  }
}

// ==================== Factory ====================

/**
 * Create a box (continuous) action/observation space
 *
 * @param config - Space configuration with bounds and shape
 * @returns Box space instance
 *
 * @example
 * ```ts
 * // Single continuous action
 * const actionSpace = box({ low: [-1], high: [1], shape: [1] })
 *
 * // Multi-dimensional action
 * const actionSpace = box({
 *   low: [-1, -1, -1],
 *   high: [1, 1, 1],
 *   shape: [3]
 * })
 *
 * // Unbounded observation (uses -Infinity, Infinity)
 * const obsSpace = box({
 *   low: [-Infinity, -Infinity],
 *   high: [Infinity, Infinity],
 *   shape: [2]
 * })
 * ```
 */
export function box(config: BoxConfig): BoxSpace {
  return new BoxSpaceImpl(config)
}

/**
 * Create a box space with uniform bounds
 *
 * @param low - Lower bound for all dimensions
 * @param high - Upper bound for all dimensions
 * @param shape - Shape of the space
 * @returns Box space instance
 *
 * @example
 * ```ts
 * // All dimensions bounded by [-1, 1]
 * const space = boxUniform(-1, 1, [3])
 * ```
 */
export function boxUniform(low: number, high: number, shape: number[]): BoxSpace {
  const size = shape.reduce((a, b) => a * b, 1)
  return new BoxSpaceImpl({
    low: new Array(size).fill(low),
    high: new Array(size).fill(high),
    shape,
  })
}
