/**
 * Multi-Objective Reward Utilities
 *
 * Helper functions for working with multi-objective rewards in MORL.
 * Provides scalarization, normalization, and weight sampling utilities.
 *
 * @example
 * ```ts
 * import { scalarize, sampleSimplex, normalizeWeights } from '@ts-torch/rl'
 *
 * const weights = sampleSimplex(3)  // [0.3, 0.5, 0.2]
 * const rewards = new Float32Array([-10, -2, 5])  // [time, transfers, coverage]
 * const scalar = scalarize(weights, rewards)  // Weighted sum
 * ```
 */

// ==================== Scalarization ====================

/**
 * Compute weighted sum (linear scalarization) of reward vector
 *
 * @param weights - Weight vector (should sum to 1)
 * @param rewards - Reward vector
 * @returns Scalar reward
 */
export function scalarize(weights: Float32Array, rewards: Float32Array): number {
  if (weights.length !== rewards.length) {
    throw new Error(`Weight length ${weights.length} != reward length ${rewards.length}`)
  }

  let sum = 0
  for (let i = 0; i < weights.length; i++) {
    sum += weights[i]! * rewards[i]!
  }
  return sum
}

/**
 * Compute Chebyshev scalarization (for non-convex Pareto fronts)
 *
 * s(w, r) = max_i { w_i * |r_i - ref_i| }
 *
 * @param weights - Weight vector
 * @param rewards - Reward vector
 * @param reference - Reference point (ideal point)
 * @returns Scalar reward (to be minimized for Chebyshev)
 */
export function chebyshevScalarize(
  weights: Float32Array,
  rewards: Float32Array,
  reference: Float32Array,
): number {
  let maxVal = -Infinity

  for (let i = 0; i < weights.length; i++) {
    const diff = Math.abs(rewards[i]! - reference[i]!)
    const weighted = weights[i]! * diff
    if (weighted > maxVal) {
      maxVal = weighted
    }
  }

  return maxVal
}

// ==================== Weight Utilities ====================

/**
 * Sample a weight vector uniformly from the (n-1)-simplex
 *
 * Uses Dirichlet(1,1,...,1) distribution which is uniform on the simplex.
 *
 * @param dim - Dimensionality (number of objectives)
 * @returns Weight vector that sums to 1
 */
export function sampleSimplex(dim: number): Float32Array {
  const weights = new Float32Array(dim)
  let sum = 0

  // Sample from Exponential(1) = -ln(U) where U ~ Uniform(0,1)
  for (let i = 0; i < dim; i++) {
    weights[i] = -Math.log(1 - Math.random() + 1e-10)
    sum += weights[i]!
  }

  // Normalize to sum to 1
  for (let i = 0; i < dim; i++) {
    weights[i] = weights[i]! / sum
  }

  return weights
}

/**
 * Normalize a weight vector to sum to 1
 *
 * @param weights - Unnormalized weights
 * @returns New normalized weight vector
 */
export function normalizeWeights(weights: Float32Array): Float32Array {
  const sum = weights.reduce((a, b) => a + b, 0)
  const result = new Float32Array(weights.length)

  for (let i = 0; i < weights.length; i++) {
    result[i] = weights[i]! / sum
  }

  return result
}

/**
 * Create uniform weight vector
 *
 * @param dim - Dimensionality
 * @returns Weight vector with equal weights
 */
export function uniformWeights(dim: number): Float32Array {
  const weights = new Float32Array(dim)
  const value = 1 / dim

  for (let i = 0; i < dim; i++) {
    weights[i] = value
  }

  return weights
}

/**
 * Create one-hot weight vector (focus on single objective)
 *
 * @param dim - Dimensionality
 * @param index - Which objective to focus on
 * @returns Weight vector with 1 at index, 0 elsewhere
 */
export function oneHotWeights(dim: number, index: number): Float32Array {
  const weights = new Float32Array(dim)
  weights[index] = 1
  return weights
}

/**
 * Generate a grid of weight vectors for systematic exploration
 *
 * @param dim - Dimensionality
 * @param resolution - Number of steps per dimension
 * @returns Array of weight vectors covering the simplex
 */
export function weightGrid(dim: number, resolution: number): Float32Array[] {
  if (dim === 1) {
    return [new Float32Array([1])]
  }

  if (dim === 2) {
    const result: Float32Array[] = []
    for (let i = 0; i <= resolution; i++) {
      const w1 = i / resolution
      const w2 = 1 - w1
      result.push(new Float32Array([w1, w2]))
    }
    return result
  }

  // General case: recursive generation for dim >= 3
  const result: Float32Array[] = []
  generateWeightGrid(dim, resolution, [], 0, result)
  return result
}

/**
 * Recursive helper for weight grid generation
 */
function generateWeightGrid(
  dim: number,
  resolution: number,
  current: number[],
  sumSoFar: number,
  result: Float32Array[],
): void {
  if (current.length === dim - 1) {
    // Last dimension is determined by constraint sum = 1
    const lastWeight = 1 - sumSoFar
    if (lastWeight >= 0) {
      result.push(new Float32Array([...current, lastWeight]))
    }
    return
  }

  const remaining = 1 - sumSoFar
  const steps = Math.floor(remaining * resolution)

  for (let i = 0; i <= steps; i++) {
    const weight = i / resolution
    if (sumSoFar + weight <= 1 + 1e-9) {
      generateWeightGrid(dim, resolution, [...current, weight], sumSoFar + weight, result)
    }
  }
}

// ==================== Pareto Utilities ====================

/**
 * Check if solution a dominates solution b
 *
 * a dominates b if a is at least as good in all objectives
 * and strictly better in at least one.
 *
 * @param a - First solution's objectives
 * @param b - Second solution's objectives
 * @param maximize - Whether to maximize objectives (default: true)
 * @returns True if a dominates b
 */
export function dominates(
  a: Float32Array,
  b: Float32Array,
  maximize: boolean = true,
): boolean {
  let atLeastAsGood = true
  let strictlyBetter = false

  for (let i = 0; i < a.length; i++) {
    if (maximize) {
      if (a[i]! < b[i]!) atLeastAsGood = false
      if (a[i]! > b[i]!) strictlyBetter = true
    } else {
      if (a[i]! > b[i]!) atLeastAsGood = false
      if (a[i]! < b[i]!) strictlyBetter = true
    }
  }

  return atLeastAsGood && strictlyBetter
}

/**
 * Compute hypervolume indicator for a Pareto front
 *
 * Simple 2D implementation. For higher dimensions, consider
 * using specialized hypervolume algorithms.
 *
 * @param front - Array of non-dominated solutions
 * @param reference - Reference point (should be dominated by all solutions)
 * @returns Hypervolume value
 */
export function hypervolume2D(
  front: Float32Array[],
  reference: Float32Array,
): number {
  if (front.length === 0) return 0
  if (front[0]!.length !== 2) {
    throw new Error('hypervolume2D only supports 2D fronts')
  }

  // Sort by first objective descending
  const sorted = [...front].sort((a, b) => b[0]! - a[0]!)

  let volume = 0
  let prevY = reference[1]!

  for (const point of sorted) {
    const width = point[0]! - reference[0]!
    const height = point[1]! - prevY

    if (width > 0 && height > 0) {
      volume += width * height
    }

    prevY = Math.max(prevY, point[1]!)
  }

  return volume
}
