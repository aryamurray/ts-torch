/**
 * Multi-Objective RL Utilities
 *
 * Shared utilities for multi-objective reinforcement learning.
 */

/**
 * Concatenate observation with weight vector for conditioned network
 *
 * In MORL, we often condition the Q-network on a weight vector that
 * specifies the trade-off between objectives. This function concatenates
 * the observation with the weights to create a conditioned input.
 *
 * @param observation - State observation
 * @param weights - Weight vector for objectives (sums to 1)
 * @returns Concatenated [observation, weights] as Float32Array
 *
 * @example
 * ```ts
 * const obs = new Float32Array([0.5, 0.3, 0.1, 0.2]) // 4-dim state
 * const weights = new Float32Array([0.6, 0.4])       // 2 objectives
 * const conditioned = conditionObservation(obs, weights)
 * // conditioned is [0.5, 0.3, 0.1, 0.2, 0.6, 0.4] (6-dim)
 * ```
 */
export function conditionObservation(
  observation: Float32Array,
  weights: Float32Array,
): Float32Array {
  const conditioned = new Float32Array(observation.length + weights.length)
  conditioned.set(observation, 0)
  conditioned.set(weights, observation.length)
  return conditioned
}
