/**
 * Base Distribution Interface
 *
 * All action distributions implement this interface for sampling,
 * log probability computation, and entropy calculation.
 */

import type { Tensor, Shape } from '@ts-torch/core'

/**
 * Base interface for action distributions
 *
 * @template ActionShape - Shape of actions (e.g., [batch] for discrete, [batch, actionDim] for continuous)
 */
export interface Distribution<ActionShape extends Shape = Shape> {
  /**
   * Sample actions from the distribution
   * @returns Tensor of sampled actions
   */
  sample(): Tensor<ActionShape>

  /**
   * Get the mode (most likely action) - deterministic action selection
   * @returns Tensor of deterministic actions
   */
  mode(): Tensor<ActionShape>

  /**
   * Compute log probability of actions
   * @param actions - Actions to evaluate
   * @returns Log probabilities for each action
   */
  logProb(actions: Tensor<ActionShape>): Tensor<readonly [number]>

  /**
   * Compute entropy of the distribution
   * @returns Entropy for each sample in batch
   */
  entropy(): Tensor<readonly [number]>
}

/**
 * Get actions either stochastically (sample) or deterministically (mode)
 */
export function getActions<S extends Shape>(
  dist: Distribution<S>,
  deterministic: boolean,
): Tensor<S> {
  return deterministic ? dist.mode() : dist.sample()
}
