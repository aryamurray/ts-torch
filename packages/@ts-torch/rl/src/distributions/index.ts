/**
 * Action Distributions
 *
 * Probability distributions for action sampling in RL policies.
 * Used by actor-critic and policy gradient methods.
 *
 * @example
 * ```ts
 * // Discrete actions
 * const catDist = categorical(logits)
 * const action = catDist.sample()
 *
 * // Continuous actions
 * const gaussDist = diagGaussian(mean, logStd)
 * const action = gaussDist.sample()
 * ```
 */

export type { Distribution } from './base.js'
export { getActions } from './base.js'

export { CategoricalDistribution, categorical } from './categorical.js'
export { DiagGaussianDistribution, diagGaussian } from './diagonal-gaussian.js'
