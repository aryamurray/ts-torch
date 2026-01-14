/**
 * Vectorized Environments
 *
 * Run multiple environment instances for efficient rollout collection.
 *
 * @example
 * ```ts
 * import { RL } from '@ts-torch/rl'
 *
 * // Declarative API - single config object
 * const vecEnv = RL.vecEnv({
 *   env: RL.envs.CartPole(),
 *   nEnvs: 8,
 *   type: 'dummy'  // or 'subproc' for parallel (future)
 * })
 *
 * // Reset all environments
 * const obs = vecEnv.reset()  // [8 * obsSize]
 *
 * // Step all environments
 * const actions = new Int32Array([0, 1, 0, 1, 0, 1, 0, 1])
 * const { observations, rewards, dones, infos } = vecEnv.step(actions)
 * ```
 */

import type { EnvConfig } from '../environment.js'
import { FunctionalEnv } from '../environment.js'
import type { Space } from '../spaces/index.js'
import { DummyVecEnv } from './dummy-vec-env.js'

// Re-export types and utilities from base
export type {
  VecEnv,
  VecEnvStepResult,
  EnvInfo,
} from './base.js'

export {
  isDiscreteActionSpace,
  isContinuousActionSpace,
  getSpaceFlatSize,
  getNumActions,
} from './base.js'

// Re-export class and legacy factory for advanced usage
export { DummyVecEnv, dummyVecEnv } from './dummy-vec-env.js'

// ==================== Declarative Config ====================

/**
 * Environment input - can be config, instance, or factory
 */
export type EnvInput<S> = 
  | EnvConfig<S>                    // Raw config object
  | FunctionalEnv<S>                // Already instantiated env
  | (() => FunctionalEnv<S>)        // Factory function

/**
 * Vectorized environment configuration
 */
export interface VecEnvConfig<S = any> {
  /** 
   * Environment configuration, instance, or factory
   * 
   * @example
   * ```ts
   * // Using built-in environment (returns FunctionalEnv)
   * env: RL.envs.CartPole()
   * 
   * // Using raw config
   * env: { createState: () => {...}, step: () => {...}, observe: () => {...} }
   * 
   * // Using factory (for environments with per-instance randomness)
   * env: () => RL.envs.CartPole()
   * ```
   */
  env: EnvInput<S>
  
  /** Number of parallel environments */
  nEnvs: number
  
  /** 
   * Type of vectorization
   * - 'dummy': Sequential execution in main thread (default)
   * - 'subproc': Parallel execution with worker threads (future)
   */
  type?: 'dummy' | 'subproc'
  
  /** Reward dimensionality (default: 1) */
  rewardDim?: number
  
  /** 
   * Action space override.
   * If not provided, defaults to discrete(actionSpace) from the environment.
   */
  actionSpace?: Space
}

// ==================== Factory Function ====================

/**
 * Create a vectorized environment from configuration
 *
 * @param config - Vectorized environment configuration
 * @returns VecEnv instance
 *
 * @example
 * ```ts
 * // Declarative style with built-in env
 * const vecEnv = vecEnv({
 *   env: RL.envs.CartPole(),
 *   nEnvs: 8,
 *   type: 'dummy'
 * })
 * 
 * // With factory for per-instance randomness
 * const vecEnv = vecEnv({
 *   env: () => RL.envs.CartPole(),
 *   nEnvs: 8
 * })
 * ```
 */
export function vecEnv<S>(config: VecEnvConfig<S>): DummyVecEnv<S> {
  const { env: envInput, nEnvs, type = 'dummy', rewardDim, actionSpace } = config
  
  if (type === 'subproc') {
    // Future: implement SubprocVecEnv with worker threads
    console.warn('subproc not yet implemented, falling back to dummy')
  }
  
  // Normalize env input to a factory function
  const envFactory = normalizeEnvInput(envInput)
  
  return new DummyVecEnv(envFactory, { nEnvs, rewardDim, actionSpace })
}

/**
 * Normalize various env input types to a factory function
 */
function normalizeEnvInput<S>(input: EnvInput<S>): () => FunctionalEnv<S> {
  // If it's already a factory function
  if (typeof input === 'function') {
    return input as () => FunctionalEnv<S>
  }
  
  // If it's a FunctionalEnv instance, create a factory from its config
  if (input instanceof FunctionalEnv) {
    // Get the config from the instance and create new instances
    const config = input.config
    return () => new FunctionalEnv(config)
  }
  
  // If it's a raw EnvConfig, wrap it
  if ('createState' in input && 'step' in input && 'observe' in input) {
    return () => new FunctionalEnv(input as EnvConfig<S>)
  }
  
  throw new Error('Invalid env input: must be EnvConfig, FunctionalEnv, or factory function')
}


