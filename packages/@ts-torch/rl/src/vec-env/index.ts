/**
 * Vectorized Environments
 *
 * Run multiple environment instances for efficient rollout collection.
 *
 * @example
 * ```ts
 * import { dummyVecEnv } from './vec-env'
 * import { CartPole } from '../envs'
 *
 * // Create 8 parallel CartPole environments
 * const vecEnv = dummyVecEnv(() => CartPole(), { nEnvs: 8 })
 *
 * // Reset all environments
 * const obs = vecEnv.reset()  // [8 * obsSize]
 *
 * // Step all environments
 * const actions = new Int32Array([0, 1, 0, 1, 0, 1, 0, 1])
 * const { observations, rewards, dones, infos } = vecEnv.step(actions)
 * ```
 */

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

export { DummyVecEnv, dummyVecEnv } from './dummy-vec-env.js'
export type { DummyVecEnvConfig } from './dummy-vec-env.js'
