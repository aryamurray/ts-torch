/**
 * RL Policies
 *
 * Neural network policies for RL algorithms.
 *
 * @example
 * ```ts
 * import { actorCriticPolicy } from './policies'
 *
 * const policy = actorCriticPolicy({
 *   netArch: { pi: [64, 64], vf: [64, 64] },
 *   activation: 'tanh',
 * }).init(device.cuda(0), { observationSize: 4, actionSpace: discrete(2) })
 * ```
 */

export {
  ActorCriticPolicy,
  actorCriticPolicy,
  mlpPolicy,
} from './actor-critic.js'

export type {
  ActorCriticPolicyConfig,
  ActorCriticPolicyDef,
  PolicySpaces,
  NetArch,
  PolicyActivation,
  ForwardResult,
  EvaluateActionsResult,
} from './actor-critic.js'

// SAC Policy
export { SACPolicy, sacPolicy } from './sac-policy.js'
export type {
  SACPolicyConfig,
  SACPolicyDef,
  SACPolicySpaces,
  SACNetArch,
} from './sac-policy.js'
