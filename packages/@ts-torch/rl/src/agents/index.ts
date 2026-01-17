/**
 * RL Agents
 *
 * This module provides agent implementations for reinforcement learning.
 */

// Base types (legacy)
export type { Agent, MOAgent, AgentConfig, TrainStepResult } from './base.js'
export { isMOAgent } from './base.js'

// Base algorithm (new SB3-style)
export { BaseAlgorithm } from './base-algorithm.js'
export type { Schedule, LearnConfig, BaseAlgorithmConfig } from './base-algorithm.js'

// On-policy base
export { OnPolicyAlgorithm } from './on-policy-base.js'
export type { OnPolicyConfig } from './on-policy-base.js'

// PPO
export { PPO, ppo } from './ppo.js'
export type { PPOConfig, PPODef, PPOAgent } from './ppo.js'

// A2C
export { A2C, a2c } from './a2c.js'
export type { A2CConfig, A2CDef } from './a2c.js'

// Off-policy base
export { OffPolicyAlgorithm } from './off-policy-base.js'
export type { OffPolicyConfig } from './off-policy-base.js'

// SAC
export { SAC, sac } from './sac.js'
export type { SACConfig, SACDef } from './sac.js'

// DQN Agent (legacy)
export { DQNAgent, dqn } from './dqn.js'
export type { DQNAgentConfig } from './dqn.js'
