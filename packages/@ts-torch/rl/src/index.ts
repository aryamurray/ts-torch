/**
 * @ts-torch/rl - Declarative Reinforcement Learning
 *
 * A declarative, high-performance API for Reinforcement Learning.
 * Inspired by Stable Baselines 3, adapted to TypeScript's declarative patterns.
 *
 * Design Philosophy:
 * - Declarative Logic: Define environments and agents via configuration objects
 * - SB3-style API: agent.learn() for training
 * - Vectorized Environments: Run multiple envs in parallel
 * - On-policy & Off-policy: PPO, A2C, DQN, SAC
 *
 * @example
 * ```ts
 * import { RL } from '@ts-torch/rl'
 * import { device } from '@ts-torch/core'
 *
 * // 1. Create vectorized environment
 * const vecEnv = RL.vecEnv.dummy(() => RL.envs.CartPole(), { nEnvs: 8 })
 *
 * // 2. Create PPO agent
 * const ppo = RL.ppo({
 *   policy: { netArch: { pi: [64, 64], vf: [64, 64] } },
 *   learningRate: 3e-4,
 *   nSteps: 2048,
 *   batchSize: 64,
 *   nEpochs: 10,
 * }).init(device.cuda(0), vecEnv)
 *
 * // 3. Train (SB3-style)
 * await ppo.learn({ totalTimesteps: 1_000_000 })
 *
 * // 4. Inference
 * const action = ppo.predict(observation, true)  // deterministic
 * ```
 */

// ==================== Spaces ====================

export { discrete, box, boxUniform, isDiscreteSpace, isBoxSpace, getSpaceSize, getActionDim } from './spaces/index.js'
export type { DiscreteSpace, BoxSpace, BoxConfig, Space } from './spaces/index.js'

// ==================== Distributions ====================

export { CategoricalDistribution, categorical, DiagGaussianDistribution, diagGaussian, getActions } from './distributions/index.js'
export type { Distribution } from './distributions/index.js'

// ==================== Environment ====================

export { env, FunctionalEnv } from './environment.js'
export type { EnvConfig, StepResult } from './environment.js'

// ==================== Vectorized Environments ====================

export { DummyVecEnv, vecEnv, dummyVecEnv, isDiscreteActionSpace, isContinuousActionSpace, getSpaceFlatSize, getNumActions } from './vec-env/index.js'
export type { VecEnv, VecEnvStepResult, EnvInfo, VecEnvConfig } from './vec-env/index.js'

// ==================== Buffers ====================

export { ReplayBuffer } from './replay-buffer.js'
export type { Transition, TransitionBatch, PERConfig } from './replay-buffer.js'

export { RolloutBuffer } from './buffers/index.js'
export type { RolloutBufferConfig, RolloutBufferSamples } from './buffers/index.js'

// ==================== Policies ====================

export { ActorCriticPolicy, actorCriticPolicy, mlpPolicy, SACPolicy, sacPolicy } from './policies/index.js'
export type { ActorCriticPolicyConfig, ActorCriticPolicyDef, PolicySpaces, NetArch, PolicyActivation, ForwardResult, EvaluateActionsResult, SACPolicyConfig, SACPolicyDef, SACPolicySpaces, SACNetArch } from './policies/index.js'

// ==================== Callbacks ====================

export { BaseCallback, CallbackList, StopTrainingCallback, EpisodeTrackingCallback, callbackList, maybeCallback } from './callbacks/index.js'
export type {
  CallbackLocals,
  BaseAlgorithmRef,
  MetricsLogger,
  // Declarative callback types
  Callbacks,
  TrainingStartData,
  TrainingEndData,
  StepData,
  EpisodeStartData,
  EpisodeEndData,
  RolloutStartData,
  RolloutEndData,
  EvalStartData,
  EvalEndData,
  CheckpointData as CallbackCheckpointData,
  BestModelData,
} from './callbacks/index.js'

// ==================== Agents ====================

// Base classes (new SB3-style)
export { BaseAlgorithm, OnPolicyAlgorithm, OffPolicyAlgorithm, PPO, ppo, A2C, a2c, SAC, sac } from './agents/index.js'
export type { Schedule, LearnConfig, BaseAlgorithmConfig, OnPolicyConfig, OffPolicyConfig, PPOConfig, PPODef, PPOAgent, A2CConfig, A2CDef, SACConfig, SACDef } from './agents/index.js'

// Legacy agents
export { DQNAgent, dqn, isMOAgent } from './agents/index.js'
export type { Agent, MOAgent, AgentConfig, TrainStepResult, DQNAgentConfig } from './agents/index.js'

// ==================== Strategies ====================

export {
  EpsilonGreedyStrategy,
  epsilonGreedy,
  EnvelopeQStrategy,
  envelopeQ,
  createStrategy,
} from './strategies/index.js'
export type {
  EpsilonGreedyConfig,
  EnvelopeConfig,
  ExplorationStrategyConfig,
  ExplorationStrategy,
} from './strategies/index.js'

// ==================== Checkpointing ====================

// Re-export from @ts-torch/nn for convenience
export {
  saveSafetensors,
  loadSafetensors,
  encodeSafetensors,
  decodeSafetensors,
  serializeMetadata,
  deserializeMetadata,
} from '@ts-torch/nn'
export type { TensorData, StateDict } from '@ts-torch/nn'

// RL-specific checkpoint types
export type { AgentStateDict } from './checkpoint.js'

// ==================== Utils ====================

export {
  SparseGraph,
  SumTree,
  scalarize,
  chebyshevScalarize,
  sampleSimplex,
  normalizeWeights,
  uniformWeights,
  oneHotWeights,
  weightGrid,
  dominates,
  hypervolume2D,
  conditionObservation,
} from './utils/index.js'
export type { AdjacencyEntry, NeighborView } from './utils/index.js'

// ==================== Built-in Environments ====================

export { CartPole, CartPoleRaw } from './envs/index.js'
export type { CartPoleState } from './envs/index.js'

// ==================== Main Namespaces ====================

import { env } from './environment.js'
import { dqn, ppo, a2c, sac } from './agents/index.js'
import { vecEnv } from './vec-env/index.js'
import { discrete, box } from './spaces/index.js'
import { actorCriticPolicy, mlpPolicy, sacPolicy } from './policies/index.js'
import { scalarize, sampleSimplex, normalizeWeights, uniformWeights, weightGrid } from './utils/index.js'
import { CartPole, CartPoleRaw } from './envs/index.js'

/**
 * Main RL namespace - declarative reinforcement learning API
 *
 * @example
 * ```ts
 * // Declarative API (recommended)
 * const vecEnv = RL.vecEnv({
 *   env: RL.envs.CartPole(),
 *   nEnvs: 8,
 *   type: 'dummy'
 * })
 *
 * // Create PPO agent
 * const agent = RL.ppo({
 *   policy: { netArch: { pi: [64, 64], vf: [64, 64] } },
 * }).init(device.cuda(0), vecEnv)
 *
 * // Train
 * await agent.learn({ totalTimesteps: 1_000_000 })
 * ```
 */
export const RL = {
  /**
   * Create a functional environment from configuration
   * @see env
   */
  env,

  /**
   * Create a vectorized environment (declarative API)
   * 
   * @example
   * ```ts
   * const vecEnv = RL.vecEnv({
   *   env: RL.envs.CartPole(),
   *   nEnvs: 8,
   *   type: 'dummy'  // or 'subproc' for parallel
   * })
   * ```
   */
  vecEnv,

  /**
   * Action and observation spaces
   */
  spaces: {
    /**
     * Create a discrete space
     * @see discrete
     */
    discrete,
    /**
     * Create a box (continuous) space
     * @see box
     */
    box,
  },

  /**
   * Policy builders
   */
  policy: {
    /**
     * Create an actor-critic policy
     * @see actorCriticPolicy
     */
    actorCritic: actorCriticPolicy,
    /**
     * Create an MLP policy
     * @see mlpPolicy
     */
    mlp: mlpPolicy,
    /**
     * Create a SAC policy
     * @see sacPolicy
     */
    sac: sacPolicy,
  },

  /**
   * Built-in environments
   */
  envs: {
    CartPole,
    CartPoleRaw,
  },

  /**
   * Create a PPO agent
   * @see ppo
   */
  ppo,

  /**
   * Create an A2C agent
   * @see a2c
   */
  a2c,

  /**
   * Create a SAC agent (continuous control)
   * @see sac
   */
  sac,

  /**
   * Create a DQN agent (legacy)
   * @see dqn
   */
  dqn,

  /**
   * Scalarize multi-objective reward with weights
   * @see scalarize
   */
  scalarize,
}

/**
 * Multi-Objective RL namespace - utilities for Pareto optimization
 *
 * @example
 * ```ts
 * // Sample weights from simplex
 * const weights = MORL.sampleSimplex(3)  // 3 objectives
 *
 * // Scalarize reward
 * const scalar = RL.scalarize(weights, rewards)
 *
 * // Generate weight grid for evaluation
 * const grid = MORL.weightGrid(3, 10)
 * ```
 */
export const MORL = {
  /**
   * Sample weight vector uniformly from simplex
   * @see sampleSimplex
   */
  sampleSimplex,

  /**
   * Normalize weights to sum to 1
   * @see normalizeWeights
   */
  normalizeWeights,

  /**
   * Create uniform weight vector
   * @see uniformWeights
   */
  uniformWeights,

  /**
   * Generate grid of weights for systematic evaluation
   * @see weightGrid
   */
  weightGrid,
}
