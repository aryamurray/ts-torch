// Re-export everything from core at the top level
export * from '@ts-torch/core'

// Export sub-modules as namespaces
export * as nn from '@ts-torch/nn'
export * as optim from '@ts-torch/optim'
export * as datasets from '@ts-torch/datasets'
export * as train from '@ts-torch/train'
export * as rl from '@ts-torch/rl'

// ==================== Common Type Re-exports ====================
// These allow direct type imports: import type { VecEnv, PPOAgent } from '@ts-torch/torch'

// RL Types
export type {
  // Environments
  VecEnv,
  VecEnvStepResult,
  EnvConfig,
  StepResult,
  // Spaces
  Space,
  DiscreteSpace,
  BoxSpace,
  // Agents
  PPOAgent,
  PPOConfig,
  PPODef,
  A2CConfig,
  A2CDef,
  SACConfig,
  SACDef,
  // Policies
  ActorCriticPolicyConfig,
  NetArch,
  // Training
  LearnConfig,
  Schedule,
  // Callbacks
  BaseCallback,
} from '@ts-torch/rl'

// NN Types
export type {
  Module,
  Sequential,
  BlockDef,
  SequenceDef,
  StateDict,
  CheckpointData,
} from '@ts-torch/nn'

// Optim Types
export type {
  Optimizer,
  SGDOptions,
  AdamOptions,
} from '@ts-torch/optim'
