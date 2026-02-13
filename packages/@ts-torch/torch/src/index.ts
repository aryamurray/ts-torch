// Re-export everything from core at the top level
export * from '@ts-torch/core'

// NN: re-export builder directly (avoids nn.nn double-nesting)
export { nn } from '@ts-torch/nn'

// RL: re-export builder namespaces directly (avoids rl.RL double-nesting)
export { RL, MORL } from '@ts-torch/rl'

// These have no builder namespace objects — no double-nesting risk
export * as optim from '@ts-torch/optim'
export * as datasets from '@ts-torch/datasets'
export * as train from '@ts-torch/train'

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
  // Callbacks (declarative)
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
  CallbackCheckpointData,
  BestModelData,
  // Callbacks (class-based)
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
