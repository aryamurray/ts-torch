/**
 * Experience Buffers
 *
 * Buffers for storing and sampling experience in RL algorithms.
 *
 * - RolloutBuffer: For on-policy algorithms (PPO, A2C) - stores trajectories with GAE
 * - ReplayBuffer: For off-policy algorithms (DQN) - discrete actions
 * - ContinuousReplayBuffer: For off-policy continuous control (SAC, TD3)
 */

export { RolloutBuffer } from './rollout-buffer.js'
export type { RolloutBufferConfig, RolloutBufferSamples } from './rollout-buffer.js'

export { ContinuousReplayBuffer } from './continuous-replay-buffer.js'
export type {
  ContinuousReplayBufferConfig,
  ContinuousTransition,
  ContinuousTransitionBatch,
} from './continuous-replay-buffer.js'
