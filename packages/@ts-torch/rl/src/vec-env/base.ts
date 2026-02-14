/**
 * Vectorized Environment Base
 *
 * Interface for running multiple environment instances in parallel.
 * Used by on-policy algorithms (PPO, A2C) for efficient rollout collection.
 *
 * Key Design:
 * - Batched operations: All envs step together
 * - Auto-reset: Environments automatically reset when done
 * - Numpy-style outputs: Returns batched Float32Arrays
 *
 * @example
 * ```ts
 * const vecEnv = new DummyVecEnv(() => RL.env.CartPole(), { nEnvs: 8 })
 * const obs = vecEnv.reset()  // [8, obsSize]
 * const { obs, rewards, dones, infos } = vecEnv.step(actions)
 * ```
 */

import type { Space, DiscreteSpace, BoxSpace } from '../spaces/index.js'

// ==================== Types ====================

/**
 * Info dictionary for a single environment
 */
export interface EnvInfo {
  /** Whether this was a terminal state (not truncation) */
  terminal?: boolean
  /** Whether this was truncated (time limit) */
  truncated?: boolean
  /** Final observation before auto-reset (for proper value bootstrapping) */
  terminalObservation?: Float32Array
  /** Episode length at termination */
  episodeLength?: number
  /** Episode reward at termination */
  episodeReward?: number
  /** Any custom info */
  [key: string]: unknown
}

/**
 * Result of stepping all environments
 */
export interface VecEnvStepResult {
  /** Observations from all envs [nEnvs, ...obsShape] as flat array */
  observations: Float32Array
  /** Rewards from all envs [nEnvs] or [nEnvs, rewardDim] */
  rewards: Float32Array
  /** Done flags [nEnvs] */
  dones: Uint8Array
  /** Info dictionaries for each env */
  infos: EnvInfo[]
}

/**
 * Base interface for vectorized environments
 */
export interface VecEnv {
  /** Number of environments */
  readonly nEnvs: number

  /** Observation space (all envs share the same) */
  readonly observationSpace: Space

  /** Action space (all envs share the same) */
  readonly actionSpace: Space

  /** Observation size (flat) */
  readonly observationSize: number

  /** Action dimension */
  readonly actionDim: number

  /** Reward dimensionality (1 for scalar, >1 for multi-objective) */
  readonly rewardDim: number

  /**
   * Reset all environments
   * @returns Initial observations [nEnvs, obsSize] as flat array
   */
  reset(): Float32Array

  /**
   * Step all environments with given actions
   *
   * @param actions - Actions for each env. Can be:
   *   - Int32Array/Float32Array of length nEnvs (discrete)
   *   - Float32Array of length nEnvs * actionDim (continuous)
   * @returns Step results with observations, rewards, dones, infos
   */
  step(actions: Int32Array | Float32Array): VecEnvStepResult

  /**
   * Get current observations without stepping
   * @returns Current observations [nEnvs, obsSize]
   */
  getObservations(): Float32Array

  /**
   * Step all environments, writing observations into a caller-provided buffer.
   *
   * Used for shared-memory mode: the rollout buffer provides a write target
   * and the env writes observations directly into it, eliminating one copy.
   * The returned observations field points to obsTarget.
   *
   * @param actions - Actions for each env
   * @param obsTarget - Buffer to write observations into [nEnvs * obsSize]
   * @returns Step results (observations field points to obsTarget)
   */
  stepInto?(actions: Int32Array | Float32Array, obsTarget: Float32Array): VecEnvStepResult

  /**
   * Close all environments and free resources
   */
  close(): void

  /**
   * Seed all environments (optional)
   * @param seeds - Array of seeds, one per env
   */
  seed?(seeds: number[]): void
}

// ==================== Utilities ====================

/**
 * Check if space is discrete
 */
export function isDiscreteActionSpace(space: Space): space is DiscreteSpace {
  return space.type === 'discrete'
}

/**
 * Check if space is continuous (box)
 */
export function isContinuousActionSpace(space: Space): space is BoxSpace {
  return space.type === 'box'
}

/**
 * Get flat size from space
 */
export function getSpaceFlatSize(space: Space): number {
  if (space.type === 'discrete') {
    return 1
  }
  return space.shape.reduce((a, b) => a * b, 1)
}

/**
 * Get number of actions from action space
 */
export function getNumActions(space: Space): number {
  if (space.type === 'discrete') {
    return space.n
  }
  return space.shape.reduce((a, b) => a * b, 1)
}
