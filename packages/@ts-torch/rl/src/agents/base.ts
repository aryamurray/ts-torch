/**
 * Agent Base Interface
 *
 * Defines the contract that all RL agents must implement.
 * Agents handle action selection and learning from experience.
 */

import type { Tensor } from '@ts-torch/core'
import type { Optimizer } from '@ts-torch/optim'
import type { TransitionBatch } from '../replay-buffer.js'
import type { AgentStateDict } from '../checkpoint.js'

// ==================== Types ====================

/**
 * Training step result
 */
export interface TrainStepResult {
  /** Loss value for this training step */
  loss: number
  /** Optional additional metrics */
  metrics?: Record<string, number>
  /** TD-errors for priority updates (optional, for PER) */
  tdErrors?: Float32Array
}

/**
 * Agent interface - all RL agents implement this
 */
export interface Agent {
  /**
   * Select an action given an observation
   *
   * @param observation - State observation as Float32Array
   * @param explore - Whether to use exploration (true) or greedy (false)
   * @returns Selected action index
   */
  act(observation: Float32Array, explore: boolean): number

  /**
   * Perform a single training step on a batch of transitions
   *
   * @param batch - Batch of transitions from replay buffer
   * @returns Training result with loss (and optionally TD-errors for PER)
   */
  trainStep(batch: TransitionBatch): TrainStepResult

  /**
   * Synchronize target network with online network (for DQN variants)
   * No-op for agents without target networks.
   */
  syncTarget(): void

  /**
   * Get all trainable parameters
   *
   * @returns Array of parameter tensors
   */
  parameters(): Tensor[]

  /**
   * Set the agent to training mode
   */
  train(): void

  /**
   * Set the agent to evaluation mode
   */
  eval(): void

  /**
   * Save agent state to file
   *
   * @param path - File path to save to
   */
  save(path: string): Promise<void>

  /**
   * Load agent state from file
   *
   * @param path - File path to load from
   */
  load(path: string): Promise<void>

  /**
   * Get serializable state dictionary
   *
   * @returns Agent state dict for checkpointing
   */
  stateDict(): AgentStateDict

  /**
   * Load state from dictionary
   *
   * @param state - State dict to load
   */
  loadStateDict(state: AgentStateDict): void

  /**
   * Get the optimizer used by this agent
   *
   * @returns The optimizer instance, or null if not available
   */
  getOptimizer(): Optimizer | null

  /**
   * Number of training steps performed
   */
  readonly stepCount: number

  /**
   * Agent configuration/metadata
   */
  readonly config: AgentConfig
}

/**
 * Base agent configuration
 */
export interface AgentConfig {
  /** Device the agent runs on */
  device: string
  /** Discount factor gamma */
  gamma: number
  /** Learning rate */
  learningRate: number
  /** Target network update frequency (steps) */
  targetUpdateFreq: number
}

/**
 * Multi-objective agent interface (extends base Agent)
 */
export interface MOAgent extends Agent {
  /**
   * Select action with explicit weight vector for multi-objective
   *
   * @param observation - State observation
   * @param weights - Weight vector for objectives
   * @param explore - Whether to explore
   * @returns Selected action index
   */
  actWithWeights(observation: Float32Array, weights: Float32Array, explore: boolean): number

  /**
   * Number of objectives
   */
  readonly numObjectives: number
}

/**
 * Type guard for multi-objective agents
 */
export function isMOAgent(agent: Agent): agent is MOAgent {
  return 'actWithWeights' in agent && 'numObjectives' in agent
}
