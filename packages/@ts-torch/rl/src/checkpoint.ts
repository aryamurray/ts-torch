/**
 * RL Agent Checkpointing
 *
 * Re-exports checkpoint utilities from @ts-torch/nn and adds
 * RL-specific types like AgentStateDict.
 *
 * @example
 * ```ts
 * import { saveCheckpoint, loadCheckpoint, AgentStateDict } from '@ts-torch/rl'
 *
 * // Save agent state
 * const state: AgentStateDict = agent.stateDict()
 * await saveCheckpoint('./agent.ckpt', {
 *   tensors: { ...flattenStateDict(state.model, 'model') },
 *   metadata: state.metadata
 * })
 *
 * // Load agent state
 * const checkpoint = await loadCheckpoint('./agent.ckpt')
 * ```
 */

// Re-export all checkpoint functionality from @ts-torch/nn
export {
  saveCheckpoint,
  loadCheckpoint,
  encodeCheckpoint,
  decodeCheckpoint,
  float32Tensor,
  paramsToTensors,
  type TensorData,
  type CheckpointData,
  type StateDict,
} from '@ts-torch/nn'

// ==================== RL-Specific Types ====================

/**
 * Agent state dict (convenience type for agent.stateDict())
 *
 * This extends the base StateDict concept with RL-specific fields
 * like target networks (for DQN) and training metadata.
 */
export interface AgentStateDict {
  /** Model parameters */
  model: Record<string, import('@ts-torch/nn').TensorData>
  /** Target network parameters (for DQN variants) */
  targetModel?: Record<string, import('@ts-torch/nn').TensorData>
  /** Optimizer state (optional) */
  optimizer?: Record<string, unknown>
  /** Training metadata */
  metadata: {
    /** Number of training steps completed */
    stepCount: number
    /** Agent version string */
    version: string
    /** Additional metadata */
    [key: string]: unknown
  }
}
