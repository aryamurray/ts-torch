/**
 * DQN Agent Implementation
 *
 * Deep Q-Network agent with Double DQN by default.
 * Supports both single-objective and multi-objective (MORL) learning
 * based on the strategy configuration.
 *
 * Features:
 * - Double DQN (uses target net for action selection, online for value)
 * - Target network with configurable sync frequency
 * - Strategy-based exploration (epsilon-greedy or envelope Q-learning)
 * - Automatic weight concatenation for MORL
 *
 * @example
 * ```ts
 * import { dqn } from '@ts-torch/rl'
 * import { nn } from '@ts-torch/nn'
 * import { device } from '@ts-torch/core'
 *
 * const agent = dqn({
 *   device: device.cuda(0),
 *   model: nn.sequence(4, nn.fc(64).relu(), nn.fc(2)),
 *   optimizer: { lr: 1e-3 },
 *   gamma: 0.99,
 *   targetUpdateFreq: 1000
 * })
 * ```
 */

import type { Tensor, DeviceType } from '@ts-torch/core'
import { run } from '@ts-torch/core'
import type { DeviceContext } from '@ts-torch/core'
import type { SequenceDef } from '@ts-torch/nn'
import type { Module } from '@ts-torch/nn'
import { Adam } from '@ts-torch/optim'
import type { Optimizer } from '@ts-torch/optim'
import type { Agent, MOAgent, AgentConfig, TrainStepResult } from './base.js'
import type { TransitionBatch } from '../replay-buffer.js'
import { EpsilonGreedyStrategy } from '../strategies/epsilon-greedy.js'
import { EnvelopeQStrategy } from '../strategies/envelope.js'
import { conditionObservation } from '../utils/morl.js'
import {
  saveCheckpoint,
  loadCheckpoint,
  type AgentStateDict,
  type TensorData,
} from '../checkpoint.js'

// ==================== Types ====================

/**
 * DQN agent configuration
 */
export interface DQNAgentConfig {
  /** Device to run on */
  device: DeviceContext<DeviceType>

  /** Model architecture definition */
  model: SequenceDef

  /** Optimizer configuration */
  optimizer?: {
    /** Learning rate (default: 1e-3) */
    lr: number
    /** Weight decay (default: 0) */
    weightDecay?: number
  }

  /** Discount factor (default: 0.99) */
  gamma?: number

  /** Target network update frequency in steps (default: 1000) */
  targetUpdateFreq?: number

  /** Number of actions (required if not in model) */
  actionSpace?: number

  /**
   * For MORL: number of reward dimensions
   * If > 1, agent operates in multi-objective mode
   */
  rewardDim?: number
}

// ==================== Implementation ====================

/**
 * Deep Q-Network Agent
 *
 * Implements Double DQN with target network for stable learning.
 */
export class DQNAgent implements Agent, MOAgent {
  // Networks
  private qNetwork: Module<any, any, any, DeviceType>
  private targetNetwork: Module<any, any, any, DeviceType>

  // Training components
  private optimizer: Optimizer
  private readonly gamma: number
  private readonly targetUpdateFreq: number

  // State
  private stepCount_: number = 0
  private trainingMode: boolean = true

  // Configuration
  private readonly deviceContext: DeviceContext<DeviceType>
  private readonly actionSpace_: number
  private readonly rewardDim_: number
  private readonly config_: AgentConfig

  // Exploration (set externally by trainer)
  private explorationStrategy: EpsilonGreedyStrategy | EnvelopeQStrategy | null = null

  constructor(config: DQNAgentConfig) {
    this.deviceContext = config.device
    this.gamma = config.gamma ?? 0.99
    this.targetUpdateFreq = config.targetUpdateFreq ?? 1000
    this.rewardDim_ = config.rewardDim ?? 1

    // Initialize Q-network from model definition
    this.qNetwork = config.model.init(config.device)

    // Clone architecture for target network
    this.targetNetwork = config.model.init(config.device)

    // Copy weights from Q to target
    this.syncTarget()

    // Create optimizer
    const lr = config.optimizer?.lr ?? 1e-3
    this.optimizer = new Adam(this.qNetwork.parameters() as unknown as Tensor[], { lr })

    // Infer action space from model output
    // The model output should be [actionSpace] or [actionSpace * rewardDim] for MORL
    this.actionSpace_ = config.actionSpace ?? this.inferActionSpace(config.model)

    // Store config for introspection
    this.config_ = {
      device: config.device.type,
      gamma: this.gamma,
      learningRate: lr,
      targetUpdateFreq: this.targetUpdateFreq,
    }
  }

  /**
   * Select an action given an observation
   */
  act(observation: Float32Array, explore: boolean): number {
    if (!this.explorationStrategy) {
      // No strategy set, always greedy
      return this.greedyAction(observation)
    }

    if (this.explorationStrategy instanceof EpsilonGreedyStrategy) {
      if (explore && Math.random() < this.explorationStrategy.currentEpsilon) {
        return Math.floor(Math.random() * this.actionSpace_)
      }
      return this.greedyAction(observation)
    }

    // For EnvelopeQStrategy, use current weights
    if (this.explorationStrategy instanceof EnvelopeQStrategy) {
      const weights = this.explorationStrategy.getWeights()
      return this.actWithWeights(observation, weights, explore)
    }

    return this.greedyAction(observation)
  }

  /**
   * Select action with explicit weight vector (for MORL)
   */
  actWithWeights(observation: Float32Array, weights: Float32Array, explore: boolean): number {
    if (!(this.explorationStrategy instanceof EnvelopeQStrategy)) {
      throw new Error('actWithWeights requires EnvelopeQStrategy')
    }

    if (explore && Math.random() < this.explorationStrategy.currentEpsilon) {
      return Math.floor(Math.random() * this.actionSpace_)
    }

    // Get Q-values from network (conditioned on weights)
    const conditionedObs = conditionObservation(observation, weights)
    const qValues = this.getQValues(conditionedObs)

    return this.explorationStrategy.selectGreedy(qValues, weights, this.actionSpace_)
  }

  /**
   * Perform a training step
   *
   * Returns loss and optionally TD-errors for prioritized replay updates.
   */
  trainStep(batch: TransitionBatch): TrainStepResult {
    this.stepCount_++
    let lossValue = 0
    let tdErrors: Float32Array | undefined

    run(() => {
      this.optimizer.zeroGrad()

      // Compute Q-values for current states
      const stateTensor = this.createTensor(batch.states, [batch.batchSize, batch.stateSize])
      const qValuesAll = (this.qNetwork as any).forward(stateTensor)

      // Get Q-values for taken actions
      const qValues = this.gatherActions(qValuesAll, batch.actions, batch.batchSize)

      // Compute target Q-values using Double DQN
      const nextStateTensor = this.createTensor(batch.nextStates, [batch.batchSize, batch.stateSize])

      // Double DQN: use online network to select actions, target network for values
      const nextQOnline = (this.qNetwork as any).forward(nextStateTensor)
      const bestActions = this.argmaxBatch(nextQOnline, batch.batchSize)

      const nextQTarget = (this.targetNetwork as any).forward(nextStateTensor)
      const nextQValues = this.gatherActions(nextQTarget, bestActions, batch.batchSize)

      // Compute TD targets: r + gamma * Q_target(s', argmax_a Q_online(s', a))
      const { targets, errors } = this.computeTargetsWithErrors(
        batch.rewards,
        nextQValues,
        batch.dones,
        batch.batchSize,
        qValues,
      )

      // Store TD-errors for PER updates
      tdErrors = errors

      // Apply importance sampling weights if provided (for PER)
      let loss: Tensor
      if (batch.weights) {
        loss = this.weightedMseLoss(qValues, targets, batch.weights)
      } else {
        loss = this.mseLoss(qValues, targets)
      }

      lossValue = this.extractScalar(loss)

      // Backward pass
      ;(loss as any).backward()

      // Optimizer step
      this.optimizer.step()
    })

    // Periodic target network sync
    if (this.stepCount_ % this.targetUpdateFreq === 0) {
      this.syncTarget()
    }

    return { loss: lossValue, tdErrors }
  }

  /**
   * Sync target network with online network
   */
  syncTarget(): void {
    const onlineParams = this.qNetwork.parameters()
    const targetParams = this.targetNetwork.parameters()

    for (let i = 0; i < onlineParams.length; i++) {
      const onlineData = (onlineParams[i] as any).data
      const targetData = (targetParams[i] as any).data

      // Copy data from online to target
      if (onlineData && targetData && typeof targetData.copy === 'function') {
        targetData.copy(onlineData)
      }
    }
  }

  /**
   * Get trainable parameters
   */
  parameters(): Tensor[] {
    return this.qNetwork.parameters() as unknown as Tensor[]
  }

  /**
   * Set to training mode
   */
  train(): void {
    this.trainingMode = true
    this.qNetwork.train()
  }

  /**
   * Set to evaluation mode
   */
  eval(): void {
    this.trainingMode = false
    this.qNetwork.eval()
  }

  /**
   * Set exploration strategy (called by trainer)
   */
  setStrategy(strategy: EpsilonGreedyStrategy | EnvelopeQStrategy): void {
    this.explorationStrategy = strategy
  }

  /**
   * Number of training steps
   */
  get stepCount(): number {
    return this.stepCount_
  }

  /**
   * Agent configuration
   */
  get config(): AgentConfig {
    return this.config_
  }

  /**
   * Number of objectives (for MORL)
   */
  get numObjectives(): number {
    return this.rewardDim_
  }

  /**
   * Action space size
   */
  get actionSpace(): number {
    return this.actionSpace_
  }

  /**
   * Whether agent is in training mode
   */
  get isTraining(): boolean {
    return this.trainingMode
  }

  /**
   * Get the optimizer used by this agent
   */
  getOptimizer(): Optimizer {
    return this.optimizer
  }

  // ==================== Checkpointing Methods ====================

  /**
   * Save agent state to file
   */
  async save(path: string): Promise<void> {
    const state = this.stateDict()
    await saveCheckpoint(path, {
      tensors: {
        ...this.flattenStateDict(state.model, 'model'),
        ...this.flattenStateDict(state.targetModel ?? {}, 'target'),
      },
      metadata: state.metadata,
    })
  }

  /**
   * Load agent state from file
   */
  async load(path: string): Promise<void> {
    const checkpoint = await loadCheckpoint(path)

    // Reconstruct state dict from flat tensors
    const model: Record<string, TensorData> = {}
    const targetModel: Record<string, TensorData> = {}

    for (const [key, tensor] of Object.entries(checkpoint.tensors)) {
      if (key.startsWith('model.')) {
        model[key.slice(6)] = tensor
      } else if (key.startsWith('target.')) {
        targetModel[key.slice(7)] = tensor
      }
    }

    const state: AgentStateDict = {
      model,
      targetModel: Object.keys(targetModel).length > 0 ? targetModel : undefined,
      metadata: checkpoint.metadata as AgentStateDict['metadata'],
    }

    this.loadStateDict(state)
  }

  /**
   * Get serializable state dictionary
   */
  stateDict(): AgentStateDict {
    const model = this.extractModuleState(this.qNetwork)
    const targetModel = this.extractModuleState(this.targetNetwork)

    return {
      model,
      targetModel,
      metadata: {
        stepCount: this.stepCount_,
        version: '1.0.0',
        config: this.config_,
      },
    }
  }

  /**
   * Load state from dictionary
   */
  loadStateDict(state: AgentStateDict): void {
    // Load model weights
    this.loadModuleState(this.qNetwork, state.model)

    // Load target model weights
    if (state.targetModel) {
      this.loadModuleState(this.targetNetwork, state.targetModel)
    }

    // Restore metadata
    if (state.metadata?.stepCount !== undefined) {
      this.stepCount_ = state.metadata.stepCount
    }
  }

  // ==================== Private Methods ====================

  /**
   * Get greedy action from Q-values
   */
  private greedyAction(observation: Float32Array): number {
    const qValues = this.getQValues(observation)
    return this.argmax(qValues)
  }

  /**
   * Get Q-values for observation
   */
  private getQValues(observation: Float32Array): Float32Array {
    let result: Float32Array = new Float32Array(this.actionSpace_)

    run(() => {
      const input = this.createTensor(observation, [1, observation.length])
      const output = (this.qNetwork as any).forward(input)
      result = this.extractArray(output)
    })

    return result
  }

  /**
   * Create a tensor on the agent's device
   */
  private createTensor(data: Float32Array, shape: number[]): Tensor {
    return (this.deviceContext as any).tensor(data, shape)
  }

  /**
   * Extract scalar value from tensor
   */
  private extractScalar(tensor: Tensor): number {
    if (typeof (tensor as any).item === 'function') {
      return (tensor as any).item()
    }
    if (typeof (tensor as any).toArray === 'function') {
      const arr = (tensor as any).toArray()
      return arr[0] ?? 0
    }
    return 0
  }

  /**
   * Extract array from tensor
   */
  private extractArray(tensor: Tensor): Float32Array {
    if (typeof (tensor as any).toArray === 'function') {
      const arr = (tensor as any).toArray()
      return arr instanceof Float32Array ? arr : new Float32Array(arr)
    }
    return new Float32Array(0)
  }

  /**
   * Gather Q-values for specific actions
   */
  private gatherActions(qValues: Tensor, actions: Int32Array | number[], batchSize: number): Tensor {
    // This is a simplified implementation - in production would use tensor gather
    const gathered = new Float32Array(batchSize)
    const qArray = this.extractArray(qValues)
    const numActions = this.actionSpace_

    for (let i = 0; i < batchSize; i++) {
      const action = typeof actions[i] === 'number' ? actions[i]! : actions[i]!
      gathered[i] = qArray[i * numActions + action]!
    }

    return this.createTensor(gathered, [batchSize])
  }

  /**
   * Argmax over batch
   */
  private argmaxBatch(qValues: Tensor, batchSize: number): number[] {
    const qArray = this.extractArray(qValues)
    const numActions = this.actionSpace_
    const result: number[] = []

    for (let i = 0; i < batchSize; i++) {
      let maxIdx = 0
      let maxVal = qArray[i * numActions]!

      for (let a = 1; a < numActions; a++) {
        const val = qArray[i * numActions + a]!
        if (val > maxVal) {
          maxVal = val
          maxIdx = a
        }
      }
      result.push(maxIdx)
    }

    return result
  }

  /**
   * Compute TD targets and TD-errors
   */
  private computeTargetsWithErrors(
    rewards: Float32Array,
    nextQValues: Tensor,
    dones: Uint8Array,
    batchSize: number,
    currentQValues: Tensor,
  ): { targets: Tensor; errors: Float32Array } {
    const nextQ = this.extractArray(nextQValues)
    const currentQ = this.extractArray(currentQValues)
    const targets = new Float32Array(batchSize)
    const errors = new Float32Array(batchSize)

    for (let i = 0; i < batchSize; i++) {
      const done = dones[i] === 1
      const target = rewards[i]! + (done ? 0 : this.gamma * nextQ[i]!)
      targets[i] = target
      errors[i] = Math.abs(target - currentQ[i]!)
    }

    return {
      targets: this.createTensor(targets, [batchSize]),
      errors,
    }
  }

  /**
   * MSE loss computation
   */
  private mseLoss(predictions: Tensor, targets: Tensor): Tensor {
    // Use tensor operations if available, otherwise fallback
    if (typeof (predictions as any).sub === 'function') {
      const diff = (predictions as any).sub(targets)
      const squared = (diff as any).mul(diff)
      return (squared as any).mean()
    }

    // Fallback: manual MSE
    const preds = this.extractArray(predictions)
    const targs = this.extractArray(targets)
    let sum = 0

    for (let i = 0; i < preds.length; i++) {
      const diff = preds[i]! - targs[i]!
      sum += diff * diff
    }

    return this.createTensor(new Float32Array([sum / preds.length]), [1])
  }

  /**
   * Argmax helper
   */
  private argmax(values: Float32Array): number {
    let maxIdx = 0
    let maxVal = values[0]!

    for (let i = 1; i < values.length; i++) {
      if (values[i]! > maxVal) {
        maxVal = values[i]!
        maxIdx = i
      }
    }

    return maxIdx
  }

  /**
   * Infer action space from model definition
   */
  private inferActionSpace(model: SequenceDef): number {
    const blocks = model.blocks
    if (blocks.length === 0) {
      throw new Error('Model must have at least one block to infer action space')
    }
    const lastBlock = blocks[blocks.length - 1]!
    return lastBlock.outFeatures
  }

  /**
   * Weighted MSE loss for prioritized replay
   */
  private weightedMseLoss(predictions: Tensor, targets: Tensor, weights: Float32Array): Tensor {
    const preds = this.extractArray(predictions)
    const targs = this.extractArray(targets)
    let sum = 0

    for (let i = 0; i < preds.length; i++) {
      const diff = preds[i]! - targs[i]!
      sum += weights[i]! * diff * diff
    }

    return this.createTensor(new Float32Array([sum / preds.length]), [1])
  }

  /**
   * Flatten state dict with prefix
   */
  private flattenStateDict(
    state: Record<string, TensorData>,
    prefix: string,
  ): Record<string, TensorData> {
    const result: Record<string, TensorData> = {}
    for (const [key, value] of Object.entries(state)) {
      result[`${prefix}.${key}`] = value
    }
    return result
  }

  /**
   * Extract module state as TensorData records
   */
  private extractModuleState(module: Module<any, any, any, DeviceType>): Record<string, TensorData> {
    const state: Record<string, TensorData> = {}
    const namedParams = module.namedParameters()

    for (const [name, param] of namedParams) {
      const tensor = (param as any).data
      if (tensor && typeof tensor.toArray === 'function') {
        const data = tensor.toArray()
        const shape = tensor.shape ?? [data.length]
        state[name] = {
          data: data instanceof Float32Array ? data : new Float32Array(data),
          shape: Array.isArray(shape) ? shape : [shape],
          dtype: 'float32',
        }
      }
    }

    return state
  }

  /**
   * Load state into module parameters
   */
  private loadModuleState(
    module: Module<any, any, any, DeviceType>,
    state: Record<string, TensorData>,
  ): void {
    const namedParams = module.namedParameters()

    for (const [name, param] of namedParams) {
      const tensorData = state[name]
      if (tensorData) {
        const paramData = (param as any).data
        if (paramData && typeof paramData.copy === 'function') {
          // Create tensor from loaded data and copy
          const loadedTensor = this.createTensor(
            tensorData.data as Float32Array,
            tensorData.shape,
          )
          paramData.copy(loadedTensor)
        }
      }
    }
  }
}

// ==================== Factory ====================

/**
 * Create a DQN agent
 *
 * @param config - Agent configuration
 * @returns DQN agent instance
 *
 * @example
 * ```ts
 * const agent = dqn({
 *   device: device.cuda(0),
 *   model: nn.sequence(4,
 *     nn.fc(128).relu(),
 *     nn.fc(64).relu(),
 *     nn.fc(2)
 *   ),
 *   gamma: 0.99,
 *   targetUpdateFreq: 1000
 * })
 * ```
 */
export function dqn(config: DQNAgentConfig): DQNAgent {
  return new DQNAgent(config)
}
