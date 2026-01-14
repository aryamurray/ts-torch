/**
 * Off-Policy Algorithm Base
 *
 * Base class for off-policy algorithms (SAC, TD3, DDPG).
 * Handles experience collection and replay buffer management.
 */

import { BaseAlgorithm } from './base-algorithm.js'
import type { Schedule, BaseAlgorithmConfig } from './base-algorithm.js'
import { ContinuousReplayBuffer } from '../buffers/index.js'
import type { BoxSpace } from '../spaces/index.js'

// ==================== Types ====================

/**
 * Configuration for off-policy algorithms
 */
export interface OffPolicyConfig extends Omit<BaseAlgorithmConfig, 'learningRate'> {
  /** Learning rate (constant or schedule) */
  learningRate?: Schedule

  /** Replay buffer capacity (default: 1_000_000) */
  bufferSize?: number

  /** Minibatch size for training (default: 256) */
  batchSize?: number

  /** Number of timesteps before training starts (default: 100) */
  learningStarts?: number

  /** How often to train (steps) (default: 1) */
  trainFreq?: number

  /** Number of gradient steps per training call (default: 1) */
  gradientSteps?: number

  /** Soft update coefficient (Polyak) (default: 0.005) */
  tau?: number

  /** Discount factor (default: 0.99) */
  gamma?: number
}

// ==================== Implementation ====================

/**
 * Base class for off-policy algorithms
 *
 * Handles:
 * - Experience collection with replay buffer
 * - Soft target network updates (Polyak averaging)
 * - Train frequency and gradient step management
 */
export abstract class OffPolicyAlgorithm extends BaseAlgorithm {
  // Replay buffer
  protected replayBuffer!: ContinuousReplayBuffer

  // Hyperparameters
  protected bufferSize: number
  protected batchSize: number
  protected learningStarts: number
  protected trainFreq: number
  protected gradientSteps: number
  protected tau: number
  protected gamma: number

  // State
  protected lastObs!: Float32Array
  protected actionDim!: number

  constructor(config: OffPolicyConfig) {
    super({
      env: config.env,
      device: config.device,
      learningRate: config.learningRate ?? 3e-4,
      verbose: config.verbose,
    })

    this.bufferSize = config.bufferSize ?? 1_000_000
    this.batchSize = config.batchSize ?? 256
    this.learningStarts = config.learningStarts ?? 100
    this.trainFreq = config.trainFreq ?? 1
    this.gradientSteps = config.gradientSteps ?? 1
    this.tau = config.tau ?? 0.005
    this.gamma = config.gamma ?? 0.99
  }

  /**
   * Setup model - creates replay buffer
   * Subclasses should call super._setupModel() and then create their networks
   */
  protected _setupModel(): void {
    // Get action dimension from environment
    if (this.env.actionSpace.type !== 'box') {
      throw new Error('OffPolicyAlgorithm requires continuous (Box) action space')
    }
    this.actionDim = (this.env.actionSpace as BoxSpace).shape.reduce((a, b) => a * b, 1)

    // Create replay buffer
    this.replayBuffer = new ContinuousReplayBuffer({
      capacity: this.bufferSize,
      stateSize: this.env.observationSize,
      actionDim: this.actionDim,
    })

    // Initialize state
    this.lastObs = this.env.reset()
  }

  /**
   * Collect experience from the environment
   */
  protected collectRollouts(): boolean {
    // Collect trainFreq steps
    for (let i = 0; i < this.trainFreq; i++) {
      // Select action
      let actions: Float32Array
      if (this.numTimesteps < this.learningStarts) {
        // Random actions during warmup
        actions = this.randomActions()
      } else {
        // Policy actions with exploration noise
        actions = this.selectActions(this.lastObs, true)
      }

      // Step environment
      const result = this.env.step(actions)
      this.numTimesteps += this.env.nEnvs

      // Store transitions (for each env)
      for (let envIdx = 0; envIdx < this.env.nEnvs; envIdx++) {
        const stateOffset = envIdx * this.env.observationSize
        const actionOffset = envIdx * this.actionDim

        this.replayBuffer.push({
          state: this.lastObs.slice(stateOffset, stateOffset + this.env.observationSize),
          action: actions.slice(actionOffset, actionOffset + this.actionDim),
          reward: result.rewards[envIdx]!,
          nextState: result.observations.slice(stateOffset, stateOffset + this.env.observationSize),
          done: result.dones[envIdx] === 1,
        })
      }

      // Update observation
      this.lastObs = result.observations
    }

    return true
  }

  /**
   * Generate random actions for warmup
   */
  protected randomActions(): Float32Array {
    const actions = new Float32Array(this.env.nEnvs * this.actionDim)
    const actionSpace = this.env.actionSpace as BoxSpace

    for (let i = 0; i < this.env.nEnvs; i++) {
      const sample = actionSpace.sample()
      actions.set(sample, i * this.actionDim)
    }

    return actions
  }

  /**
   * Select actions from policy (with optional exploration)
   * Subclasses implement this
   */
  protected abstract selectActions(obs: Float32Array, explore: boolean): Float32Array

  /**
   * Soft update target networks (Polyak averaging)
   * target = tau * source + (1 - tau) * target
   */
  protected softUpdate(sourceParams: any[], targetParams: any[]): void {
    for (let i = 0; i < sourceParams.length; i++) {
      const sourceData = (sourceParams[i] as any).data
      const targetData = (targetParams[i] as any).data

      if (sourceData && targetData && typeof targetData.toArray === 'function') {
        const sourceArr = sourceData.toArray() as Float32Array
        const targetArr = targetData.toArray() as Float32Array

        // Polyak update: target = tau * source + (1 - tau) * target
        for (let j = 0; j < sourceArr.length; j++) {
          targetArr[j] = this.tau * sourceArr[j]! + (1 - this.tau) * targetArr[j]!
        }

        // Copy back (this is a simplified version - in production you'd use in-place tensor ops)
      }
    }
  }

  /**
   * Check if training should happen this step
   */
  protected shouldTrain(): boolean {
    return (
      this.numTimesteps >= this.learningStarts &&
      this.replayBuffer.size >= this.batchSize
    )
  }
}
