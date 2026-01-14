/**
 * On-Policy Algorithm Base
 *
 * Base class for on-policy algorithms (PPO, A2C).
 * Handles rollout collection and GAE advantage computation.
 */

import type { DeviceType } from '@ts-torch/core'
import type { Optimizer } from '@ts-torch/optim'
import { Adam } from '@ts-torch/optim'
import { BaseAlgorithm } from './base-algorithm.js'
import type { Schedule, BaseAlgorithmConfig } from './base-algorithm.js'
import { RolloutBuffer } from '../buffers/rollout-buffer.js'
import type { ActorCriticPolicy, ActorCriticPolicyConfig } from '../policies/index.js'
import { actorCriticPolicy } from '../policies/index.js'

// ==================== Types ====================

/**
 * Configuration for on-policy algorithms
 */
export interface OnPolicyConfig extends Omit<BaseAlgorithmConfig, 'learningRate'> {
  /** Policy configuration or 'MlpPolicy' shorthand */
  policy: ActorCriticPolicyConfig | 'MlpPolicy'

  /** Learning rate (constant or schedule) */
  learningRate?: Schedule

  /** Steps per environment per rollout (default: 2048) */
  nSteps?: number

  /** Discount factor (default: 0.99) */
  gamma?: number

  /** GAE lambda (default: 0.95) */
  gaeLambda?: number

  /** Entropy coefficient for loss (default: 0.0) */
  entCoef?: number

  /** Value function coefficient for loss (default: 0.5) */
  vfCoef?: number

  /** Max gradient norm for clipping (default: 0.5) */
  maxGradNorm?: number
}

// ==================== Implementation ====================

/**
 * Base class for on-policy algorithms
 *
 * Handles:
 * - Rollout collection
 * - GAE advantage computation
 * - Policy and value network management
 */
export abstract class OnPolicyAlgorithm extends BaseAlgorithm {
  // Policy
  protected policy!: ActorCriticPolicy<DeviceType>
  protected policyConfig: ActorCriticPolicyConfig | 'MlpPolicy'

  // Rollout buffer
  protected rolloutBuffer!: RolloutBuffer

  // Hyperparameters
  protected nSteps: number
  protected gamma: number
  protected gaeLambda: number
  protected entCoef: number
  protected vfCoef: number
  protected maxGradNorm: number

  // Optimizer
  protected optimizer!: Optimizer

  // State
  protected lastObs!: Float32Array
  protected lastEpisodeStarts!: Uint8Array

  constructor(config: OnPolicyConfig) {
    super({
      env: config.env,
      device: config.device,
      learningRate: config.learningRate ?? 3e-4,
      verbose: config.verbose,
    })

    this.policyConfig = config.policy
    this.nSteps = config.nSteps ?? 2048
    this.gamma = config.gamma ?? 0.99
    this.gaeLambda = config.gaeLambda ?? 0.95
    this.entCoef = config.entCoef ?? 0.0
    this.vfCoef = config.vfCoef ?? 0.5
    this.maxGradNorm = config.maxGradNorm ?? 0.5
  }

  /**
   * Setup model, policy, optimizer, and buffer
   */
  protected _setupModel(): void {
    // Create policy
    const policyDef = this.policyConfig === 'MlpPolicy'
      ? actorCriticPolicy({ netArch: { pi: [64, 64], vf: [64, 64] }, activation: 'tanh' })
      : actorCriticPolicy(this.policyConfig)

    // Determine action space
    const actionSpace = this.env.actionSpace.type === 'discrete'
      ? this.env.actionSpace
      : this.env.actionSpace

    this.policy = policyDef.init(this.device_, {
      observationSize: this.env.observationSize,
      actionSpace,
    })

    // Create rollout buffer
    const actionDim = this.env.actionSpace.type === 'discrete' ? 1 : this.env.actionDim

    this.rolloutBuffer = new RolloutBuffer({
      bufferSize: this.nSteps,
      nEnvs: this.env.nEnvs,
      observationSize: this.env.observationSize,
      actionDim,
      gamma: this.gamma,
      gaeLambda: this.gaeLambda,
    })

    // Create optimizer
    this.optimizer = new Adam(this.policy.parameters(), {
      lr: this.getCurrentLr(),
    })

    // Initialize state
    this.lastObs = this.env.reset()
    this.lastEpisodeStarts = new Uint8Array(this.env.nEnvs).fill(1) // All envs start fresh
  }

  /**
   * Collect rollouts from the environment
   *
   * Fills the rollout buffer with experience.
   * @returns True to continue training
   */
  protected collectRollouts(): boolean {
    this.rolloutBuffer.reset()
    this.policy.train()

    for (let step = 0; step < this.nSteps; step++) {
      // Get action, value, log_prob from policy
      const { actions, values, logProbs } = this.policy.forward(this.lastObs, false)

      // Convert actions for environment (discrete = single int per env)
      const envActions = this.env.actionSpace.type === 'discrete'
        ? Int32Array.from(actions, a => Math.round(a))
        : actions

      // Step environment
      const stepResult = this.env.step(envActions)

      // Update timesteps
      this.numTimesteps += this.env.nEnvs

      // Store in buffer
      this.rolloutBuffer.add(
        this.lastObs,
        actions,
        stepResult.rewards,
        this.lastEpisodeStarts,
        values,
        logProbs,
      )

      // Update state for next step
      this.lastObs = stepResult.observations
      this.lastEpisodeStarts = stepResult.dones
    }

    // Compute returns and advantages
    const lastValues = this.policy.predictValues(this.lastObs)
    this.rolloutBuffer.computeReturnsAndAdvantage(lastValues, this.lastEpisodeStarts)

    return true
  }

  /**
   * Update learning rate in optimizer
   */
  protected updateLearningRate(): void {
    // Get current LR from schedule
    void this.getCurrentLr()
    // Adam doesn't have a setLr method in the current implementation
    // This would need to be added to the optimizer base class
    // For now, we'll skip LR updates or implement a workaround
  }

  /**
   * Predict action for a single observation
   */
  predict(observation: Float32Array, deterministic: boolean = false): number | Float32Array {
    this.policy.eval()
    const { actions } = this.policy.forward(observation, deterministic)

    if (this.env.actionSpace.type === 'discrete') {
      return Math.round(actions[0]!)
    }
    return actions
  }

  /**
   * Get trainable parameters
   */
  parameters(): any[] {
    return this.policy.parameters()
  }

  /**
   * Set to training mode
   */
  train(): void {
    this.policy.train()
  }

  /**
   * Set to evaluation mode
   */
  eval(): void {
    this.policy.eval()
  }

  /**
   * Save algorithm state
   */
  async save(_path: string): Promise<void> {
    // TODO: Implement checkpointing
    console.warn('save() not yet fully implemented')
  }
}
