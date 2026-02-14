/**
 * On-Policy Algorithm Base
 *
 * Base class for on-policy algorithms (PPO, A2C).
 * Handles rollout collection and GAE advantage computation.
 */

import type { DeviceType } from '@ts-torch/core'
import { Logger } from '@ts-torch/core'
import type { Optimizer } from '@ts-torch/optim'
import { Adam } from '@ts-torch/optim'
import { BaseAlgorithm } from './base-algorithm.js'
import type { Schedule, BaseAlgorithmConfig } from './base-algorithm.js'
import { RolloutBuffer } from '../buffers/rollout-buffer.js'
import type { ActorCriticPolicy, ActorCriticPolicyConfig } from '../policies/index.js'
import { actorCriticPolicy } from '../policies/index.js'
import type { StepData, EpisodeEndData } from '../callbacks/types.js'
import type { VecEnvStepResult } from '../vec-env/base.js'

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
   * Invokes onStep and onEpisodeEnd callbacks.
   * Uses shared-memory mode when the VecEnv supports stepInto().
   * @returns True to continue training
   */
  protected collectRollouts(): boolean {
    this.rolloutBuffer.reset()
    this.policy.train()

    const useSharedObs = typeof this.env.stepInto === 'function'
    const isDiscrete = this.env.actionSpace.type === 'discrete'

    if (useSharedObs) {
      return this.collectRolloutsShared(isDiscrete)
    }
    return this.collectRolloutsCopy(isDiscrete)
  }

  /**
   * Shared-memory rollout collection: env writes obs directly into rollout buffer.
   * Eliminates one obs copy per step.
   */
  private collectRolloutsShared(isDiscrete: boolean): boolean {
    const env = this.env as Required<Pick<typeof this.env, 'stepInto'>> & typeof this.env

    // Copy initial observations into buffer slot 0
    this.rolloutBuffer.getObsWriteTarget(0).set(this.lastObs)

    // Track current obs view in the buffer
    let obsRef = this.rolloutBuffer.getObsWriteTarget(0)

    for (let step = 0; step < this.nSteps; step++) {
      // Policy reads observations from the buffer directly
      const { actions, values, logProbs } = this.policy.forward(obsRef, false)

      const envActions = isDiscrete
        ? Int32Array.from(actions, a => Math.round(a))
        : actions

      // Determine where env should write the NEXT observations:
      // - For steps 0..nSteps-2: into the next buffer slot
      // - For the last step: into this.lastObs (used for GAE bootstrap)
      const nextObsTarget = step < this.nSteps - 1
        ? this.rolloutBuffer.getObsWriteTarget(step + 1)
        : this.lastObs

      // Env writes obs directly into the target buffer
      const stepResult = env.stepInto(envActions, nextObsTarget)

      // Update timesteps
      this.numTimesteps += this.env.nEnvs

      // Track episode rewards/lengths and handle callbacks
      if (!this.handleStepTracking(stepResult, envActions, obsRef)) {
        return false
      }

      // Store everything except observations (already in the buffer)
      this.rolloutBuffer.addWithoutObs(
        actions,
        stepResult.rewards,
        this.lastEpisodeStarts,
        values,
        logProbs,
      )

      // Update episode starts
      if (this.lastEpisodeStarts.length === stepResult.dones.length) {
        this.lastEpisodeStarts.set(stepResult.dones)
      } else {
        this.lastEpisodeStarts = new Uint8Array(stepResult.dones)
      }

      // Advance obs reference to next slot
      obsRef = nextObsTarget
    }

    // this.lastObs now contains the final observations (written by env on last step)
    const lastValues = this.policy.predictValues(this.lastObs)
    this.rolloutBuffer.computeReturnsAndAdvantage(lastValues, this.lastEpisodeStarts)

    return true
  }

  /**
   * Standard rollout collection with obs copies (fallback when stepInto unavailable).
   */
  private collectRolloutsCopy(isDiscrete: boolean): boolean {
    for (let step = 0; step < this.nSteps; step++) {
      const { actions, values, logProbs } = this.policy.forward(this.lastObs, false)

      const envActions = isDiscrete
        ? Int32Array.from(actions, a => Math.round(a))
        : actions

      const stepResult = this.env.step(envActions)

      // Update timesteps
      this.numTimesteps += this.env.nEnvs

      // Track episode rewards/lengths and handle callbacks
      if (!this.handleStepTracking(stepResult, envActions)) {
        return false
      }

      // Store in buffer (copies observations)
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
      if (this.lastEpisodeStarts.length === stepResult.dones.length) {
        this.lastEpisodeStarts.set(stepResult.dones)
      } else {
        this.lastEpisodeStarts = new Uint8Array(stepResult.dones)
      }
    }

    const lastValues = this.policy.predictValues(this.lastObs)
    this.rolloutBuffer.computeReturnsAndAdvantage(lastValues, this.lastEpisodeStarts)

    return true
  }

  /**
   * Handle per-step tracking, episode completion, and callbacks.
   * @returns true to continue training, false to stop
   */
  private handleStepTracking(
    stepResult: VecEnvStepResult,
    envActions: Int32Array | Float32Array,
    currentObs?: Float32Array,
  ): boolean {
    const episodeRewards = this.currentEpisodeRewards
    const episodeLengths = this.currentEpisodeLengths
    if (episodeRewards && episodeLengths) {
      for (let i = 0; i < this.env.nEnvs; i++) {
        episodeRewards[i] = (episodeRewards[i] ?? 0) + (stepResult.rewards[i] ?? 0)
        episodeLengths[i] = (episodeLengths[i] ?? 0) + 1
      }
    }

    // Invoke onStep callback
    if (this.callbacks?.onStep) {
      const stepData: StepData = {
        timestep: this.numTimesteps,
        observations: currentObs ?? this.lastObs,
        actions: envActions,
        rewards: stepResult.rewards,
        dones: Array.from(stepResult.dones, d => d === 1),
        infos: stepResult.infos ?? [],
      }
      const continueTraining = this.callbacks.onStep(stepData)
      if (continueTraining === false) {
        return false
      }
    }

    // Check for episode completions and invoke onEpisodeEnd
    for (let i = 0; i < this.env.nEnvs; i++) {
      if (stepResult.dones[i] === 1) {
        this.episodesCompleted++

        if (this.callbacks?.onEpisodeEnd && this.currentEpisodeRewards && this.currentEpisodeLengths) {
          const episodeEndData: EpisodeEndData = {
            envIndex: i,
            episodeReward: this.currentEpisodeRewards[i]!,
            episodeLength: this.currentEpisodeLengths[i]!,
            timestep: this.numTimesteps,
            info: stepResult.infos?.[i] ?? {},
          }
          const continueTraining = this.callbacks.onEpisodeEnd(episodeEndData)
          if (continueTraining === false) {
            return false
          }
        }

        // Reset episode tracking for this env
        if (this.currentEpisodeRewards && this.currentEpisodeLengths) {
          this.currentEpisodeRewards[i] = 0
          this.currentEpisodeLengths[i] = 0
        }
      }
    }

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
   * Predict actions for a batch of observations
   *
   * @param observations - Flat [nObs * obsSize] observation array
   * @param deterministic - Whether to use deterministic policy
   * @param out - Optional pre-allocated output buffer to avoid allocation
   * @returns Action array (Int32Array for discrete, Float32Array for continuous)
   */
  predictBatch(
    observations: Float32Array,
    deterministic: boolean = false,
    out?: Int32Array | Float32Array,
  ): Int32Array | Float32Array {
    this.policy.eval()
    const { actions } = this.policy.forward(observations, deterministic)

    if (this.env.actionSpace.type === 'discrete') {
      const nObs = observations.length / this.env.observationSize
      const result = out instanceof Int32Array && out.length >= nObs ? out : new Int32Array(nObs)
      for (let i = 0; i < nObs; i++) {
        result[i] = Math.round(actions[i]!)
      }
      return result
    }

    if (out instanceof Float32Array && out.length >= actions.length) {
      out.set(actions)
      return out
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
    Logger.warn('save() not yet fully implemented')
  }
}
