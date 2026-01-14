/**
 * Proximal Policy Optimization (PPO)
 *
 * Implementation of PPO with clipped surrogate objective.
 * The most popular and stable on-policy algorithm.
 *
 * Key features:
 * - Clipped surrogate objective for stable updates
 * - Multiple epochs of updates per rollout
 * - Value function clipping (optional)
 * - KL early stopping (optional)
 *
 * @example
 * ```ts
 * const ppo = RL.ppo({
 *   policy: { netArch: { pi: [64, 64], vf: [64, 64] } },
 *   learningRate: 3e-4,
 *   nSteps: 2048,
 *   batchSize: 64,
 *   nEpochs: 10,
 *   clipRange: 0.2,
 * }).init(device.cuda(0), vecEnv)
 *
 * await ppo.learn({ totalTimesteps: 1_000_000 })
 * ```
 */

import { run, device as deviceModule } from '@ts-torch/core'
import type { DeviceType } from '@ts-torch/core'
import type { DeviceContext } from '@ts-torch/core'
import { OnPolicyAlgorithm } from './on-policy-base.js'
import type { OnPolicyConfig } from './on-policy-base.js'
import type { Schedule } from './base-algorithm.js'
import type { VecEnv } from '../vec-env/index.js'
import type { ActorCriticPolicyConfig } from '../policies/index.js'

// CPU device for tensor creation
const cpu = deviceModule.cpu()

// ==================== Types ====================

/**
 * PPO-specific configuration
 */
export interface PPOConfig extends Omit<OnPolicyConfig, 'env' | 'device' | 'policy'> {
  /** Policy configuration or 'MlpPolicy' shorthand */
  policy?: ActorCriticPolicyConfig | 'MlpPolicy'
  /** Minibatch size for updates (default: 64) */
  batchSize?: number

  /** Number of epochs to update per rollout (default: 10) */
  nEpochs?: number

  /** PPO clip range for policy (default: 0.2) */
  clipRange?: Schedule

  /** Clip range for value function (null = disabled, default: null) */
  clipRangeVf?: Schedule | null

  /** Whether to normalize advantages (default: true) */
  normalizeAdvantage?: boolean

  /** Target KL divergence for early stopping (null = disabled, default: null) */
  targetKl?: number | null
}

/**
 * Full PPO configuration (after init)
 */
interface PPOFullConfig extends Omit<PPOConfig, 'policy'> {
  policy: ActorCriticPolicyConfig | 'MlpPolicy'
  env: VecEnv
  device: DeviceContext<DeviceType>
}

// ==================== Implementation ====================

/**
 * Proximal Policy Optimization (PPO) Algorithm
 *
 * Uses clipped surrogate objective for stable policy updates.
 */
export class PPO extends OnPolicyAlgorithm {
  // PPO-specific hyperparameters
  private batchSize: number
  private nEpochs: number
  private clipRange: Schedule
  private normalizeAdvantage: boolean
  private targetKl: number | null

  constructor(config: PPOFullConfig) {
    super(config)

    this.batchSize = config.batchSize ?? 64
    this.nEpochs = config.nEpochs ?? 10
    this.clipRange = config.clipRange ?? 0.2
    // Note: clipRangeVf not yet implemented
    this.normalizeAdvantage = config.normalizeAdvantage ?? true
    this.targetKl = config.targetKl ?? null
  }

  /**
   * Get current clip range value
   */
  private getCurrentClipRange(): number {
    if (typeof this.clipRange === 'number') {
      return this.clipRange
    }
    const progressRemaining = 1.0 - (this.numTimesteps / this.totalTimesteps)
    return this.clipRange(progressRemaining)
  }

  /**
   * Perform PPO update
   *
   * For each epoch:
   * - Iterate over minibatches from rollout buffer
   * - Compute clipped surrogate loss using tensor operations
   * - Compute value loss using tensor MSE
   * - Compute entropy bonus
   * - Backprop and update
   */
  protected _train(): void {
    const clipRange = this.getCurrentClipRange()

    // Track metrics
    let totalPolicyLoss = 0
    let totalValueLoss = 0
    let totalEntropyLoss = 0
    let totalApproxKl = 0
    let nUpdates = 0

    // Multiple epochs over the data
    for (let epoch = 0; epoch < this.nEpochs; epoch++) {
      // Iterate over minibatches
      for (const batch of this.rolloutBuffer.get(this.batchSize)) {
        const {
          observations,
          actions,
          oldLogProbs,
          advantages,
          returns,
        } = batch

        // Normalize advantages (done on CPU, creates normalized array)
        let normalizedAdvantages = advantages
        if (this.normalizeAdvantage && advantages.length > 1) {
          const mean = advantages.reduce((a, b) => a + b, 0) / advantages.length
          let variance = 0
          for (let i = 0; i < advantages.length; i++) {
            variance += (advantages[i]! - mean) ** 2
          }
          variance /= advantages.length
          const std = Math.sqrt(variance + 1e-8)

          normalizedAdvantages = new Float32Array(advantages.length)
          for (let i = 0; i < advantages.length; i++) {
            normalizedAdvantages[i] = (advantages[i]! - mean) / std
          }
        }

        // For tracking (computed after tensor ops)
        let policyLossValue = 0
        let valueLossValue = 0
        let entropyValue = 0
        let approxKlValue = 0

        // All tensor operations in single run() block for autograd
        run(() => {
          this.optimizer.zeroGrad()

          const obsSize = this.policy.getObservationSize()

          // Create observation tensor [batch, obsSize]
          const obsTensor = cpu.tensor(observations, [batch.batchSize, obsSize] as const)

          // Forward pass through policy networks
          const policyOutput = this.policy.getPolicyOutput(obsTensor)
          const valueTensor = this.policy.getValueOutput(obsTensor)

          // Create distribution from policy output
          const dist = this.policy.getDistribution(policyOutput, batch.batchSize)

          // Get log probs tensor for actions [batch, nActions] or [batch, actionDim]
          const logProbsAllTensor = dist.logProbsTensor()

          // Create tensors for old log probs, advantages, returns
          const oldLogProbsTensor = cpu.tensor(oldLogProbs, [batch.batchSize] as const)
          const advantagesTensor = cpu.tensor(normalizedAdvantages, [batch.batchSize] as const)
          const returnsTensor = cpu.tensor(returns, [batch.batchSize, 1] as const)

          // For discrete actions: select log_probs for taken actions using one-hot
          // For continuous: compute log_prob directly
          let logProbsTensor: any

          if (this.policy.isDiscreteAction()) {
            // Create one-hot encoding [batch, nActions]
            const nActions = (dist as any).numActions
            const oneHot = new Float32Array(batch.batchSize * nActions)
            for (let b = 0; b < batch.batchSize; b++) {
              const action = Math.round(actions[b]!)
              oneHot[b * nActions + action] = 1.0
            }
            const oneHotTensor = cpu.tensor(oneHot, [batch.batchSize, nActions] as const)

            // selected_log_probs = (log_probs_all * one_hot).sumDim(1)
            const maskedLogProbs = (logProbsAllTensor as any).mul(oneHotTensor)
            logProbsTensor = (maskedLogProbs as any).sumDim(1)
          } else {
            // For continuous: use logProbTensor which computes proper Gaussian log prob
            logProbsTensor = (dist as any).logProbTensor(actions)
          }

          // Compute ratio = exp(log_prob_new - log_prob_old)
          const logDiff = (logProbsTensor as any).sub(oldLogProbsTensor)
          const ratio = (logDiff as any).exp()

          // Compute surrogate losses
          // surr1 = ratio * advantage
          const surr1 = (ratio as any).mul(advantagesTensor)

          // surr2 = clamp(ratio, 1-eps, 1+eps) * advantage
          const clippedRatio = (ratio as any).clamp(1 - clipRange, 1 + clipRange)
          const surr2 = (clippedRatio as any).mul(advantagesTensor)

          // Policy loss = -min(surr1, surr2).mean()
          const minSurr = (surr1 as any).minimum(surr2)
          const policyLoss = (minSurr as any).mean().neg()

          // Value loss = MSE(values, returns)
          const valueLoss = (valueTensor as any).mseLoss(returnsTensor)

          // Entropy bonus (use distribution's mean entropy)
          const entropyLoss = dist.meanEntropyTensor().neg()

          // Total loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss
          const scaledValueLoss = (valueLoss as any).mulScalar(this.vfCoef)
          const scaledEntropyLoss = (entropyLoss as any).mulScalar(this.entCoef)
          const totalLoss = (policyLoss as any).add(scaledValueLoss).add(scaledEntropyLoss)

          // Backward pass - gradients flow through the computational graph
          ;(totalLoss as any).backward()

          // Gradient clipping
          if (this.maxGradNorm > 0) {
            this.clipGradients(this.maxGradNorm)
          }

          // Optimizer step
          this.optimizer.step()

          // Extract scalar values for logging
          policyLossValue = (policyLoss as any).item?.() ?? 0
          valueLossValue = (valueLoss as any).item?.() ?? 0
          entropyValue = -((entropyLoss as any).item?.() ?? 0)

          // Compute approx KL for early stopping
          // KL ≈ 0.5 * mean((log_prob_old - log_prob_new)^2)
          const klDiff = (oldLogProbsTensor as any).sub(logProbsTensor)
          const klSquared = (klDiff as any).mul(klDiff)
          approxKlValue = ((klSquared as any).mean().item?.() ?? 0) * 0.5
        })

        // Track metrics
        totalPolicyLoss += policyLossValue
        totalValueLoss += valueLossValue
        totalEntropyLoss += entropyValue
        totalApproxKl += approxKlValue
        nUpdates++

        // KL early stopping
        if (this.targetKl !== null && approxKlValue > 1.5 * this.targetKl) {
          if (this.verbose > 0) {
            console.log(`Early stopping at epoch ${epoch} due to KL divergence: ${approxKlValue.toFixed(4)}`)
          }
          return
        }
      }
    }

    // Log metrics
    if (this.verbose > 1 && nUpdates > 0) {
      console.log(
        `PPO Update - ` +
        `Policy Loss: ${(totalPolicyLoss / nUpdates).toFixed(4)}, ` +
        `Value Loss: ${(totalValueLoss / nUpdates).toFixed(4)}, ` +
        `Entropy: ${(totalEntropyLoss / nUpdates).toFixed(4)}, ` +
        `Approx KL: ${(totalApproxKl / nUpdates).toFixed(6)}`,
      )
    }
  }

  /**
   * Clip gradients by global norm
   * 
   * Scales all parameter gradients so that the global norm <= maxNorm.
   * Uses the formula: grad = grad * (maxNorm / totalNorm) when totalNorm > maxNorm
   */
  private clipGradients(maxNorm: number): void {
    const params = this.policy.parameters()
    const grads: any[] = []
    let totalNormSq = 0

    // Compute total gradient norm and collect gradients
    for (const param of params) {
      const grad = (param as any).grad
      if (grad) {
        grads.push(grad)
        const gradNormSq = ((grad as any).mul(grad) as any).sum().item?.() ?? 0
        totalNormSq += gradNormSq
      }
    }

    const totalNorm = Math.sqrt(totalNormSq)

    // Clip gradients if norm exceeds threshold
    if (totalNorm > maxNorm) {
      const clipCoef = maxNorm / (totalNorm + 1e-6)
      
      // Scale each gradient: grad = grad * clipCoef
      // Using addScaledInplace: grad += (clipCoef - 1) * grad = clipCoef * grad
      for (const grad of grads) {
        ;(grad as any).addScaledInplace(grad, clipCoef - 1)
      }
      
      if (this.verbose > 1) {
        console.log(`Gradient norm ${totalNorm.toFixed(4)} clipped to ${maxNorm}`)
      }
    }
  }
}

// ==================== Factory ====================

/**
 * PPO configuration definition (before env/device)
 */
export interface PPODef {
  /**
   * Initialize PPO on a device with an environment
   */
  init(device: DeviceContext<DeviceType>, env: VecEnv): PPO
}

/**
 * Create a PPO algorithm
 *
 * @param config - PPO configuration
 * @returns PPO definition that can be initialized with device and env
 *
 * @example
 * ```ts
 * const ppo = RL.ppo({
 *   policy: { netArch: { pi: [64, 64], vf: [64, 64] } },
 *   nSteps: 2048,
 *   batchSize: 64,
 *   nEpochs: 10,
 * }).init(device.cuda(0), vecEnv)
 *
 * await ppo.learn({ totalTimesteps: 1_000_000 })
 * ```
 */
export function ppo(config: PPOConfig = {}): PPODef {
  // Default policy if not specified
  const policyConfig: ActorCriticPolicyConfig | 'MlpPolicy' = config.policy ?? 'MlpPolicy'

  return {
    init(device: DeviceContext<DeviceType>, env: VecEnv): PPO {
      return new PPO({
        ...config,
        policy: policyConfig,
        env,
        device,
      })
    },
  }
}
