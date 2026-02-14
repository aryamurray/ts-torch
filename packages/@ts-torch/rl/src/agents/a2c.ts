/**
 * Advantage Actor-Critic (A2C)
 *
 * Synchronous, deterministic variant of A3C (Asynchronous Advantage Actor-Critic).
 * Simpler than PPO - uses a single gradient step per rollout without clipping.
 *
 * Key differences from PPO:
 * - No clipped surrogate objective
 * - Single gradient step per rollout (no epochs/minibatches)
 * - Typically uses shorter rollouts (nSteps=5 vs PPO's 2048)
 * - Faster per-update but may need more environment steps
 *
 * @example
 * ```ts
 * const a2c = RL.a2c({
 *   policy: { netArch: { pi: [64, 64], vf: [64, 64] } },
 *   learningRate: 7e-4,
 *   nSteps: 5,
 *   gamma: 0.99,
 * }).init(device.cuda(0), vecEnv)
 *
 * await a2c.learn({ totalTimesteps: 1_000_000 })
 * ```
 */

import { run, device as deviceModule, Logger, int64 } from '@ts-torch/core'
import type { DeviceType } from '@ts-torch/core'
import type { DeviceContext } from '@ts-torch/core'
import { OnPolicyAlgorithm } from './on-policy-base.js'
import type { OnPolicyConfig } from './on-policy-base.js'
import type { VecEnv } from '../vec-env/index.js'
import type { ActorCriticPolicyConfig } from '../policies/index.js'

// CPU device for tensor creation
const cpu = deviceModule.cpu()

// ==================== Types ====================

/**
 * A2C-specific configuration
 */
export interface A2CConfig extends Omit<OnPolicyConfig, 'env' | 'device' | 'policy'> {
  /** Policy configuration or 'MlpPolicy' shorthand */
  policy?: ActorCriticPolicyConfig | 'MlpPolicy'

  /** Whether to normalize advantages (default: false for A2C) */
  normalizeAdvantage?: boolean

  /** Whether to use RMSprop optimizer instead of Adam (default: false) */
  useRmsprop?: boolean

  /** RMSprop alpha/smoothing constant (default: 0.99) */
  rmsAlpha?: number

  /** RMSprop epsilon (default: 1e-5) */
  rmsEpsilon?: number
}

/**
 * Full A2C configuration (after init)
 */
interface A2CFullConfig extends Omit<A2CConfig, 'policy'> {
  policy: ActorCriticPolicyConfig | 'MlpPolicy'
  env: VecEnv
  device: DeviceContext<DeviceType>
}

// ==================== Implementation ====================

/**
 * Advantage Actor-Critic (A2C) Algorithm
 *
 * Simple policy gradient with advantage estimation.
 * Good baseline algorithm, faster per-update than PPO.
 */
export class A2C extends OnPolicyAlgorithm {
  // A2C-specific parameters
  private normalizeAdvantage: boolean

  constructor(config: A2CFullConfig) {
    // A2C typically uses shorter rollouts
    const nSteps = config.nSteps ?? 5
    const learningRate = config.learningRate ?? 7e-4
    const gaeLambda = config.gaeLambda ?? 1.0  // No GAE by default (lambda=1)

    super({
      ...config,
      nSteps,
      learningRate,
      gaeLambda,
    })

    this.normalizeAdvantage = config.normalizeAdvantage ?? false
  }

  /**
   * Perform A2C update
   *
   * Single gradient step over all collected data (no minibatches).
   * Uses tensor operations for proper autograd.
   */
  protected _train(): void {
    // Hoist invariants
    const obsSize = this.policy.getObservationSize()
    const isDiscrete = this.policy.isDiscreteAction()

    // Get all data as single batch (batchSize = null)
    for (const batch of this.rolloutBuffer.get(null)) {
      const {
        observations,
        actions,
        advantages,
        returns,
      } = batch

      // In-place advantage normalization (mutates the reusable buffer directly)
      if (this.normalizeAdvantage && advantages.length > 1) {
        let mean = 0
        for (let i = 0; i < advantages.length; i++) {
          mean += advantages[i]!
        }
        mean /= advantages.length
        let variance = 0
        for (let i = 0; i < advantages.length; i++) {
          const d = advantages[i]! - mean
          variance += d * d
        }
        const invStd = 1 / Math.sqrt(variance / advantages.length + 1e-8)
        for (let i = 0; i < advantages.length; i++) {
          advantages[i] = (advantages[i]! - mean) * invStd
        }
      }

      // For logging
      let policyLossValue = 0
      let valueLossValue = 0
      let entropyValue = 0

      // All tensor operations in single run() block
      run(() => {
        this.optimizer.zeroGrad()

        // Create observation tensor [batch, obsSize]
        const obsTensor = cpu.tensor(observations, [batch.batchSize, obsSize] as const)

        // Forward pass through policy networks
        const policyOutput = this.policy.getPolicyOutput(obsTensor)
        const valueTensor = this.policy.getValueOutput(obsTensor)

        // Create distribution from policy output
        const dist = this.policy.getDistribution(policyOutput, batch.batchSize)

        // Get log probs tensor
        const logProbsAllTensor = dist.logProbsTensor()

        // Create tensors for advantages, returns
        const advantagesTensor = cpu.tensor(advantages, [batch.batchSize] as const)
        const returnsTensor = cpu.tensor(returns, [batch.batchSize, 1] as const)

        // Select log_probs for taken actions
        let logProbsTensor: any

        if (isDiscrete) {
          // Use gather to select log-probs for taken actions: 1 FFI call vs one-hot → mul → sumDim (3 calls)
          const actionIndices = new BigInt64Array(batch.batchSize)
          for (let b = 0; b < batch.batchSize; b++) {
            actionIndices[b] = BigInt(Math.round(actions[b]!))
          }
          const indexTensor = cpu.tensor(actionIndices, [batch.batchSize, 1] as const, int64)
          logProbsTensor = (logProbsAllTensor as any).gather(1, indexTensor).squeeze(1)
        } else {
          // For continuous: use logProbTensor
          logProbsTensor = (dist as any).logProbTensor(actions)
        }

        // Policy loss = -mean(log_prob * advantage) - standard REINFORCE
        const weightedLogProbs = (logProbsTensor as any).mul(advantagesTensor)
        const policyLoss = (weightedLogProbs as any).mean().neg()

        // Value loss = MSE(values, returns)
        const valueLoss = (valueTensor as any).mseLoss(returnsTensor)

        // Entropy bonus (maximize entropy = minimize negative entropy)
        const entropyLoss = dist.meanEntropyTensor().neg()

        // Total loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss
        const scaledValueLoss = (valueLoss as any).mulScalar(this.vfCoef)
        const scaledEntropyLoss = (entropyLoss as any).mulScalar(this.entCoef)
        const totalLoss = (policyLoss as any).add(scaledValueLoss).add(scaledEntropyLoss)

        // Backward pass
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
      })

      // Log metrics
      Logger.debug(
        `A2C Update - ` +
        `Policy Loss: ${policyLossValue.toFixed(4)}, ` +
        `Value Loss: ${valueLossValue.toFixed(4)}, ` +
        `Entropy: ${entropyValue.toFixed(4)}`,
      )
    }
  }

  /**
   * Clip gradients by global norm
   */
  private clipGradients(maxNorm: number): void {
    const params = this.policy.parameters()
    const grads: any[] = []
    let totalNormSq = 0

    for (const param of params) {
      const grad = (param as any).grad
      if (grad) {
        grads.push(grad)
        const gradNormSq = ((grad as any).mul(grad) as any).sum().item?.() ?? 0
        totalNormSq += gradNormSq
      }
    }

    const totalNorm = Math.sqrt(totalNormSq)

    if (totalNorm > maxNorm) {
      const clipCoef = maxNorm / (totalNorm + 1e-6)
      for (const grad of grads) {
        ;(grad as any).mulScalarInplace(clipCoef)
      }
      Logger.debug(`Gradient norm ${totalNorm.toFixed(4)} clipped to ${maxNorm}`)
    }
  }
}

// ==================== Factory ====================

/**
 * A2C configuration definition (before env/device)
 */
export interface A2CDef {
  /**
   * Initialize A2C on a device with an environment
   */
  init(device: DeviceContext<DeviceType>, env: VecEnv): A2C
}

/**
 * Create an A2C algorithm
 *
 * @param config - A2C configuration
 * @returns A2C definition that can be initialized with device and env
 *
 * @example
 * ```ts
 * const a2c = RL.a2c({
 *   policy: { netArch: { pi: [64, 64], vf: [64, 64] } },
 *   nSteps: 5,
 *   learningRate: 7e-4,
 * }).init(device.cuda(0), vecEnv)
 *
 * await a2c.learn({ totalTimesteps: 1_000_000 })
 * ```
 */
export function a2c(config: A2CConfig = {}): A2CDef {
  const policyConfig: ActorCriticPolicyConfig | 'MlpPolicy' = config.policy ?? 'MlpPolicy'

  return {
    init(device: DeviceContext<DeviceType>, env: VecEnv): A2C {
      return new A2C({
        ...config,
        policy: policyConfig,
        env,
        device,
      })
    },
  }
}
