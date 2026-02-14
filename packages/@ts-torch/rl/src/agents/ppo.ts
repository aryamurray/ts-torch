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

import { run, device as deviceModule, Logger, int64, float32, getLib, Tensor } from '@ts-torch/core'
import type { DeviceType } from '@ts-torch/core'
import type { DeviceContext } from '@ts-torch/core'
import { OnPolicyAlgorithm } from './on-policy-base.js'
import type { OnPolicyConfig } from './on-policy-base.js'
import type { Schedule } from './base-algorithm.js'
import type { VecEnv } from '../vec-env/index.js'
import type { ActorCriticPolicyConfig } from '../policies/index.js'

// CPU device for tensor creation
const cpu = deviceModule.cpu()

// ==================== Lazy Native Function Resolution ====================

let _nativePolicyForward: Function | null | undefined
function getNativePolicyForward(): Function | null {
  if (_nativePolicyForward !== undefined) return _nativePolicyForward
  try {
    const fn = getLib().ts_policy_forward
    _nativePolicyForward = typeof fn === 'function' ? fn : null
  } catch {
    _nativePolicyForward = null
  }
  return _nativePolicyForward
}

let _nativeBackwardAndClip: Function | null | undefined
function getNativeBackwardAndClip(): Function | null {
  if (_nativeBackwardAndClip !== undefined) return _nativeBackwardAndClip
  try {
    const fn = getLib().ts_backward_and_clip
    _nativeBackwardAndClip = typeof fn === 'function' ? fn : null
  } catch {
    _nativeBackwardAndClip = null
  }
  return _nativeBackwardAndClip
}

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
   * Perform PPO update - dispatches to native or JS path
   */
  protected _train(): void {
    const clipRange = this.getCurrentClipRange()
    const obsSize = this.policy.getObservationSize()
    const isDiscrete = this.policy.isDiscreteAction()
    const needsKl = this.targetKl !== null
    const clipLow = 1 - clipRange
    const clipHigh = 1 + clipRange

    const useNative = isDiscrete
      && getNativePolicyForward() !== null
      && getNativeBackwardAndClip() !== null

    if (useNative) {
      this._trainNative(obsSize, needsKl, clipLow, clipHigh)
    } else {
      this._trainJS(obsSize, isDiscrete, needsKl, clipLow, clipHigh)
    }
  }

  /**
   * Native PPO training path — uses fused C++ ops to reduce FFI overhead.
   *
   * ts_policy_forward: fuses piNet forward + vfNet forward + categorical dist (saves ~19 FFI calls)
   * ts_backward_and_clip: fuses zero_grad + backward + grad clip (saves ~61 FFI calls)
   */
  private _trainNative(
    obsSize: number,
    needsKl: boolean,
    clipLow: number,
    clipHigh: number,
  ): void {
    const nativePolicyForward = getNativePolicyForward()!
    const nativeBackwardAndClip = getNativeBackwardAndClip()!

    // Extract parameter handles once before epoch loop
    const piParams = this.policy.policyNetParameters()
    const vfParams = this.policy.valueNetParameters()
    const piHandles = piParams.map((p: any) => p.data._handle)
    const vfHandles = vfParams.map((p: any) => p.data._handle)
    const allHandles = [...piHandles, ...vfHandles]
    const activationType = this.policy.getActivationType()
    const nActions = this.policy.getNumActions()

    let totalPolicyLoss = 0
    let totalValueLoss = 0
    let totalEntropyLoss = 0
    let totalApproxKl = 0
    let nUpdates = 0

    for (let epoch = 0; epoch < this.nEpochs; epoch++) {
      for (const batch of this.rolloutBuffer.get(this.batchSize)) {
        const { observations, actions, oldLogProbs, advantages, returns } = batch

        // In-place advantage normalization
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

        let policyLossValue = 0
        let valueLossValue = 0
        let entropyValue = 0
        let approxKlValue = 0

        run(() => {
          // 1. Fused forward pass: piNet + vfNet + categorical distribution → 1 FFI call
          const fwdResult = nativePolicyForward(
            observations,
            actions,
            batch.batchSize,
            obsSize,
            nActions,
            piHandles,
            vfHandles,
            activationType,
          )

          // Wrap returned handles as Tensor objects for PPO loss computation
          // Tensor constructor is @internal — use `as any` to bypass .d.ts restriction
          const TensorCtor = Tensor as any
          const actionLogProbs = new TensorCtor(fwdResult.actionLogProbs, [batch.batchSize] as const, float32)
          const entropy = new TensorCtor(fwdResult.entropy, [] as const, float32)
          const values = new TensorCtor(fwdResult.values, [batch.batchSize] as const, float32)

          // 2. PPO loss computation in TS (~12 FFI calls — cheap)
          const oldLogProbsTensor = cpu.tensor(oldLogProbs, [batch.batchSize] as const)
          const advantagesTensor = cpu.tensor(advantages, [batch.batchSize] as const)
          const returnsTensor = cpu.tensor(returns, [batch.batchSize, 1] as const)

          const logDiff = (actionLogProbs as any).sub(oldLogProbsTensor)
          const ratio = (logDiff as any).exp()

          const surr1 = (ratio as any).mul(advantagesTensor)
          const clippedRatio = (ratio as any).clamp(clipLow, clipHigh)
          const surr2 = (clippedRatio as any).mul(advantagesTensor)

          const minSurr = (surr1 as any).minimum(surr2)
          const policyLoss = (minSurr as any).mean().neg()

          const valueLoss = (values as any).unsqueeze(1).mseLoss(returnsTensor)

          const entropyLoss = (entropy as any).neg()

          const scaledValueLoss = (valueLoss as any).mulScalar(this.vfCoef)
          const scaledEntropyLoss = (entropyLoss as any).mulScalar(this.entCoef)
          const totalLoss = (policyLoss as any).add(scaledValueLoss).add(scaledEntropyLoss)

          // 3. Fused backward + grad clip → 1 FFI call
          nativeBackwardAndClip(
            (totalLoss as any)._handle,
            allHandles,
            this.maxGradNorm,
          )

          // 4. Invalidate JS gradient caches — ts_backward_and_clip populated gradients
          //    in C++ without going through the JS .grad getter. The optimizer's step()
          //    will call .grad which would return stale cached handles from a previous scope.
          //    FRAGILE: reaches into Tensor._gradCache (private). If that field is renamed,
          //    this will silently stop working and cause a use-after-free segfault.
          for (const p of piParams) {
            ;(p as any).data._gradCache = undefined
          }
          for (const p of vfParams) {
            ;(p as any).data._gradCache = undefined
          }

          // 5. Optimizer step (stays in TS)
          this.optimizer.step()

          // Extract metrics
          policyLossValue = (policyLoss as any).item?.() ?? 0
          valueLossValue = (valueLoss as any).item?.() ?? 0
          entropyValue = (entropy as any).item?.() ?? 0

          if (needsKl) {
            const klDiff = (oldLogProbsTensor as any).sub(actionLogProbs)
            const klSquared = (klDiff as any).mul(klDiff)
            approxKlValue = ((klSquared as any).mean().item?.() ?? 0) * 0.5
          }
        })

        totalPolicyLoss += policyLossValue
        totalValueLoss += valueLossValue
        totalEntropyLoss += entropyValue
        totalApproxKl += approxKlValue
        nUpdates++

        if (needsKl && approxKlValue > 1.5 * this.targetKl!) {
          Logger.info(`Early stopping at epoch ${epoch} due to KL divergence: ${approxKlValue.toFixed(4)}`)
          return
        }
      }
    }

    if (nUpdates > 0) {
      Logger.debug(
        `PPO Update (native) - ` +
        `Policy Loss: ${(totalPolicyLoss / nUpdates).toFixed(4)}, ` +
        `Value Loss: ${(totalValueLoss / nUpdates).toFixed(4)}, ` +
        `Entropy: ${(totalEntropyLoss / nUpdates).toFixed(4)}, ` +
        `Approx KL: ${(totalApproxKl / nUpdates).toFixed(6)}`,
      )
    }
  }

  /**
   * JS PPO training path — fallback when native ops unavailable (continuous actions, etc.)
   */
  private _trainJS(
    obsSize: number,
    isDiscrete: boolean,
    needsKl: boolean,
    clipLow: number,
    clipHigh: number,
  ): void {
    let totalPolicyLoss = 0
    let totalValueLoss = 0
    let totalEntropyLoss = 0
    let totalApproxKl = 0
    let nUpdates = 0

    for (let epoch = 0; epoch < this.nEpochs; epoch++) {
      for (const batch of this.rolloutBuffer.get(this.batchSize)) {
        const { observations, actions, oldLogProbs, advantages, returns } = batch

        // In-place advantage normalization
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

        let policyLossValue = 0
        let valueLossValue = 0
        let entropyValue = 0
        let approxKlValue = 0

        run(() => {
          this.optimizer.zeroGrad()

          const obsTensor = cpu.tensor(observations, [batch.batchSize, obsSize] as const)

          const policyOutput = this.policy.getPolicyOutput(obsTensor)
          const valueTensor = this.policy.getValueOutput(obsTensor)

          const dist = this.policy.getDistribution(policyOutput, batch.batchSize)

          const logProbsAllTensor = dist.logProbsTensor()

          const oldLogProbsTensor = cpu.tensor(oldLogProbs, [batch.batchSize] as const)
          const advantagesTensor = cpu.tensor(advantages, [batch.batchSize] as const)
          const returnsTensor = cpu.tensor(returns, [batch.batchSize, 1] as const)

          let logProbsTensor: any

          if (isDiscrete) {
            const actionIndices = new BigInt64Array(batch.batchSize)
            for (let b = 0; b < batch.batchSize; b++) {
              actionIndices[b] = BigInt(Math.round(actions[b]!))
            }
            const indexTensor = cpu.tensor(actionIndices, [batch.batchSize, 1] as const, int64)
            logProbsTensor = (logProbsAllTensor as any).gather(1, indexTensor).squeeze(1)
          } else {
            logProbsTensor = (dist as any).logProbTensor(actions)
          }

          const logDiff = (logProbsTensor as any).sub(oldLogProbsTensor)
          const ratio = (logDiff as any).exp()

          const surr1 = (ratio as any).mul(advantagesTensor)
          const clippedRatio = (ratio as any).clamp(clipLow, clipHigh)
          const surr2 = (clippedRatio as any).mul(advantagesTensor)

          const minSurr = (surr1 as any).minimum(surr2)
          const policyLoss = (minSurr as any).mean().neg()

          const valueLoss = (valueTensor as any).mseLoss(returnsTensor)

          const entropyLoss = dist.meanEntropyTensor().neg()

          const scaledValueLoss = (valueLoss as any).mulScalar(this.vfCoef)
          const scaledEntropyLoss = (entropyLoss as any).mulScalar(this.entCoef)
          const totalLoss = (policyLoss as any).add(scaledValueLoss).add(scaledEntropyLoss)

          ;(totalLoss as any).backward()

          if (this.maxGradNorm > 0) {
            this.clipGradients(this.maxGradNorm)
          }

          this.optimizer.step()

          policyLossValue = (policyLoss as any).item?.() ?? 0
          valueLossValue = (valueLoss as any).item?.() ?? 0
          entropyValue = -((entropyLoss as any).item?.() ?? 0)

          if (needsKl) {
            const klDiff = (oldLogProbsTensor as any).sub(logProbsTensor)
            const klSquared = (klDiff as any).mul(klDiff)
            approxKlValue = ((klSquared as any).mean().item?.() ?? 0) * 0.5
          }
        })

        totalPolicyLoss += policyLossValue
        totalValueLoss += valueLossValue
        totalEntropyLoss += entropyValue
        totalApproxKl += approxKlValue
        nUpdates++

        if (needsKl && approxKlValue > 1.5 * this.targetKl!) {
          Logger.info(`Early stopping at epoch ${epoch} due to KL divergence: ${approxKlValue.toFixed(4)}`)
          return
        }
      }
    }

    if (nUpdates > 0) {
      Logger.debug(
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
 * Initialized PPO agent - the type returned by `ppo({...}).init(device, env)`
 *
 * This is a type alias for `PPO` to make annotations clearer.
 *
 * @example
 * ```ts
 * import type { PPOAgent } from '@ts-torch/rl'
 *
 * const agent: PPOAgent = RL.ppo({...}).init(device, env)
 * await agent.learn({ totalTimesteps: 1_000_000 })
 * const action = agent.predict(observation)
 * ```
 */
export type PPOAgent = PPO

/**
 * PPO configuration definition (before env/device)
 */
export interface PPODef {
  /**
   * Initialize PPO on a device with an environment
   *
   * @param device - Device context (e.g., `device.cuda(0)` or `device.cpu()`)
   * @param env - Vectorized environment
   * @returns Initialized PPO agent ready for training
   *
   * @example
   * ```ts
   * const agent = RL.ppo({
   *   learningRate: 3e-4,
   *   nSteps: 2048,
   * }).init(device.cuda(0), vecEnv)
   * ```
   */
  init(device: DeviceContext<DeviceType>, env: VecEnv): PPOAgent
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
