/**
 * Soft Actor-Critic (SAC)
 *
 * Off-policy maximum entropy deep reinforcement learning.
 * From: "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning
 *        with a Stochastic Actor" (Haarnoja et al., 2018)
 *
 * Key features:
 * - Maximum entropy framework (encourages exploration)
 * - Off-policy learning with replay buffer
 * - Twin Q-networks to reduce overestimation bias
 * - Automatic entropy coefficient tuning
 * - Works with continuous action spaces
 *
 * @example
 * ```ts
 * const sac = RL.sac({
 *   policy: { netArch: { pi: [256, 256], qf: [256, 256] } },
 *   learningRate: 3e-4,
 *   bufferSize: 1_000_000,
 *   batchSize: 256,
 * }).init(device.cuda(0), vecEnv)
 *
 * await sac.learn({ totalTimesteps: 1_000_000 })
 * ```
 */

import { run, device as deviceModule, Tensor, Logger } from '@ts-torch/core'
import type { DeviceType, DeviceContext } from '@ts-torch/core'
import type { Optimizer } from '@ts-torch/optim'
import { Adam } from '@ts-torch/optim'
import { OffPolicyAlgorithm } from './off-policy-base.js'
import type { OffPolicyConfig } from './off-policy-base.js'
import type { VecEnv } from '../vec-env/index.js'
import type { BoxSpace } from '../spaces/index.js'
import { sacPolicy, SACPolicy } from '../policies/sac-policy.js'
import type { SACPolicyConfig } from '../policies/sac-policy.js'

// CPU device for tensor creation
const cpu = deviceModule.cpu()

// ==================== Types ====================

/**
 * SAC-specific configuration
 */
export interface SACConfig extends Omit<OffPolicyConfig, 'env' | 'device'> {
  /** Policy configuration */
  policy?: SACPolicyConfig

  /**
   * Entropy regularization coefficient.
   * - 'auto': Learn automatically (recommended)
   * - 'auto_X': Learn automatically with initial value X (e.g., 'auto_0.1')
   * - number: Fixed value
   * Default: 'auto'
   */
  entCoef?: 'auto' | `auto_${number}` | number

  /**
   * Target entropy for automatic tuning.
   * - 'auto': -dim(action_space)
   * - number: Custom value
   * Default: 'auto'
   */
  targetEntropy?: 'auto' | number

  /** How often to update target networks (gradient steps) */
  targetUpdateInterval?: number

  /** Actor learning rate (if different from main learningRate) */
  actorLearningRate?: number

  /** Critic learning rate (if different from main learningRate) */
  criticLearningRate?: number
}

/**
 * Full SAC configuration (after init)
 */
interface SACFullConfig extends Omit<SACConfig, 'policy'> {
  policy?: SACPolicyConfig
  env: VecEnv
  device: DeviceContext<DeviceType>
}

// ==================== Implementation ====================

/**
 * Soft Actor-Critic (SAC) Algorithm
 *
 * Maximum entropy reinforcement learning for continuous action spaces.
 */
export class SAC extends OffPolicyAlgorithm {
  // Policy
  protected policy!: SACPolicy<DeviceType>
  private policyConfig: SACPolicyConfig

  // Entropy coefficient
  private entCoefConfig: 'auto' | `auto_${number}` | number
  private entCoef: number = 0.2  // Current entropy coefficient
  private logEntCoef: number = Math.log(0.2)  // Log for optimization
  private targetEntropy: number = 0
  private autoEntropyTuning: boolean = false
  private entCoefLearningRate: number

  // Target network update
  private targetUpdateInterval: number

  // Optimizers
  private actorOptimizer!: Optimizer
  private criticOptimizer!: Optimizer

  // Stats
  private actorLosses: number[] = []
  private criticLosses: number[] = []
  private entCoefLosses: number[] = []

  constructor(config: SACFullConfig) {
    super(config)

    this.policyConfig = config.policy ?? { netArch: { pi: [256, 256], qf: [256, 256] } }
    this.entCoefConfig = config.entCoef ?? 'auto'
    this.targetUpdateInterval = config.targetUpdateInterval ?? 1
    this.entCoefLearningRate = this.getLearningRate()
  }

  /**
   * Get current learning rate
   */
  private getLearningRate(): number {
    const lr = this.learningRate
    if (typeof lr === 'number') return lr
    return lr(1.0)  // Initial value
  }

  /**
   * Setup SAC model
   */
  protected _setupModel(): void {
    // Call base setup (creates replay buffer)
    super._setupModel()

    // Create SAC policy
    this.policy = sacPolicy(this.policyConfig).init(this.device, {
      observationSize: this.env.observationSize,
      actionSpace: this.env.actionSpace as BoxSpace,
    })

    // Create optimizers
    const lr = this.getLearningRate()
    this.actorOptimizer = new Adam(this.policy.actorParameters(), { lr })
    this.criticOptimizer = new Adam(this.policy.criticParameters(), { lr })

    // Setup entropy coefficient
    this.setupEntropyCoef()

    // Set policy to training mode
    this.policy.train()
  }

  /**
   * Setup entropy coefficient (possibly learnable)
   */
  private setupEntropyCoef(): void {
    // Determine target entropy
    if (typeof this.targetEntropy === 'number') {
      this.targetEntropy = this.targetEntropy
    } else {
      // auto: -dim(action_space)
      this.targetEntropy = -this.actionDim
    }

    // Setup entropy coefficient
    if (typeof this.entCoefConfig === 'string' && this.entCoefConfig.startsWith('auto')) {
      this.autoEntropyTuning = true

      // Parse initial value if provided (e.g., 'auto_0.1')
      let initValue = 1.0
      if (this.entCoefConfig.includes('_')) {
        const parts = this.entCoefConfig.split('_')
        if (parts[1]) {
          initValue = parseFloat(parts[1])
          if (initValue <= 0) {
            throw new Error('Initial entropy coefficient must be > 0')
          }
        }
      }

      this.logEntCoef = Math.log(initValue)
      this.entCoef = initValue
    } else if (typeof this.entCoefConfig === 'number') {
      this.autoEntropyTuning = false
      this.entCoef = this.entCoefConfig
      this.logEntCoef = Math.log(this.entCoefConfig)
    }
  }

  /**
   * Select actions from policy
   */
  protected selectActions(obs: Float32Array, explore: boolean): Float32Array {
    const { actions } = this.policy.getAction(obs, !explore)
    return actions
  }

  /**
   * Perform SAC training step
   *
   * Updates:
   * 1. Q-networks (critic)
   * 2. Policy (actor)
   * 3. Entropy coefficient (if auto)
   * 4. Target networks (soft update)
   */
  protected _train(): void {
    if (!this.shouldTrain()) {
      return
    }

    const obsSize = this.policy.getObservationSize()
    const actionDim = this.policy.getActionDim()

    // Train for gradient_steps
    for (let step = 0; step < this.gradientSteps; step++) {
      // Sample batch from replay buffer
      const batch = this.replayBuffer.sample(this.batchSize)
      const batchSize = batch.rewards.length

      // Track losses for logging
      let criticLossValue = 0
      let actorLossValue = 0
      let meanLogProb = 0

      // ========== 1. Update Critics ==========
      // Q-loss = E[(Q(s,a) - (r + gamma * (1-done) * (min_Q_target - alpha * log_prob)))^2]
      run(() => {
        this.criticOptimizer.zeroGrad()

        // Create tensors from batch data (no `as const` for consistent shape typing)
        const statesTensor = cpu.tensor(batch.states, [batchSize, obsSize]) as Tensor
        const actionsTensor = cpu.tensor(batch.actions, [batchSize, actionDim]) as Tensor
        const nextStatesTensor = cpu.tensor(batch.nextStates, [batchSize, obsSize]) as Tensor
        const rewardsTensor = cpu.tensor(batch.rewards, [batchSize, 1]) as Tensor
        
        // Convert dones to float32
        const donesFloat = new Float32Array(batchSize)
        for (let i = 0; i < batchSize; i++) {
          donesFloat[i] = batch.dones[i]!
        }
        const donesTensor = cpu.tensor(donesFloat, [batchSize, 1]) as Tensor

        // Get next actions and log probs from current policy (no grad for target computation)
        const nextActorOutput = this.policy.getActorOutput(nextStatesTensor as Tensor)
        const { actionTensor: nextActionTensor, logProbTensor: nextLogProbTensor } = 
          this.policy.sampleActionTensor(nextActorOutput, false)

        // Get target Q-values for next state-action pairs
        const targetQValues = this.policy.getTargetCriticOutputs(nextStatesTensor as Tensor, nextActionTensor)

        // Take minimum across target critics
        let minTargetQ = targetQValues[0]!
        for (let i = 1; i < targetQValues.length; i++) {
          minTargetQ = minTargetQ.minimum(targetQValues[i]!)
        }

        // Compute target: r + gamma * (1 - done) * (min_Q_target - alpha * log_prob)
        const entCoefTensor = cpu.tensor([this.entCoef], [1]) as Tensor
        const gammaTensor = cpu.tensor([this.gamma], [1]) as Tensor
        const onesTensor = cpu.ones([batchSize, 1]) as Tensor
        
        const entropyTerm = (nextLogProbTensor.reshape([batchSize, 1]) as Tensor).mul(entCoefTensor)
        const nextValue = minTargetQ.sub(entropyTerm as Tensor)
        const notDone = onesTensor.sub(donesTensor)
        const discountedValue = (nextValue as Tensor).mul(notDone).mul(gammaTensor)
        const targetQ = rewardsTensor.add(discountedValue as Tensor)

        // Get current Q-values from critic networks
        const currentQValues = this.policy.getCriticOutputs(statesTensor as Tensor, actionsTensor as Tensor)

        // Compute MSE loss for each critic
        let totalCriticLoss: Tensor | null = null
        for (const currentQ of currentQValues) {
          const criticLoss = currentQ.mseLoss(targetQ as Tensor)
          if (totalCriticLoss === null) {
            totalCriticLoss = criticLoss as Tensor
          } else {
            totalCriticLoss = totalCriticLoss.add(criticLoss as Tensor) as Tensor
          }
        }

        // Backprop critic loss
        totalCriticLoss!.backward()
        this.criticOptimizer.step()

        criticLossValue = totalCriticLoss!.item()
      })

      this.criticLosses.push(criticLossValue)

      // ========== 2. Update Actor ==========
      // Actor loss = E[alpha * log_prob - min_Q(s, pi(s))]
      run(() => {
        this.actorOptimizer.zeroGrad()

        // Create state tensor
        const statesTensor = cpu.tensor(batch.states, [batchSize, obsSize]) as Tensor

        // Get new actions from policy
        const actorOutput = this.policy.getActorOutput(statesTensor as Tensor)
        const { actionTensor, logProbTensor } = this.policy.sampleActionTensor(actorOutput, false)

        // Get Q-values for new actions
        const qValues = this.policy.getCriticOutputs(statesTensor as Tensor, actionTensor)

        // Take minimum Q
        let minQ = qValues[0]!
        for (let i = 1; i < qValues.length; i++) {
          minQ = minQ.minimum(qValues[i]!)
        }

        // Actor loss: alpha * log_prob - Q
        const entCoefTensor = cpu.tensor([this.entCoef], [1]) as Tensor
        const logProbReshaped = logProbTensor.reshape([batchSize, 1]) as Tensor
        const entropyTerm = logProbReshaped.mul(entCoefTensor)
        const actorLoss = (entropyTerm as Tensor).sub(minQ).mean()

        // Backprop actor loss
        actorLoss.backward()
        this.actorOptimizer.step()

        actorLossValue = actorLoss.item()
        meanLogProb = logProbTensor.mean().item()
      })

      this.actorLosses.push(actorLossValue)

      // ========== 3. Update Entropy Coefficient (if auto) ==========
      if (this.autoEntropyTuning) {
        // Loss = -log(alpha) * (log_prob + target_entropy)
        // Gradient: d(loss)/d(log_alpha) = -(log_prob + target_entropy)
        const gradient = -(meanLogProb + this.targetEntropy)
        
        // Update log_ent_coef with simple gradient descent
        this.logEntCoef -= this.entCoefLearningRate * gradient
        this.entCoef = Math.exp(this.logEntCoef)
        
        // Track loss
        const entCoefLoss = -this.logEntCoef * (meanLogProb + this.targetEntropy)
        this.entCoefLosses.push(entCoefLoss)
      }

      // ========== 4. Soft Update Target Networks ==========
      if (step % this.targetUpdateInterval === 0) {
        this.softUpdateTargets()
      }
    }

    // Log stats
    if (this.actorLosses.length > 0) {
      const avgActorLoss = this.actorLosses.reduce((a, b) => a + b, 0) / this.actorLosses.length
      const avgCriticLoss = this.criticLosses.reduce((a, b) => a + b, 0) / this.criticLosses.length

      Logger.debug(
        `SAC Update - ` +
        `Actor Loss: ${avgActorLoss.toFixed(4)}, ` +
        `Critic Loss: ${avgCriticLoss.toFixed(4)}, ` +
        `Ent Coef: ${this.entCoef.toFixed(4)}`,
      )

      // Clear stats
      this.actorLosses = []
      this.criticLosses = []
      this.entCoefLosses = []
    }
  }

  /**
   * Soft update target networks using Polyak averaging
   * target = tau * source + (1 - tau) * target
   */
  private softUpdateTargets(): void {
    const sourceParams = this.policy.criticParameters()
    const targetParams = this.policy.criticTargetParameters()

    run(() => {
      for (let i = 0; i < sourceParams.length; i++) {
        const source = sourceParams[i] as any
        const target = targetParams[i] as any
        
        if (source.data && target.data) {
          // target.data = tau * source.data + (1 - tau) * target.data
          // Using: target += tau * (source - target)
          const diff = source.data.sub(target.data)
          target.data.addScaledInplace(diff, this.tau)
        }
      }
    })
  }

  /**
   * Predict action for given observation
   *
   * @param observation - Single observation
   * @param deterministic - If true, return mean action
   */
  predict(observation: Float32Array, deterministic: boolean = false): Float32Array {
    const { actions } = this.policy.getAction(observation, deterministic)
    return actions
  }

  /**
   * Get current entropy coefficient
   */
  getEntCoef(): number {
    return this.entCoef
  }

  /**
   * Save the algorithm to a file
   */
  async save(path: string): Promise<void> {
    // TODO: Implement checkpoint saving
    Logger.info(`Saving SAC to ${path}`)
  }

  /**
   * Get all trainable parameters
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
}

// ==================== Factory ====================

/**
 * SAC definition (before env/device)
 */
export interface SACDef {
  /**
   * Initialize SAC on a device with an environment
   */
  init(device: DeviceContext<DeviceType>, env: VecEnv): SAC
}

/**
 * Create a SAC algorithm
 *
 * @param config - SAC configuration
 * @returns SAC definition that can be initialized with device and env
 *
 * @example
 * ```ts
 * const sac = RL.sac({
 *   policy: { netArch: { pi: [256, 256], qf: [256, 256] } },
 *   bufferSize: 1_000_000,
 *   batchSize: 256,
 *   entCoef: 'auto',
 * }).init(device.cuda(0), vecEnv)
 *
 * await sac.learn({ totalTimesteps: 1_000_000 })
 * ```
 */
export function sac(config: SACConfig = {}): SACDef {
  return {
    init(device: DeviceContext<DeviceType>, env: VecEnv): SAC {
      return new SAC({
        ...config,
        env,
        device,
      })
    },
  }
}
