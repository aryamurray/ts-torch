/**
 * SAC Policy
 *
 * Policy for Soft Actor-Critic with:
 * - Squashed Gaussian actor (outputs mean and log_std, samples with tanh squashing)
 * - Twin Q-networks (for reducing overestimation bias)
 * - Target Q-networks (soft updated)
 *
 * @example
 * ```ts
 * const policyConfig = sacPolicy({
 *   netArch: { pi: [256, 256], qf: [256, 256] },
 * })
 *
 * const policy = policyConfig.init(device.cuda(0), {
 *   observationSize: 11,
 *   actionSpace: box({ low: [-1, -1, -1], high: [1, 1, 1], shape: [3] }),
 * })
 * ```
 */

import type { DeviceType } from '@ts-torch/core'
import { run, device as deviceModule, Tensor, cat } from '@ts-torch/core'
import type { DeviceContext } from '@ts-torch/core'
import type { Module, SequenceDef } from '@ts-torch/nn'
import { nn } from '@ts-torch/nn'
import type { BoxSpace } from '../spaces/index.js'

/**
 * Tensor type for RL operations with dynamic shapes.
 * We use the base Tensor type (with default parameters) for maximum compatibility
 * with all tensor methods. The shape/dtype will be inferred at runtime.
 */
type RLTensor = Tensor

// CPU device for tensor creation
const cpu = deviceModule.cpu()

// Constants
const LOG_STD_MIN = -20
const LOG_STD_MAX = 2

// ==================== Types ====================

/**
 * Network architecture for SAC
 */
export interface SACNetArch {
  /** Hidden layer sizes for actor (policy) network */
  pi: number[]
  /** Hidden layer sizes for Q-networks */
  qf: number[]
}

/**
 * SAC policy configuration
 */
export interface SACPolicyConfig {
  /** Network architecture */
  netArch?: SACNetArch

  /** Custom actor network (overrides netArch.pi) */
  actorNet?: SequenceDef

  /** Custom Q-network definition (overrides netArch.qf) */
  criticNet?: SequenceDef

  /** Number of Q-networks (default: 2 for twin Q) */
  nCritics?: number

  /** Activation function (default: 'relu') */
  activation?: 'relu' | 'tanh' | 'gelu'
}

/**
 * Spaces for SAC policy initialization
 */
export interface SACPolicySpaces {
  observationSize: number
  actionSpace: BoxSpace
}

/**
 * SAC policy definition (before initialization)
 */
export interface SACPolicyDef {
  init<Dev extends DeviceType>(
    device: DeviceContext<Dev>,
    spaces: SACPolicySpaces,
  ): SACPolicy<Dev>
}

// ==================== Implementation ====================

/**
 * Squashed Gaussian Actor for SAC
 *
 * Outputs mean and log_std, samples actions with tanh squashing.
 */
export class SACPolicy<Dev extends DeviceType = DeviceType> {
  private readonly device_: DeviceContext<Dev>
  private readonly observationSize: number
  private readonly actionDim: number
  private readonly actionScale: Float32Array
  private readonly actionBias: Float32Array

  // Networks
  private actor: Module<any, any, any, Dev>
  private critics: Module<any, any, any, Dev>[]
  private criticTargets: Module<any, any, any, Dev>[]

  constructor(
    device: DeviceContext<Dev>,
    spaces: SACPolicySpaces,
    actor: Module<any, any, any, Dev>,
    critics: Module<any, any, any, Dev>[],
    criticTargets: Module<any, any, any, Dev>[],
  ) {
    this.device_ = device
    this.observationSize = spaces.observationSize
    this.actionDim = spaces.actionSpace.shape.reduce((a, b) => a * b, 1)

    // Compute action scaling for squashing
    // action = tanh(raw_action) * scale + bias
    this.actionScale = new Float32Array(this.actionDim)
    this.actionBias = new Float32Array(this.actionDim)
    for (let i = 0; i < this.actionDim; i++) {
      const low = spaces.actionSpace.low[i]!
      const high = spaces.actionSpace.high[i]!
      this.actionScale[i] = (high - low) / 2
      this.actionBias[i] = (high + low) / 2
    }

    this.actor = actor
    this.critics = critics
    this.criticTargets = criticTargets
  }

  /**
   * Get action from observation (with optional deterministic mode)
   *
   * @param observations - Batch of observations [batch * obsSize]
   * @param deterministic - If true, return mean action (no sampling)
   * @returns Actions [batch * actionDim] and log_probs [batch]
   */
  getAction(
    observations: Float32Array,
    deterministic: boolean = false,
  ): { actions: Float32Array; logProbs: Float32Array } {
    const batchSize = observations.length / this.observationSize

    let actions: Float32Array
    let logProbs: Float32Array

    run(() => {
      const obsTensor = cpu.tensor(observations, [batchSize, this.observationSize] as const)

      // Get mean and log_std from actor
      const actorOutput = (this.actor as any).forward(obsTensor)
      const outputArr = actorOutput.toArray() as Float32Array

      // Split into mean and log_std
      const mean = new Float32Array(batchSize * this.actionDim)
      const logStd = new Float32Array(batchSize * this.actionDim)

      for (let b = 0; b < batchSize; b++) {
        for (let a = 0; a < this.actionDim; a++) {
          mean[b * this.actionDim + a] = outputArr[b * this.actionDim * 2 + a]!
          // Clamp log_std
          const rawLogStd = outputArr[b * this.actionDim * 2 + this.actionDim + a]!
          logStd[b * this.actionDim + a] = Math.max(LOG_STD_MIN, Math.min(LOG_STD_MAX, rawLogStd))
        }
      }

      if (deterministic) {
        // Return tanh(mean) scaled to action space
        actions = new Float32Array(batchSize * this.actionDim)
        logProbs = new Float32Array(batchSize)

        for (let b = 0; b < batchSize; b++) {
          for (let a = 0; a < this.actionDim; a++) {
            const idx = b * this.actionDim + a
            const tanhMean = Math.tanh(mean[idx]!)
            actions[idx] = tanhMean * this.actionScale[a]! + this.actionBias[a]!
          }
          logProbs[b] = 0 // No entropy for deterministic
        }
      } else {
        // Sample from Gaussian and apply tanh squashing
        actions = new Float32Array(batchSize * this.actionDim)
        logProbs = new Float32Array(batchSize)

        for (let b = 0; b < batchSize; b++) {
          let logProbSum = 0

          for (let a = 0; a < this.actionDim; a++) {
            const idx = b * this.actionDim + a
            const mu = mean[idx]!
            const sigma = Math.exp(logStd[idx]!)

            // Sample: raw = mu + sigma * noise
            const noise = this.sampleNormal()
            const rawAction = mu + sigma * noise

            // Apply tanh squashing
            const tanhAction = Math.tanh(rawAction)
            actions[idx] = tanhAction * this.actionScale[a]! + this.actionBias[a]!

            // Compute log probability with squashing correction
            // log_prob = log_prob_gaussian - log(1 - tanh^2(raw))
            const logProbGaussian = -0.5 * (noise ** 2 + Math.log(2 * Math.PI)) - logStd[idx]!
            const squashingCorrection = Math.log(1 - tanhAction ** 2 + 1e-6)
            logProbSum += logProbGaussian - squashingCorrection
          }

          logProbs[b] = logProbSum
        }
      }
    })

    return { actions: actions!, logProbs: logProbs! }
  }

  /**
   * Compute Q-values for state-action pairs
   *
   * @param observations - States [batch * obsSize]
   * @param actions - Actions [batch * actionDim]
   * @returns Q-values from each critic [[batch], [batch], ...]
   */
  getQValues(
    observations: Float32Array,
    actions: Float32Array,
  ): Float32Array[] {
    const batchSize = observations.length / this.observationSize
    const qValues: Float32Array[] = []

    run(() => {
      // Concatenate obs and actions
      const input = new Float32Array(batchSize * (this.observationSize + this.actionDim))
      for (let b = 0; b < batchSize; b++) {
        const obsOffset = b * this.observationSize
        const actOffset = b * this.actionDim
        const inputOffset = b * (this.observationSize + this.actionDim)

        for (let i = 0; i < this.observationSize; i++) {
          input[inputOffset + i] = observations[obsOffset + i]!
        }
        for (let i = 0; i < this.actionDim; i++) {
          input[inputOffset + this.observationSize + i] = actions[actOffset + i]!
        }
      }

      const inputTensor = cpu.tensor(input, [batchSize, this.observationSize + this.actionDim] as const)

      // Get Q-values from each critic
      for (const critic of this.critics) {
        const qOutput = (critic as any).forward(inputTensor)
        qValues.push(qOutput.toArray() as Float32Array)
      }
    })

    return qValues
  }

  /**
   * Compute target Q-values (from target networks)
   */
  getTargetQValues(
    observations: Float32Array,
    actions: Float32Array,
  ): Float32Array[] {
    const batchSize = observations.length / this.observationSize
    const qValues: Float32Array[] = []

    run(() => {
      const input = new Float32Array(batchSize * (this.observationSize + this.actionDim))
      for (let b = 0; b < batchSize; b++) {
        const obsOffset = b * this.observationSize
        const actOffset = b * this.actionDim
        const inputOffset = b * (this.observationSize + this.actionDim)

        for (let i = 0; i < this.observationSize; i++) {
          input[inputOffset + i] = observations[obsOffset + i]!
        }
        for (let i = 0; i < this.actionDim; i++) {
          input[inputOffset + this.observationSize + i] = actions[actOffset + i]!
        }
      }

      const inputTensor = cpu.tensor(input, [batchSize, this.observationSize + this.actionDim] as const)

      for (const critic of this.criticTargets) {
        const qOutput = (critic as any).forward(inputTensor)
        qValues.push(qOutput.toArray() as Float32Array)
      }
    })

    return qValues
  }

  /**
   * Get actor parameters
   */
  actorParameters(): any[] {
    return this.actor.parameters()
  }

  /**
   * Get critic parameters
   */
  criticParameters(): any[] {
    const params: any[] = []
    for (const critic of this.critics) {
      params.push(...critic.parameters())
    }
    return params
  }

  /**
   * Get all parameters
   */
  parameters(): any[] {
    return [...this.actorParameters(), ...this.criticParameters()]
  }

  /**
   * Get critic target parameters (for soft update)
   */
  criticTargetParameters(): any[] {
    const params: any[] = []
    for (const critic of this.criticTargets) {
      params.push(...critic.parameters())
    }
    return params
  }

  /**
   * Set to training mode
   */
  train(): void {
    this.actor.train()
    for (const critic of this.critics) {
      critic.train()
    }
  }

  /**
   * Set to evaluation mode
   */
  eval(): void {
    this.actor.eval()
    for (const critic of this.critics) {
      critic.eval()
    }
  }

  get device(): DeviceContext<Dev> {
    return this.device_
  }

  /**
   * Sample from standard normal
   */
  private sampleNormal(): number {
    const u1 = Math.random()
    const u2 = Math.random()
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
  }

  // ==================== Tensor Methods (for training with autograd) ====================

  /**
   * Get actor network output tensor (mean and log_std concatenated)
   *
   * Must be called within a run() block. Returns tensor connected to computational graph.
   *
   * @param obsTensor - Observation tensor [batch, obsSize]
   * @returns Actor output tensor [batch, actionDim * 2] containing [mean, log_std]
   */
  getActorOutput(obsTensor: RLTensor): RLTensor {
    return (this.actor as any).forward(obsTensor) as RLTensor
  }

  /**
   * Get Q-values from critic networks as tensors
   *
   * Must be called within a run() block. Returns tensors connected to computational graph.
   *
   * @param obsTensor - Observation tensor [batch, obsSize]
   * @param actionTensor - Action tensor [batch, actionDim]
   * @returns Array of Q-value tensors [batch, 1] from each critic
   */
  getCriticOutputs(obsTensor: RLTensor, actionTensor: RLTensor): RLTensor[] {
    // Concatenate observations and actions: [batch, obsSize + actionDim]
    const inputTensor = cat([obsTensor, actionTensor], 1) as RLTensor
    
    const qValues: RLTensor[] = []
    for (const critic of this.critics) {
      qValues.push((critic as any).forward(inputTensor) as RLTensor)
    }
    return qValues
  }

  /**
   * Get Q-values from target critic networks as tensors
   *
   * Must be called within a run() block. Returns tensors connected to computational graph.
   *
   * @param obsTensor - Observation tensor [batch, obsSize]
   * @param actionTensor - Action tensor [batch, actionDim]
   * @returns Array of target Q-value tensors [batch, 1] from each target critic
   */
  getTargetCriticOutputs(obsTensor: RLTensor, actionTensor: RLTensor): RLTensor[] {
    // Concatenate observations and actions: [batch, obsSize + actionDim]
    const inputTensor = cat([obsTensor, actionTensor], 1) as RLTensor
    
    const qValues: RLTensor[] = []
    for (const critic of this.criticTargets) {
      qValues.push((critic as any).forward(inputTensor) as RLTensor)
    }
    return qValues
  }

  /**
   * Sample action from actor output with reparameterization trick
   *
   * Must be called within a run() block.
   *
   * @param actorOutput - Output from getActorOutput() [batch, actionDim * 2]
   * @param deterministic - If true, return mean action (no sampling)
   * @returns Object with action tensor [batch, actionDim] and log_prob tensor [batch]
   */
  sampleActionTensor(actorOutput: RLTensor, deterministic: boolean = false): { actionTensor: RLTensor; logProbTensor: RLTensor } {
    // Split actor output into mean and log_std
    // actorOutput is [batch, actionDim * 2]
    const batchSize = actorOutput.shape[0]!
    
    // Split the tensor using narrow: narrow(dim, start, length)
    const meanTensor = actorOutput.narrow(1, 0, this.actionDim) as RLTensor
    const logStdRaw = actorOutput.narrow(1, this.actionDim, this.actionDim) as RLTensor
    
    // Clamp log_std to valid range
    const logStdTensor = logStdRaw.clamp(LOG_STD_MIN, LOG_STD_MAX) as RLTensor
    const stdTensor = logStdTensor.exp() as RLTensor
    
    let actionTensor: RLTensor
    let logProbTensor: RLTensor
    
    if (deterministic) {
      // Deterministic action: tanh(mean)
      actionTensor = meanTensor.tanh() as RLTensor
      // Log prob is 0 for deterministic (no entropy term)
      logProbTensor = cpu.zeros([batchSize]) as RLTensor
    } else {
      // Reparameterization trick: action = tanh(mean + std * noise)
      const noiseTensor = cpu.randn([batchSize, this.actionDim]) as RLTensor
      const preSquash = meanTensor.add(stdTensor.mul(noiseTensor)) as RLTensor
      actionTensor = preSquash.tanh() as RLTensor
      
      // Compute log probability with squashing correction
      // log_prob = sum(-0.5 * noise^2 - 0.5 * log(2*pi) - log_std) - sum(log(1 - tanh^2(pre_squash)))
      // Simplified: log_prob = sum(-0.5 * ((x - mean) / std)^2 - log_std - 0.5 * log(2*pi) - log(1 - tanh^2))
      
      const LOG_2PI = Math.log(2 * Math.PI)
      
      // Gaussian log probability per dimension
      const logProbGaussian = noiseTensor.mul(noiseTensor).mulScalar(-0.5)
        .sub(logStdTensor)
        .subScalar(LOG_2PI * 0.5) as RLTensor
      
      // Squashing correction: -log(1 - tanh^2(pre_squash) + eps)
      // = -log(1 - action^2 + eps) since action = tanh(pre_squash)
      const actionSquared = actionTensor.mul(actionTensor) as RLTensor
      const oneMinusActionSq = (cpu.ones([batchSize, this.actionDim]) as RLTensor).sub(actionSquared) as RLTensor
      const squashCorrection = oneMinusActionSq.clampMin(1e-6).log() as RLTensor
      
      // Total log prob: sum over action dimensions
      const logProbTotal = logProbGaussian.sub(squashCorrection) as RLTensor
      logProbTensor = logProbTotal.sumDim(1, false) as RLTensor
    }
    
    // Scale action to action space bounds
    const scaleTensor = cpu.tensor(this.actionScale, [1, this.actionDim]) as RLTensor
    const biasTensor = cpu.tensor(this.actionBias, [1, this.actionDim]) as RLTensor
    const scaledAction = actionTensor.mul(scaleTensor).add(biasTensor) as RLTensor
    
    return { actionTensor: scaledAction, logProbTensor }
  }

  /**
   * Compute log probability of given actions under current policy
   *
   * Must be called within a run() block.
   *
   * @param actorOutput - Output from getActorOutput() [batch, actionDim * 2]
   * @param actions - Action tensor [batch, actionDim] (already scaled to action space)
   * @returns Log probability tensor [batch]
   */
  logProbTensor(actorOutput: any, actions: any): any {
    const batchSize = actorOutput.shape[0]
    
    // Split actor output into mean and log_std using narrow(dim, start, length)
    const meanTensor = actorOutput.narrow(1, 0, this.actionDim)
    const logStdRaw = actorOutput.narrow(1, this.actionDim, this.actionDim)
    const logStdTensor = logStdRaw.clamp(LOG_STD_MIN, LOG_STD_MAX)
    const stdTensor = logStdTensor.exp()
    
    // Unscale actions back to [-1, 1] range
    const scaleTensor = cpu.tensor(this.actionScale, [1, this.actionDim])
    const biasTensor = cpu.tensor(this.actionBias, [1, this.actionDim])
    const unscaledAction = actions.sub(biasTensor).div(scaleTensor)
    
    // Clip to prevent numerical issues with atanh
    const clippedAction = unscaledAction.clamp(-0.999, 0.999)
    
    // Inverse of tanh to get pre-squash value
    // atanh(x) = 0.5 * log((1+x)/(1-x))
    const onePlusX = cpu.ones([batchSize, this.actionDim]).add(clippedAction)
    const oneMinusX = cpu.ones([batchSize, this.actionDim]).sub(clippedAction)
    const preSquash = onePlusX.div(oneMinusX).log().mulScalar(0.5)
    
    // Gaussian log probability
    const diff = preSquash.sub(meanTensor)
    const normalizedDiff = diff.div(stdTensor)
    const LOG_2PI = Math.log(2 * Math.PI)
    const logProbGaussian = normalizedDiff.mul(normalizedDiff).mulScalar(-0.5)
      .sub(logStdTensor)
      .subScalar(LOG_2PI * 0.5)
    
    // Squashing correction
    const actionSquared = clippedAction.mul(clippedAction)
    const oneMinusActionSq = cpu.ones([batchSize, this.actionDim]).sub(actionSquared)
    const squashCorrection = oneMinusActionSq.clampMin(1e-6).log()
    
    // Total log prob
    const logProbTotal = logProbGaussian.sub(squashCorrection)
    return logProbTotal.sumDim(1, false) // [batch]
  }

  /**
   * Get entropy of the current policy (approximate)
   *
   * For Gaussian: entropy = 0.5 * (1 + log(2*pi*e*sigma^2)) = 0.5 + 0.5*log(2*pi) + log_std
   *
   * Must be called within a run() block.
   *
   * @param actorOutput - Output from getActorOutput() [batch, actionDim * 2]
   * @returns Entropy tensor [batch]
   */
  entropyTensor(actorOutput: any): any {
    // Extract log_std using narrow(dim, start, length)
    const logStdRaw = actorOutput.narrow(1, this.actionDim, this.actionDim)
    const logStdTensor = logStdRaw.clamp(LOG_STD_MIN, LOG_STD_MAX)
    
    // Gaussian entropy per dimension: 0.5 * (1 + log(2*pi)) + log_std
    const entropyConst = 0.5 * (1 + Math.log(2 * Math.PI))
    const entropyPerDim = logStdTensor.add(cpu.tensor([entropyConst], [1]))
    
    // Sum over action dimensions
    return entropyPerDim.sumDim(1, false) // [batch]
  }

  /**
   * Get observation size
   */
  getObservationSize(): number {
    return this.observationSize
  }

  /**
   * Get action dimension
   */
  getActionDim(): number {
    return this.actionDim
  }

  /**
   * Get action scale tensor (for rescaling)
   */
  getActionScale(): Float32Array {
    return this.actionScale
  }

  /**
   * Get action bias tensor (for rescaling)
   */
  getActionBias(): Float32Array {
    return this.actionBias
  }
}

// ==================== Factory ====================

/**
 * Build MLP for SAC
 */
function buildSacMlp(
  inputSize: number,
  hiddenSizes: number[],
  outputSize: number,
  activation: 'relu' | 'tanh' | 'gelu',
): SequenceDef {
  const blocks: any[] = []
  
  for (const hidden of hiddenSizes) {
    const block = nn.fc(hidden)
    switch (activation) {
      case 'relu': blocks.push(block.relu()); break
      case 'tanh': blocks.push(block.tanh()); break
      case 'gelu': blocks.push(block.gelu()); break
    }
  }
  blocks.push(nn.fc(outputSize))

  return nn.sequence(nn.input(inputSize), ...blocks)
}

/**
 * Create a SAC policy definition
 *
 * @param config - Policy configuration
 * @returns Policy definition for initialization
 */
export function sacPolicy(config: SACPolicyConfig = {}): SACPolicyDef {
  return {
    init<Dev extends DeviceType>(
      device: DeviceContext<Dev>,
      spaces: SACPolicySpaces,
    ): SACPolicy<Dev> {
      const netArch = config.netArch ?? { pi: [256, 256], qf: [256, 256] }
      const activation = config.activation ?? 'relu'
      const nCritics = config.nCritics ?? 2

      const actionDim = spaces.actionSpace.shape.reduce((a, b) => a * b, 1)

      // Actor outputs mean and log_std (2 * actionDim)
      const actorDef = buildSacMlp(
        spaces.observationSize,
        netArch.pi,
        actionDim * 2,  // mean + log_std
        activation,
      )

      // Critics take (obs, action) and output Q-value
      const criticInputSize = spaces.observationSize + actionDim
      const critics: Module<any, any, any, Dev>[] = []
      const criticTargets: Module<any, any, any, Dev>[] = []

      for (let i = 0; i < nCritics; i++) {
        const criticDef = buildSacMlp(criticInputSize, netArch.qf, 1, activation)
        critics.push(criticDef.init(device))
        criticTargets.push(criticDef.init(device))  // Separate initialization for target
      }

      // Copy critic weights to targets (initial sync)
      for (let i = 0; i < nCritics; i++) {
        const sourceParams = critics[i]!.parameters()
        const targetParams = criticTargets[i]!.parameters()
        for (let j = 0; j < sourceParams.length; j++) {
          const sourceData = (sourceParams[j] as any).data
          const targetData = (targetParams[j] as any).data
          if (targetData && typeof targetData.copy === 'function') {
            targetData.copy(sourceData)
          }
        }
      }

      return new SACPolicy(
        device,
        spaces,
        actorDef.init(device),
        critics,
        criticTargets,
      )
    },
  }
}
