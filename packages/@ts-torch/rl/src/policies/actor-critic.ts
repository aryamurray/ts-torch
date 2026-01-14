/**
 * Actor-Critic Policy
 *
 * Neural network policy for on-policy algorithms (PPO, A2C).
 * Consists of:
 * - Feature extractor (optional shared layers)
 * - Actor network (policy head) - outputs action distribution parameters
 * - Critic network (value head) - outputs state value estimate
 *
 * @example
 * ```ts
 * // Simple MLP policy
 * const policyConfig = actorCriticPolicy({
 *   netArch: { pi: [64, 64], vf: [64, 64] },
 *   activation: 'tanh',
 * })
 *
 * const policy = policyConfig.init(device.cuda(0), {
 *   observationSize: 4,
 *   actionSpace: discrete(2),
 * })
 *
 * const { actions, values, logProbs } = policy.forward(observations)
 * ```
 */

import type { DeviceType } from '@ts-torch/core'
import { run, device as deviceModule } from '@ts-torch/core'
import type { DeviceContext } from '@ts-torch/core'
import type { Module, SequenceDef } from '@ts-torch/nn'
import { nn } from '@ts-torch/nn'
import type { Space, DiscreteSpace, BoxSpace } from '../spaces/index.js'
import { CategoricalDistribution } from '../distributions/categorical.js'
import { DiagGaussianDistribution } from '../distributions/diagonal-gaussian.js'

// CPU device for tensor creation (data is created on CPU then moved if needed)
const cpu = deviceModule.cpu()

// ==================== Types ====================

/**
 * Supported activation functions for policy networks
 */
export type PolicyActivation = 'relu' | 'tanh' | 'gelu'

/**
 * Network architecture specification
 */
export interface NetArch {
  /** Hidden layer sizes for policy (actor) network */
  pi: number[]
  /** Hidden layer sizes for value (critic) network */
  vf: number[]
}

/**
 * Configuration for ActorCriticPolicy
 */
export interface ActorCriticPolicyConfig {
  /**
   * Network architecture.
   * Either specify `netArch` for simple MLP or provide custom `piNet`/`vfNet`.
   */
  netArch?: NetArch

  /** Activation function (default: 'tanh') */
  activation?: PolicyActivation

  /** Custom policy network definition (overrides netArch.pi) */
  piNet?: SequenceDef

  /** Custom value network definition (overrides netArch.vf) */
  vfNet?: SequenceDef

  /** Whether to use orthogonal initialization (default: true) */
  orthoInit?: boolean

  /** Initial value for log_std (continuous actions, default: 0) */
  logStdInit?: number

  /** Whether log_std is state-dependent or shared (default: false = shared) */
  stateDependentStd?: boolean
}

/**
 * Spaces configuration for policy initialization
 */
export interface PolicySpaces {
  /** Size of observation vector */
  observationSize: number
  /** Action space */
  actionSpace: Space
}

/**
 * Definition object for ActorCriticPolicy (before initialization)
 */
export interface ActorCriticPolicyDef {
  /** Initialize the policy on a device */
  init<Dev extends DeviceType>(
    device: DeviceContext<Dev>,
    spaces: PolicySpaces,
  ): ActorCriticPolicy<Dev>
}

/**
 * Forward pass result
 */
export interface ForwardResult {
  /** Sampled actions */
  actions: Float32Array
  /** Value estimates */
  values: Float32Array
  /** Log probabilities of actions */
  logProbs: Float32Array
}

/**
 * Evaluate actions result (for training)
 */
export interface EvaluateActionsResult {
  /** Value estimates */
  values: Float32Array
  /** Log probabilities of actions */
  logProbs: Float32Array
  /** Entropy of action distribution */
  entropy: Float32Array
}

// ==================== Implementation ====================

/**
 * Actor-Critic Policy for on-policy RL
 *
 * Provides a unified interface for:
 * - Getting action distributions from observations
 * - Sampling actions with value estimates
 * - Evaluating log probabilities of existing actions
 */
export class ActorCriticPolicy<Dev extends DeviceType = DeviceType> {
  private readonly device_: DeviceContext<Dev>
  private readonly observationSize: number
  private readonly actionSpace_: Space
  private readonly isDiscrete: boolean

  // Networks
  private piNet: Module<any, any, any, Dev>
  private vfNet: Module<any, any, any, Dev>

  // For continuous actions: learnable log_std
  private logStd: Float32Array | null = null

  // Configuration
  private readonly logStdInit: number

  constructor(
    device: DeviceContext<Dev>,
    spaces: PolicySpaces,
    config: ActorCriticPolicyConfig,
    piNet: Module<any, any, any, Dev>,
    vfNet: Module<any, any, any, Dev>,
  ) {
    this.device_ = device
    this.observationSize = spaces.observationSize
    this.actionSpace_ = spaces.actionSpace
    this.isDiscrete = spaces.actionSpace.type === 'discrete'
    this.piNet = piNet
    this.vfNet = vfNet
    this.logStdInit = config.logStdInit ?? 0

    // Initialize log_std for continuous actions
    if (!this.isDiscrete) {
      const actionDim = (spaces.actionSpace as BoxSpace).shape.reduce((a, b) => a * b, 1)
      this.logStd = new Float32Array(actionDim).fill(this.logStdInit)
    }
  }

  /**
   * Forward pass - get actions, values, and log probabilities
   *
   * @param observations - Batch of observations [batch, obsSize] as flat array
   * @param deterministic - Whether to use deterministic actions (default: false)
   * @returns Actions, values, and log probabilities
   */
  forward(observations: Float32Array, deterministic: boolean = false): ForwardResult {
    const batchSize = observations.length / this.observationSize

    let actions: Float32Array
    let values: Float32Array
    let logProbs: Float32Array

    run(() => {
      // Create observation tensor
      const obsTensor = cpu.tensor(observations, [batchSize, this.observationSize] as const)

      // Get value estimates
      const valueTensor = (this.vfNet as any).forward(obsTensor)
      values = valueTensor.toArray() as Float32Array

      // Get policy output (logits or mean)
      const policyOutput = (this.piNet as any).forward(obsTensor)

      // Create distribution and sample
      if (this.isDiscrete) {
        const dist = new CategoricalDistribution(policyOutput)
        const actionTensor = deterministic ? dist.mode() : dist.sample()
        actions = actionTensor.toArray() as Float32Array
        logProbs = dist.logProb(actionTensor).toArray() as Float32Array
      } else {
        // Continuous: policyOutput is mean, use stored logStd
        const actionDim = (this.actionSpace_ as BoxSpace).shape.reduce((a, b) => a * b, 1)
        const logStdTensor = cpu.tensor(this.logStd!, [actionDim] as const)
        const dist = new DiagGaussianDistribution(policyOutput, logStdTensor)
        const actionTensor = deterministic ? dist.mode() : dist.sample()
        actions = actionTensor.toArray() as Float32Array
        logProbs = dist.logProb(actionTensor).toArray() as Float32Array
      }
    })

    return { actions: actions!, values: values!, logProbs: logProbs! }
  }

  /**
   * Evaluate log probabilities and entropy for given actions
   *
   * Used during training to compute policy loss.
   *
   * @param observations - Batch of observations [batch, obsSize]
   * @param actions - Actions to evaluate [batch] or [batch, actionDim]
   * @returns Values, log probabilities, and entropy
   */
  evaluateActions(observations: Float32Array, actions: Float32Array): EvaluateActionsResult {
    const batchSize = observations.length / this.observationSize

    let values: Float32Array
    let logProbs: Float32Array
    let entropy: Float32Array

    run(() => {
      const obsTensor = cpu.tensor(observations, [batchSize, this.observationSize] as const)

      // Get value estimates
      const valueTensor = (this.vfNet as any).forward(obsTensor)
      values = valueTensor.toArray() as Float32Array

      // Get policy output
      const policyOutput = (this.piNet as any).forward(obsTensor)

      if (this.isDiscrete) {
        const actionTensor = cpu.tensor(actions, [batchSize] as const)
        const dist = new CategoricalDistribution(policyOutput)
        logProbs = dist.logProb(actionTensor).toArray() as Float32Array
        entropy = dist.entropy().toArray() as Float32Array
      } else {
        const actionDim = (this.actionSpace_ as BoxSpace).shape.reduce((a, b) => a * b, 1)
        const actionTensor = cpu.tensor(actions, [batchSize, actionDim] as const)
        const logStdTensor = cpu.tensor(this.logStd!, [actionDim] as const)
        const dist = new DiagGaussianDistribution(policyOutput, logStdTensor)
        logProbs = dist.logProb(actionTensor).toArray() as Float32Array
        entropy = dist.entropy().toArray() as Float32Array
      }
    })

    return { values: values!, logProbs: logProbs!, entropy: entropy! }
  }

  /**
   * Get value estimates only
   *
   * @param observations - Batch of observations
   * @returns Value estimates [batch]
   */
  predictValues(observations: Float32Array): Float32Array {
    const batchSize = observations.length / this.observationSize
    let values: Float32Array

    run(() => {
      const obsTensor = cpu.tensor(observations, [batchSize, this.observationSize] as const)
      const valueTensor = (this.vfNet as any).forward(obsTensor)
      values = valueTensor.toArray() as Float32Array
    })

    return values!
  }

  // ==================== Tensor Methods (for training with autograd) ====================

  /**
   * Get policy network output tensor (logits for discrete, mean for continuous)
   * 
   * Must be called within a run() block. Returns tensor connected to computational graph.
   * 
   * @param obsTensor - Observation tensor [batch, obsSize]
   * @returns Policy output tensor (logits or mean)
   */
  getPolicyOutput(obsTensor: any): any {
    return (this.piNet as any).forward(obsTensor)
  }

  /**
   * Get value network output tensor
   * 
   * Must be called within a run() block. Returns tensor connected to computational graph.
   * 
   * @param obsTensor - Observation tensor [batch, obsSize]
   * @returns Value tensor [batch, 1]
   */
  getValueOutput(obsTensor: any): any {
    return (this.vfNet as any).forward(obsTensor)
  }

  /**
   * Create action distribution from policy output
   * 
   * Must be called within a run() block.
   * 
   * @param policyOutput - Output from getPolicyOutput()
   * @param batchSize - Batch size (needed for continuous actions)
   * @returns Distribution object with logProb(), entropy(), sample() methods
   */
  getDistribution(policyOutput: any, _batchSize?: number): CategoricalDistribution | DiagGaussianDistribution {
    if (this.isDiscrete) {
      return new CategoricalDistribution(policyOutput)
    } else {
      const actionDim = (this.actionSpace_ as BoxSpace).shape.reduce((a, b) => a * b, 1)
      const logStdTensor = cpu.tensor(this.logStd!, [actionDim] as const)
      return new DiagGaussianDistribution(policyOutput, logStdTensor)
    }
  }

  /**
   * Create action tensor from Float32Array
   * 
   * Must be called within a run() block.
   * 
   * @param actions - Actions as Float32Array
   * @param batchSize - Batch size
   * @returns Action tensor
   */
  createActionTensor(actions: Float32Array, batchSize: number): any {
    if (this.isDiscrete) {
      return cpu.tensor(actions, [batchSize] as const)
    } else {
      const actionDim = (this.actionSpace_ as BoxSpace).shape.reduce((a, b) => a * b, 1)
      return cpu.tensor(actions, [batchSize, actionDim] as const)
    }
  }

  /**
   * Get observation size for creating tensors
   */
  getObservationSize(): number {
    return this.observationSize
  }

  /**
   * Check if action space is discrete
   */
  isDiscreteAction(): boolean {
    return this.isDiscrete
  }

  // ==================== Array Methods (for inference) ====================

  /**
   * Get all trainable parameters
   */
  parameters(): any[] {
    return [
      ...this.piNet.parameters(),
      ...this.vfNet.parameters(),
    ]
  }

  /**
   * Set to training mode
   */
  train(): void {
    this.piNet.train()
    this.vfNet.train()
  }

  /**
   * Set to evaluation mode
   */
  eval(): void {
    this.piNet.eval()
    this.vfNet.eval()
  }

  /**
   * Get the action space
   */
  get actionSpace(): Space {
    return this.actionSpace_
  }

  /**
   * Get the device
   */
  get device(): DeviceContext<Dev> {
    return this.device_
  }
}

// ==================== Factory ====================

/**
 * Create a Block definition for activation
 */
function activationBlock(activation: PolicyActivation) {
  return (block: any) => {
    switch (activation) {
      case 'relu': return block.relu()
      case 'tanh': return block.tanh()
      case 'gelu': return block.gelu()
      default: return block.relu()
    }
  }
}

/**
 * Build MLP from layer sizes
 */
function buildMlp(
  inputSize: number,
  hiddenSizes: number[],
  outputSize: number,
  activation: PolicyActivation,
): SequenceDef {
  const applyActivation = activationBlock(activation)

  const blocks = []
  for (let i = 0; i < hiddenSizes.length; i++) {
    blocks.push(applyActivation(nn.fc(hiddenSizes[i]!)))
  }
  blocks.push(nn.fc(outputSize))

  return nn.sequence(inputSize, ...blocks)
}

/**
 * Create an ActorCriticPolicy definition
 *
 * @param config - Policy configuration
 * @returns Policy definition that can be initialized on a device
 *
 * @example
 * ```ts
 * const policyDef = actorCriticPolicy({
 *   netArch: { pi: [64, 64], vf: [64, 64] },
 *   activation: 'tanh',
 * })
 *
 * const policy = policyDef.init(device.cuda(0), {
 *   observationSize: 4,
 *   actionSpace: discrete(2),
 * })
 * ```
 */
export function actorCriticPolicy(config: ActorCriticPolicyConfig = {}): ActorCriticPolicyDef {
  return {
    init<Dev extends DeviceType>(
      device: DeviceContext<Dev>,
      spaces: PolicySpaces,
    ): ActorCriticPolicy<Dev> {
      const activation = config.activation ?? 'tanh'
      const netArch = config.netArch ?? { pi: [64, 64], vf: [64, 64] }

      // Determine output sizes
      let piOutputSize: number
      if (spaces.actionSpace.type === 'discrete') {
        piOutputSize = (spaces.actionSpace as DiscreteSpace).n
      } else {
        // Continuous: output is mean (same dim as action)
        piOutputSize = (spaces.actionSpace as BoxSpace).shape.reduce((a, b) => a * b, 1)
      }
      const vfOutputSize = 1

      // Build networks
      const piDef = config.piNet ?? buildMlp(
        spaces.observationSize,
        netArch.pi,
        piOutputSize,
        activation,
      )

      const vfDef = config.vfNet ?? buildMlp(
        spaces.observationSize,
        netArch.vf,
        vfOutputSize,
        activation,
      )

      // Initialize on device
      const piNet = piDef.init(device)
      const vfNet = vfDef.init(device)

      return new ActorCriticPolicy(device, spaces, config, piNet, vfNet)
    },
  }
}

/**
 * Shorthand for MLP policy
 */
export function mlpPolicy(
  piLayers: number[] = [64, 64],
  vfLayers: number[] = [64, 64],
  activation: PolicyActivation = 'tanh',
): ActorCriticPolicyDef {
  return actorCriticPolicy({
    netArch: { pi: piLayers, vf: vfLayers },
    activation,
  })
}
