/**
 * Fluent Model Builders
 *
 * Provides a declarative, fluent API for defining neural network architectures.
 * Configuration (cheap, pure JS) is separated from memory allocation (expensive C++ bindings).
 *
 * @example
 * ```ts
 * import { nn } from '@ts-torch/nn'
 * import { device } from '@ts-torch/core'
 *
 * // 1. Define Model (No memory allocated - just JS objects)
 * const VisionConfig = nn.sequence(784,
 *   nn.fc(128).relu().dropout(0.2),
 *   nn.fc(64).gelu(),
 *   nn.fc(10)
 * )
 *
 * // 2. Initialize (Memory allocated here via C++ bindings)
 * const model = VisionConfig.init(device.cuda(0))
 * ```
 */

import { Sequential } from './modules/container.js'
import { Linear } from './modules/linear.js'
import { ReLU, Sigmoid, Tanh, Softmax, LeakyReLU, GELU } from './modules/activation.js'
import { Dropout } from './modules/dropout.js'
import { BatchNorm1d } from './modules/normalization.js'
import { Module, type float32 } from './module.js'
import type { Shape, DeviceType } from '@ts-torch/core'
import type { DeviceContext as DeviceContextType } from '@ts-torch/core'

/**
 * Device context type - matches the DeviceContext from @ts-torch/core
 */
type DeviceContext<Dev extends DeviceType = DeviceType> = DeviceContextType<Dev>

// ==================== Types ====================

/**
 * Supported activation functions
 */
export type ActivationType = 'relu' | 'gelu' | 'sigmoid' | 'tanh' | 'leaky_relu' | 'softmax'

/**
 * Weight initialization strategies
 */
export type InitStrategy = 'kaiming_uniform' | 'kaiming_normal' | 'xavier_uniform' | 'xavier_normal' | 'zeros'

/**
 * Block definition - a chainable configuration object for a single layer block.
 * Pure JS object with no memory allocation until `.init()` is called on the parent sequence.
 */
export interface BlockDef {
  /** Number of output features for the linear layer */
  readonly outFeatures: number
  /** Whether to include bias term (default: true) */
  readonly bias: boolean
  /** Weight initialization strategy (default: 'kaiming_uniform') */
  readonly initStrategy: InitStrategy
  /** Activation function to apply after the linear layer */
  readonly activation: ActivationType | undefined
  /** Options for the activation function (e.g., negativeSlope for LeakyReLU) */
  readonly activationOptions: { negativeSlope?: number; dim?: number } | undefined
  /** Dropout probability (0 = disabled) */
  readonly dropoutP: number | undefined
  /** Whether to apply batch normalization before activation */
  readonly useBatchNorm: boolean | undefined

  // Fluent activation methods
  relu(): BlockDef
  gelu(): BlockDef
  sigmoid(): BlockDef
  tanh(): BlockDef
  leakyRelu(negativeSlope?: number): BlockDef
  softmax(dim?: number): BlockDef

  // Fluent configuration methods
  dropout(p: number): BlockDef
  batchNorm(): BlockDef
  noBias(): BlockDef
  withInit(strategy: InitStrategy): BlockDef
}

/**
 * Sequence definition - holds input size and block configurations.
 * Pure JS object with no memory allocation until `.init()` is called.
 */
export interface SequenceDef {
  /** Input feature size */
  readonly inputSize: number
  /** Array of block definitions */
  readonly blocks: readonly BlockDef[]

  /**
   * Initialize the model on the specified device.
   * This is where memory allocation happens via C++ bindings.
   */
  init<Dev extends DeviceType>(device: DeviceContext<Dev>): Sequential<Shape, Shape, float32, Dev>
}

// ==================== Implementation ====================

/**
 * Implementation of BlockDef - immutable, each method returns a new instance
 */
class BlockDefImpl implements BlockDef {
  readonly activation: ActivationType | undefined
  readonly activationOptions: { negativeSlope?: number; dim?: number } | undefined
  readonly dropoutP: number | undefined
  readonly useBatchNorm: boolean | undefined

  constructor(
    readonly outFeatures: number,
    readonly bias: boolean = true,
    readonly initStrategy: InitStrategy = 'kaiming_uniform',
    activation?: ActivationType,
    activationOptions?: { negativeSlope?: number; dim?: number },
    dropoutP?: number,
    useBatchNorm?: boolean,
  ) {
    this.activation = activation
    this.activationOptions = activationOptions
    this.dropoutP = dropoutP
    this.useBatchNorm = useBatchNorm
  }

  private clone(overrides: {
    outFeatures?: number
    bias?: boolean
    initStrategy?: InitStrategy
    activation?: ActivationType | undefined
    activationOptions?: { negativeSlope?: number; dim?: number } | undefined
    dropoutP?: number
    useBatchNorm?: boolean
  }): BlockDef {
    return new BlockDefImpl(
      overrides.outFeatures ?? this.outFeatures,
      overrides.bias ?? this.bias,
      overrides.initStrategy ?? this.initStrategy,
      'activation' in overrides ? overrides.activation : this.activation,
      'activationOptions' in overrides ? overrides.activationOptions : this.activationOptions,
      overrides.dropoutP ?? this.dropoutP,
      overrides.useBatchNorm ?? this.useBatchNorm,
    )
  }

  // Activation methods
  relu(): BlockDef {
    return this.clone({ activation: 'relu', activationOptions: undefined })
  }

  gelu(): BlockDef {
    return this.clone({ activation: 'gelu', activationOptions: undefined })
  }

  sigmoid(): BlockDef {
    return this.clone({ activation: 'sigmoid', activationOptions: undefined })
  }

  tanh(): BlockDef {
    return this.clone({ activation: 'tanh', activationOptions: undefined })
  }

  leakyRelu(negativeSlope = 0.01): BlockDef {
    return this.clone({ activation: 'leaky_relu', activationOptions: { negativeSlope } })
  }

  softmax(dim = -1): BlockDef {
    return this.clone({ activation: 'softmax', activationOptions: { dim } })
  }

  // Configuration methods
  dropout(p: number): BlockDef {
    return this.clone({ dropoutP: p })
  }

  batchNorm(): BlockDef {
    return this.clone({ useBatchNorm: true })
  }

  noBias(): BlockDef {
    return this.clone({ bias: false })
  }

  withInit(strategy: InitStrategy): BlockDef {
    return this.clone({ initStrategy: strategy })
  }
}

/**
 * Implementation of SequenceDef
 */
class SequenceDefImpl implements SequenceDef {
  constructor(
    readonly inputSize: number,
    readonly blocks: readonly BlockDef[],
  ) {}

  init<Dev extends DeviceType>(device: DeviceContext<Dev>): Sequential<Shape, Shape, float32, Dev> {
    const layers: Module<any, any, float32, 'cpu'>[] = []
    let prevSize = this.inputSize

    for (const block of this.blocks) {
      // Linear layer
      layers.push(new Linear(prevSize, block.outFeatures, {
        bias: block.bias,
        init: block.initStrategy,
      }))

      // BatchNorm (if enabled) - applied before activation
      if (block.useBatchNorm) {
        layers.push(new BatchNorm1d(block.outFeatures) as Module<any, any, float32, 'cpu'>)
      }

      // Activation (if specified)
      if (block.activation) {
        layers.push(createActivation(block.activation, block.activationOptions) as Module<any, any, float32, 'cpu'>)
      }

      // Dropout (if specified)
      if (block.dropoutP !== undefined && block.dropoutP > 0) {
        layers.push(new Dropout(block.dropoutP) as Module<any, any, float32, 'cpu'>)
      }

      prevSize = block.outFeatures
    }

    // Create on CPU and move to target device
    const cpuModel = new Sequential<Shape, Shape, float32, 'cpu'>(...layers)
    return cpuModel.to(device.type) as unknown as Sequential<Shape, Shape, float32, Dev>
  }
}

/**
 * Create activation module from type and options
 * @internal
 */
function createActivation(
  type: ActivationType,
  options?: { negativeSlope?: number; dim?: number },
): Module<any, any, any> {
  switch (type) {
    case 'relu':
      return new ReLU()
    case 'gelu':
      return new GELU()
    case 'tanh':
      return new Tanh()
    case 'sigmoid':
      return new Sigmoid()
    case 'leaky_relu':
      return new LeakyReLU(options?.negativeSlope ?? 0.01)
    case 'softmax':
      return new Softmax(options?.dim ?? -1)
    default:
      throw new Error(`Unknown activation: ${type}`)
  }
}

// ==================== Factory Functions ====================

/**
 * Create a fully-connected (linear) block definition.
 *
 * This returns a pure JS configuration object - no memory is allocated.
 * Chain methods like `.relu()`, `.dropout()` to configure the block.
 *
 * @param outFeatures - Number of output features
 * @returns BlockDef that can be chained and used in `nn.sequence()`
 *
 * @example
 * ```ts
 * // Simple linear block
 * nn.fc(128)
 *
 * // With activation
 * nn.fc(128).relu()
 *
 * // Full configuration
 * nn.fc(128).relu().dropout(0.2).noBias().withInit('xavier_uniform')
 * ```
 */
export function fc(outFeatures: number): BlockDef {
  return new BlockDefImpl(outFeatures)
}

/**
 * Create a sequence definition from an input size and block definitions.
 *
 * This returns a pure JS configuration object - no memory is allocated.
 * Call `.init(device)` to compile the model and allocate memory.
 *
 * @param inputSize - Number of input features
 * @param blocks - Block definitions created with `nn.fc()`
 * @returns SequenceDef that can be initialized with `.init(device)`
 *
 * @example
 * ```ts
 * const config = nn.sequence(784,
 *   nn.fc(128).relu().dropout(0.2),
 *   nn.fc(64).gelu(),
 *   nn.fc(10)
 * )
 *
 * const model = config.init(device.cuda(0))
 * ```
 */
export function sequence(inputSize: number, ...blocks: BlockDef[]): SequenceDef {
  if (blocks.length === 0) {
    throw new Error('Sequence requires at least one block')
  }
  return new SequenceDefImpl(inputSize, blocks)
}

// ==================== nn Namespace ====================

/**
 * Neural network namespace - fluent model building
 *
 * @example
 * ```ts
 * import { nn } from '@ts-torch/nn'
 * import { device } from '@ts-torch/core'
 *
 * // Define model architecture (no memory allocated)
 * const config = nn.sequence(784,
 *   nn.fc(128).relu().dropout(0.2),
 *   nn.fc(64).gelu(),
 *   nn.fc(10)
 * )
 *
 * // Initialize model on device (memory allocated here)
 * const model = config.init(device.cuda(0))
 * ```
 */
export const nn = {
  sequence,
  fc,
}
