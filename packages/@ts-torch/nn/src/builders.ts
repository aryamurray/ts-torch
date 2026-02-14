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
 * const VisionConfig = nn.sequence(
 *   nn.input(784),
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
import { Linear, FusedLinear } from './modules/linear.js'
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
 * Input layer definition - declares the shape of a single sample (excludes batch dimension).
 * Used as the first argument to `nn.sequence()`.
 */
export interface InputDef {
  kind: 'input'
  shape: number[]
}

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
 * Sequence definition - holds input definition and block configurations.
 * Pure JS object with no memory allocation until `.init()` is called.
 */
export interface SequenceDef {
  /** Input definition describing sample shape */
  readonly inputDef: InputDef
  /** Array of block definitions */
  readonly blocks: readonly BlockDef[]

  /**
   * Initialize the model on the specified device.
   * This is where memory allocation happens via C++ bindings.
   */
  init<Dev extends DeviceType>(device: DeviceContext<Dev>): Sequential<Shape, Shape, float32, Dev>

  /**
   * Load model from a directory (config.json + model.safetensors).
   * Architecture comes from this config definition; weights come from the directory.
   */
  load<Dev extends DeviceType>(device: DeviceContext<Dev>, directory: string): Promise<{ model: Sequential<Shape, Shape, float32, Dev>; metadata: Record<string, unknown> }>

  /**
   * Serialize this config to a JSON-compatible object.
   */
  toJSON(): object
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
/** Current config schema version */
const CONFIG_VERSION = 1

class SequenceDefImpl implements SequenceDef {
  constructor(
    readonly inputDef: InputDef,
    readonly blocks: readonly BlockDef[],
  ) {}

  init<Dev extends DeviceType>(device: DeviceContext<Dev>): Sequential<Shape, Shape, float32, Dev> {
    const layers: Module<any, any, float32, 'cpu'>[] = []
    let prevSize = this.inputDef.shape.reduce((a, b) => a * b, 1)

    for (const block of this.blocks) {
      // Check if we can use fused linear+activation
      // Fusion is possible when:
      // 1. Activation is fusible (relu, sigmoid, tanh)
      // 2. No batch normalization is enabled
      const canFuse =
        !block.useBatchNorm &&
        (block.activation === 'relu' || block.activation === 'sigmoid' || block.activation === 'tanh')

      if (canFuse) {
        // Use fused linear+activation module
        layers.push(new FusedLinear(prevSize, block.outFeatures, {
          bias: block.bias,
          init: block.initStrategy,
          activation: block.activation as 'relu' | 'sigmoid' | 'tanh',
        }) as Module<any, any, float32, 'cpu'>)
      } else {
        // Use separate linear layer
        layers.push(new Linear(prevSize, block.outFeatures, {
          bias: block.bias,
          init: block.initStrategy,
        }))

        // BatchNorm (if enabled) - applied before activation
        if (block.useBatchNorm) {
          layers.push(new BatchNorm1d(block.outFeatures) as Module<any, any, float32, 'cpu'>)
        }

        // Activation (if specified) - only if not already fused
        if (block.activation) {
          layers.push(createActivation(block.activation, block.activationOptions) as Module<any, any, float32, 'cpu'>)
        }
      }

      // Dropout (if specified) - always applied after linear/activation
      if (block.dropoutP !== undefined && block.dropoutP > 0) {
        layers.push(new Dropout(block.dropoutP) as Module<any, any, float32, 'cpu'>)
      }

      prevSize = block.outFeatures
    }

    // Create on CPU and move to target device
    const cpuModel = new Sequential<Shape, Shape, float32, 'cpu'>(...layers)
    cpuModel._config = this.toJSON()
    return cpuModel.to(device.type) as unknown as Sequential<Shape, Shape, float32, Dev>
  }

  async load<Dev extends DeviceType>(device: DeviceContext<Dev>, directory: string): Promise<{ model: Sequential<Shape, Shape, float32, Dev>; metadata: Record<string, unknown> }> {
    const { join } = await import('node:path')
    const { loadSafetensors, deserializeMetadata } = await import('./safetensors.js')

    const model = this.init(device)
    const { tensors, metadata } = await loadSafetensors(join(directory, 'model.safetensors'))
    model.loadStateDict(tensors)
    return { model, metadata: deserializeMetadata(metadata) }
  }

  toJSON(): object {
    const blocks = this.blocks.map((block) => {
      const obj: Record<string, unknown> = { outFeatures: block.outFeatures }

      // Only serialize non-default values
      if (block.bias === false) obj.bias = false
      if (block.initStrategy !== 'kaiming_uniform') obj.init = block.initStrategy
      if (block.activation !== undefined) obj.activation = block.activation
      if (block.activationOptions !== undefined) {
        if (block.activation === 'leaky_relu' && block.activationOptions.negativeSlope !== undefined) {
          obj.negativeSlope = block.activationOptions.negativeSlope
        }
        if (block.activation === 'softmax' && block.activationOptions.dim !== undefined) {
          obj.dim = block.activationOptions.dim
        }
      }
      if (block.dropoutP !== undefined && block.dropoutP > 0) obj.dropoutP = block.dropoutP
      if (block.useBatchNorm) obj.batchNorm = true

      return obj
    })

    return {
      format: 'ts-torch-sequence',
      version: CONFIG_VERSION,
      input: { shape: [...this.inputDef.shape] },
      blocks,
    }
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
 * Create an input layer definition describing the shape of a single sample.
 * Shape excludes the batch dimension — batching is a data pipeline concern.
 *
 * @example
 * ```ts
 * nn.input(784)           // flat input (sugar for [784])
 * nn.input([1, 28, 28])   // explicit shape (e.g. image)
 * nn.input({ shape: [1, 28, 28] })  // object form
 * ```
 */
export function input(size: number): InputDef
export function input(shape: number[]): InputDef
export function input(opts: { shape: number[] }): InputDef
export function input(arg: number | number[] | { shape: number[] }): InputDef {
  if (typeof arg === 'number') {
    return { kind: 'input', shape: [arg] }
  }
  if (Array.isArray(arg)) {
    return { kind: 'input', shape: arg }
  }
  return { kind: 'input', shape: arg.shape }
}

/**
 * Create a sequence definition from an input definition and block definitions.
 *
 * This returns a pure JS configuration object - no memory is allocated.
 * Call `.init(device)` to compile the model and allocate memory.
 *
 * @param inputDef - Input definition created with `nn.input()`
 * @param blocks - Block definitions created with `nn.fc()`
 * @returns SequenceDef that can be initialized with `.init(device)`
 *
 * @example
 * ```ts
 * const config = nn.sequence(
 *   nn.input(784),
 *   nn.fc(128).relu().dropout(0.2),
 *   nn.fc(64).gelu(),
 *   nn.fc(10)
 * )
 *
 * const model = config.init(device.cuda(0))
 * ```
 */
export function sequence(inputDef: InputDef, ...blocks: BlockDef[]): SequenceDef {
  if (inputDef.kind !== 'input') {
    throw new Error('First argument to sequence() must be an InputDef created with nn.input()')
  }
  if (blocks.length === 0) {
    throw new Error('Sequence requires at least one block')
  }
  return new SequenceDefImpl(inputDef, blocks)
}

// ==================== Config Validation ====================

class UnknownFormatError extends Error {
  constructor(format: string) {
    super(`Unknown config format: "${format}" (expected "ts-torch-sequence")`)
    this.name = 'UnknownFormatError'
  }
}

class UnsupportedConfigVersionError extends Error {
  constructor(version: number) {
    super(`Unsupported config version: ${version} (this reader supports up to ${CONFIG_VERSION})`)
    this.name = 'UnsupportedConfigVersionError'
  }
}

const VALID_ACTIVATIONS = new Set<string>(['relu', 'gelu', 'sigmoid', 'tanh', 'leaky_relu', 'softmax'])
const VALID_INIT_STRATEGIES = new Set<string>(['kaiming_uniform', 'kaiming_normal', 'xavier_uniform', 'xavier_normal', 'zeros'])

/**
 * Reconstruct a SequenceDef from a JSON config object with schema validation.
 *
 * @param json - JSON object (typically from config.json)
 * @returns Validated SequenceDef
 */
export function fromJSON(json: unknown): SequenceDef {
  if (!json || typeof json !== 'object') {
    throw new Error('Config must be a non-null object')
  }

  const obj = json as Record<string, unknown>

  // Format check
  if (obj.format !== 'ts-torch-sequence') {
    throw new UnknownFormatError(String(obj.format ?? ''))
  }

  // Version check
  const version = obj.version
  if (typeof version !== 'number' || !Number.isInteger(version) || version < 1) {
    throw new Error(`Invalid config version: ${version}`)
  }
  if (version > CONFIG_VERSION) {
    throw new UnsupportedConfigVersionError(version)
  }

  // Input validation
  const inputObj = obj.input as Record<string, unknown> | undefined
  if (!inputObj || typeof inputObj !== 'object') {
    throw new Error('Config must have an "input" object')
  }
  const shape = inputObj.shape
  if (!Array.isArray(shape) || shape.length === 0 || !shape.every((d: unknown) => typeof d === 'number' && Number.isInteger(d) && d > 0)) {
    throw new Error(`Invalid input shape: ${JSON.stringify(shape)} (must be array of positive integers)`)
  }

  // Blocks validation
  const blocks = obj.blocks
  if (!Array.isArray(blocks) || blocks.length === 0) {
    throw new Error('Config must have a non-empty "blocks" array')
  }

  const blockDefs: BlockDef[] = blocks.map((b: unknown, i: number) => {
    if (!b || typeof b !== 'object') {
      throw new Error(`Block ${i} must be an object`)
    }
    const block = b as Record<string, unknown>

    // outFeatures (required)
    if (typeof block.outFeatures !== 'number' || !Number.isInteger(block.outFeatures) || block.outFeatures <= 0) {
      throw new Error(`Block ${i}: outFeatures must be a positive integer, got ${block.outFeatures}`)
    }

    // bias (optional, defaults true)
    if (block.bias !== undefined && typeof block.bias !== 'boolean') {
      throw new Error(`Block ${i}: bias must be a boolean if present`)
    }

    // init (optional)
    if (block.init !== undefined) {
      if (typeof block.init !== 'string' || !VALID_INIT_STRATEGIES.has(block.init)) {
        throw new Error(`Block ${i}: unknown init strategy "${block.init}"`)
      }
    }

    // activation (optional)
    if (block.activation !== undefined) {
      if (typeof block.activation !== 'string' || !VALID_ACTIVATIONS.has(block.activation)) {
        throw new Error(`Block ${i}: unknown activation "${block.activation}"`)
      }
    }

    // dropoutP (optional)
    if (block.dropoutP !== undefined) {
      if (typeof block.dropoutP !== 'number' || block.dropoutP < 0 || block.dropoutP >= 1) {
        throw new Error(`Block ${i}: dropoutP must be a number in [0, 1), got ${block.dropoutP}`)
      }
    }

    // Build activation options
    let activationOptions: { negativeSlope?: number; dim?: number } | undefined
    if (block.activation === 'leaky_relu' && block.negativeSlope !== undefined) {
      activationOptions = { negativeSlope: block.negativeSlope as number }
    }
    if (block.activation === 'softmax' && block.dim !== undefined) {
      activationOptions = { dim: block.dim as number }
    }

    return new BlockDefImpl(
      block.outFeatures as number,
      (block.bias as boolean) ?? true,
      (block.init as InitStrategy) ?? 'kaiming_uniform',
      block.activation as ActivationType | undefined,
      activationOptions,
      block.dropoutP as number | undefined,
      (block.batchNorm as boolean) ?? undefined,
    )
  })

  return new SequenceDefImpl(
    { kind: 'input', shape: shape as number[] },
    blockDefs,
  )
}

/**
 * Load a model from a directory (config.json + model.safetensors).
 *
 * @param device - Target device
 * @param directory - Path to directory containing config.json and model.safetensors
 * @returns Object with loaded model and deserialized metadata
 */
async function load<Dev extends DeviceType>(
  device: DeviceContext<Dev>,
  directory: string,
): Promise<{ model: Sequential<Shape, Shape, float32, Dev>; metadata: Record<string, unknown> }> {
  const { readFile } = await import('node:fs/promises')
  const { join } = await import('node:path')
  const { loadSafetensors, deserializeMetadata } = await import('./safetensors.js')

  // 1. Read and parse config.json
  const configPath = join(directory, 'config.json')
  const configJson = JSON.parse(await readFile(configPath, 'utf-8'))
  const config = fromJSON(configJson)

  // 2. Load safetensors weights
  const safetensorsPath = join(directory, 'model.safetensors')
  const { tensors, metadata } = await loadSafetensors(safetensorsPath)

  // 3. Init model and load weights (loadStateDict validates internally)
  const model = config.init(device)
  model.loadStateDict(tensors)

  return { model, metadata: deserializeMetadata(metadata) }
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
 * const config = nn.sequence(
 *   nn.input(784),
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
  input,
  sequence,
  fc,
  fromJSON,
  load,
}
