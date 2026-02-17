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

import { Sequential, HeadedSequential } from './modules/container.js'
import { Linear, FusedLinear } from './modules/linear.js'
import { ReLU, Sigmoid, Tanh, Softmax, LeakyReLU, GELU } from './modules/activation.js'
import { Dropout, Dropout2d } from './modules/dropout.js'
import { BatchNorm1d, BatchNorm2d } from './modules/normalization.js'
import { Conv2d } from './modules/conv.js'
import { MaxPool2d, AvgPool2d, AdaptiveAvgPool2d } from './modules/pooling.js'
import { Embedding } from './modules/embedding.js'
import { TransformerEncoderLayer, TransformerEncoder } from './modules/transformer.js'
import { Flatten } from './modules/flatten.js'
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
 * Block definition - a chainable configuration object for a single FC layer block.
 * Pure JS object with no memory allocation until `.init()` is called on the parent sequence.
 */
export interface BlockDef {
  readonly kind: 'fc'
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
 * Conv2d block definition
 */
export interface Conv2dBlockDef {
  readonly kind: 'conv2d'
  readonly outChannels: number
  readonly kernelSize: number
  readonly stride: number
  readonly padding: number
  readonly dilation: number
  readonly groups: number
  readonly bias: boolean
  readonly activation: ActivationType | undefined
  readonly activationOptions: { negativeSlope?: number; dim?: number } | undefined
  readonly dropoutP: number | undefined
  readonly useBatchNorm: boolean | undefined

  relu(): Conv2dBlockDef
  gelu(): Conv2dBlockDef
  sigmoid(): Conv2dBlockDef
  tanh(): Conv2dBlockDef
  leakyRelu(negativeSlope?: number): Conv2dBlockDef
  softmax(dim?: number): Conv2dBlockDef
  dropout(p: number): Conv2dBlockDef
  batchNorm(): Conv2dBlockDef
  noBias(): Conv2dBlockDef
  withStride(s: number): Conv2dBlockDef
  withPadding(p: number): Conv2dBlockDef
}

/**
 * Pooling block definition
 */
export interface PoolBlockDef {
  readonly kind: 'maxPool2d' | 'avgPool2d' | 'adaptiveAvgPool2d'
  readonly kernelSize: number | undefined
  readonly stride: number | undefined
  readonly padding: number | undefined
  readonly outputSize: number | undefined
}

/**
 * Flatten block definition
 */
export interface FlattenBlockDef {
  readonly kind: 'flatten'
  readonly startDim: number
  readonly endDim: number
}

/**
 * Embedding block definition
 */
export interface EmbeddingBlockDef {
  readonly kind: 'embedding'
  readonly numEmbeddings: number
  readonly embeddingDim: number
  readonly paddingIdx: number | undefined
}

/**
 * Transformer encoder block definition
 */
export interface TransformerEncoderBlockDef {
  readonly kind: 'transformerEncoder'
  readonly nHead: number
  readonly numLayers: number
  readonly dimFeedforward: number | undefined
  readonly dropout: number | undefined
  readonly activation: 'relu' | 'gelu' | undefined
  readonly normFirst: boolean | undefined
}

/**
 * Headless sequence definition — used inside nn.heads() for head branches.
 * No inputDef — input shape is inferred from the shared layers' output at init() time.
 */
export interface HeadlessSequenceDef {
  readonly blocks: readonly AnyBlockDef[]
}

/**
 * Heads block definition — terminal block that splits into named head branches.
 * Must be the last block in a sequence.
 */
export interface HeadsBlockDef {
  readonly kind: 'heads'
  readonly headDefs: Record<string, HeadlessSequenceDef>
  readonly defaultHead?: string
}

/**
 * Union of all block definition types
 */
export type AnyBlockDef =
  | BlockDef
  | Conv2dBlockDef
  | PoolBlockDef
  | FlattenBlockDef
  | EmbeddingBlockDef
  | TransformerEncoderBlockDef
  | HeadsBlockDef

/**
 * Sequence definition - holds input definition and block configurations.
 * Pure JS object with no memory allocation until `.init()` is called.
 */
export interface SequenceDef {
  /** Input definition describing sample shape */
  readonly inputDef: InputDef
  /** Array of block definitions */
  readonly blocks: readonly AnyBlockDef[]

  /**
   * Initialize the model on the specified device.
   * This is where memory allocation happens via C++ bindings.
   */
  init<Dev extends DeviceType>(device: DeviceContext<Dev>): Sequential<Shape, Shape, float32, Dev>

  /**
   * Load model from a directory (config.json + model.safetensors).
   * Architecture comes from this config definition; weights come from the directory.
   */
  load<Dev extends DeviceType>(
    device: DeviceContext<Dev>,
    directory: string,
  ): Promise<{ model: Sequential<Shape, Shape, float32, Dev>; metadata: Record<string, unknown> }>

  /**
   * Serialize this config to a JSON-compatible object.
   */
  toJSON(): object
}

/**
 * Identity module — passes input through unchanged.
 * Used internally when a HeadedSequential has no shared layers.
 * @internal
 */
class Identity extends Module<any, any, any, any> {
  forward(input: any): any {
    return input
  }
}

// ==================== Shape Tracking ====================

type ShapeState =
  | { mode: '1d'; features: number }
  | { mode: 'spatial'; c: number; h: number; w: number }
  | { mode: 'sequence'; embedDim: number }

function shapeStateFromInput(shape: number[]): ShapeState {
  if (shape.length === 3) {
    return { mode: 'spatial', c: shape[0]!, h: shape[1]!, w: shape[2]! }
  }
  if (shape.length === 1) {
    return { mode: '1d', features: shape[0]! }
  }
  // For 2+ dims that aren't spatial, treat as 1d with flattened features
  return { mode: '1d', features: shape.reduce((a, b) => a * b, 1) }
}

function convOutputDim(input: number, kernel: number, stride: number, padding: number, dilation: number): number {
  return Math.floor((input + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1)
}

// ==================== Implementation ====================

/**
 * Implementation of BlockDef - immutable, each method returns a new instance
 */
class BlockDefImpl implements BlockDef {
  readonly kind = 'fc' as const
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
 * Implementation of Conv2dBlockDef
 */
class Conv2dBlockDefImpl implements Conv2dBlockDef {
  readonly kind = 'conv2d' as const
  readonly activation: ActivationType | undefined
  readonly activationOptions: { negativeSlope?: number; dim?: number } | undefined
  readonly dropoutP: number | undefined
  readonly useBatchNorm: boolean | undefined

  constructor(
    readonly outChannels: number,
    readonly kernelSize: number,
    readonly stride: number = 1,
    readonly padding: number = 0,
    readonly dilation: number = 1,
    readonly groups: number = 1,
    readonly bias: boolean = true,
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

  private clone(
    overrides: Partial<{
      outChannels: number
      kernelSize: number
      stride: number
      padding: number
      dilation: number
      groups: number
      bias: boolean
      activation: ActivationType | undefined
      activationOptions: { negativeSlope?: number; dim?: number } | undefined
      dropoutP: number
      useBatchNorm: boolean
    }>,
  ): Conv2dBlockDef {
    return new Conv2dBlockDefImpl(
      overrides.outChannels ?? this.outChannels,
      overrides.kernelSize ?? this.kernelSize,
      overrides.stride ?? this.stride,
      overrides.padding ?? this.padding,
      overrides.dilation ?? this.dilation,
      overrides.groups ?? this.groups,
      overrides.bias ?? this.bias,
      'activation' in overrides ? overrides.activation : this.activation,
      'activationOptions' in overrides ? overrides.activationOptions : this.activationOptions,
      overrides.dropoutP ?? this.dropoutP,
      overrides.useBatchNorm ?? this.useBatchNorm,
    )
  }

  relu(): Conv2dBlockDef {
    return this.clone({ activation: 'relu', activationOptions: undefined })
  }
  gelu(): Conv2dBlockDef {
    return this.clone({ activation: 'gelu', activationOptions: undefined })
  }
  sigmoid(): Conv2dBlockDef {
    return this.clone({ activation: 'sigmoid', activationOptions: undefined })
  }
  tanh(): Conv2dBlockDef {
    return this.clone({ activation: 'tanh', activationOptions: undefined })
  }
  leakyRelu(negativeSlope = 0.01): Conv2dBlockDef {
    return this.clone({ activation: 'leaky_relu', activationOptions: { negativeSlope } })
  }
  softmax(dim = -1): Conv2dBlockDef {
    return this.clone({ activation: 'softmax', activationOptions: { dim } })
  }
  dropout(p: number): Conv2dBlockDef {
    return this.clone({ dropoutP: p })
  }
  batchNorm(): Conv2dBlockDef {
    return this.clone({ useBatchNorm: true })
  }
  noBias(): Conv2dBlockDef {
    return this.clone({ bias: false })
  }

  withStride(s: number): Conv2dBlockDef {
    return this.clone({ stride: s })
  }
  withPadding(p: number): Conv2dBlockDef {
    return this.clone({ padding: p })
  }
}

/**
 * Implementation of SequenceDef
 */
/** Current config schema version */
const CONFIG_VERSION = 3

/**
 * Process a single block, appending layers and updating shape state.
 * Shared by SequenceDefImpl.init() and head branch processing.
 * @internal
 */
function processBlock(
  block: Exclude<AnyBlockDef, HeadsBlockDef>,
  state: ShapeState,
  layers: Module<any, any, float32, 'cpu'>[],
  buildFcLayers: (layers: Module<any, any, float32, 'cpu'>[], block: BlockDef, inFeatures: number) => void,
): ShapeState {
  switch (block.kind) {
    case 'fc': {
      if (state.mode !== '1d') {
        throw new Error(
          `nn.fc() requires 1D input (mode='1d'), but current mode is '${state.mode}'. ` +
            `Add nn.flatten() before nn.fc() to convert spatial/sequence data to 1D.`,
        )
      }
      const inFeatures = state.features
      buildFcLayers(layers, block, inFeatures)
      return { mode: '1d', features: block.outFeatures }
    }

    case 'conv2d': {
      if (state.mode !== 'spatial') {
        throw new Error(
          `nn.conv2d() requires spatial input (mode='spatial' with shape [C, H, W]), ` +
            `but current mode is '${state.mode}'. Use nn.input([C, H, W]) for image data.`,
        )
      }
      const inChannels = state.c
      layers.push(
        new Conv2d(inChannels, block.outChannels, block.kernelSize, {
          stride: block.stride,
          padding: block.padding,
          dilation: block.dilation,
          groups: block.groups,
          bias: block.bias,
        }) as Module<any, any, float32, 'cpu'>,
      )

      if (block.useBatchNorm) {
        layers.push(new BatchNorm2d(block.outChannels) as Module<any, any, float32, 'cpu'>)
      }
      if (block.activation) {
        layers.push(createActivation(block.activation, block.activationOptions) as Module<any, any, float32, 'cpu'>)
      }
      if (block.dropoutP !== undefined && block.dropoutP > 0) {
        layers.push(new Dropout2d(block.dropoutP) as Module<any, any, float32, 'cpu'>)
      }

      const outH = convOutputDim(state.h, block.kernelSize, block.stride, block.padding, block.dilation)
      const outW = convOutputDim(state.w, block.kernelSize, block.stride, block.padding, block.dilation)
      return { mode: 'spatial', c: block.outChannels, h: outH, w: outW }
    }

    case 'maxPool2d':
    case 'avgPool2d': {
      if (state.mode !== 'spatial') {
        throw new Error(`nn.${block.kind}() requires spatial input, but current mode is '${state.mode}'.`)
      }
      const ks = block.kernelSize!
      const poolStride = block.stride ?? ks
      const poolPadding = block.padding ?? 0
      if (block.kind === 'maxPool2d') {
        layers.push(new MaxPool2d(ks, { stride: poolStride, padding: poolPadding }) as Module<any, any, float32, 'cpu'>)
      } else {
        layers.push(new AvgPool2d(ks, { stride: poolStride, padding: poolPadding }) as Module<any, any, float32, 'cpu'>)
      }
      const outH = convOutputDim(state.h, ks, poolStride, poolPadding, 1)
      const outW = convOutputDim(state.w, ks, poolStride, poolPadding, 1)
      return { mode: 'spatial', c: state.c, h: outH, w: outW }
    }

    case 'adaptiveAvgPool2d': {
      if (state.mode !== 'spatial') {
        throw new Error(`nn.adaptiveAvgPool2d() requires spatial input, but current mode is '${state.mode}'.`)
      }
      const outSize = block.outputSize!
      layers.push(new AdaptiveAvgPool2d(outSize) as Module<any, any, float32, 'cpu'>)
      return { mode: 'spatial', c: state.c, h: outSize, w: outSize }
    }

    case 'flatten': {
      if (state.mode === 'spatial') {
        layers.push(new Flatten(block.startDim, block.endDim) as Module<any, any, float32, 'cpu'>)
        return { mode: '1d', features: state.c * state.h * state.w }
      } else if (state.mode === 'sequence') {
        layers.push(new Flatten(block.startDim, block.endDim) as Module<any, any, float32, 'cpu'>)
        return { mode: '1d', features: state.embedDim }
      } else {
        layers.push(new Flatten(block.startDim, block.endDim) as Module<any, any, float32, 'cpu'>)
        return state
      }
    }

    case 'embedding': {
      layers.push(
        new Embedding(block.numEmbeddings, block.embeddingDim, {
          paddingIdx: block.paddingIdx ?? null,
        }) as Module<any, any, float32, 'cpu'>,
      )
      return { mode: 'sequence', embedDim: block.embeddingDim }
    }

    case 'transformerEncoder': {
      if (state.mode !== 'sequence') {
        throw new Error(
          `nn.transformerEncoder() requires sequence input (mode='sequence'), ` +
            `but current mode is '${state.mode}'. Add nn.embedding() before nn.transformerEncoder().`,
        )
      }
      const dModel = state.embedDim
      const encOpts: Record<string, unknown> = { batchFirst: true }
      if (block.dimFeedforward !== undefined) encOpts.dimFeedforward = block.dimFeedforward
      if (block.dropout !== undefined) encOpts.dropout = block.dropout
      if (block.activation !== undefined) encOpts.activation = block.activation
      if (block.normFirst !== undefined) encOpts.normFirst = block.normFirst
      const encoderLayer = new TransformerEncoderLayer(dModel, block.nHead, encOpts as any)
      layers.push(new TransformerEncoder(encoderLayer, block.numLayers) as Module<any, any, float32, 'cpu'>)
      return state
    }
  }
}

/**
 * Serialize a single block definition to a JSON-compatible object.
 * @internal
 */
function serializeBlock(block: AnyBlockDef): Record<string, unknown> {
  switch (block.kind) {
    case 'fc': {
      const obj: Record<string, unknown> = { kind: 'fc', outFeatures: block.outFeatures }
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
    }

    case 'conv2d': {
      const obj: Record<string, unknown> = {
        kind: 'conv2d',
        outChannels: block.outChannels,
        kernelSize: block.kernelSize,
      }
      if (block.stride !== 1) obj.stride = block.stride
      if (block.padding !== 0) obj.padding = block.padding
      if (block.dilation !== 1) obj.dilation = block.dilation
      if (block.groups !== 1) obj.groups = block.groups
      if (block.bias === false) obj.bias = false
      if (block.activation !== undefined) obj.activation = block.activation
      if (block.activationOptions !== undefined) {
        if (block.activation === 'leaky_relu' && block.activationOptions.negativeSlope !== undefined) {
          obj.negativeSlope = block.activationOptions.negativeSlope
        }
      }
      if (block.dropoutP !== undefined && block.dropoutP > 0) obj.dropoutP = block.dropoutP
      if (block.useBatchNorm) obj.batchNorm = true
      return obj
    }

    case 'maxPool2d':
    case 'avgPool2d': {
      const obj: Record<string, unknown> = { kind: block.kind, kernelSize: block.kernelSize }
      if (block.stride !== undefined && block.stride !== block.kernelSize) obj.stride = block.stride
      if (block.padding !== undefined && block.padding !== 0) obj.padding = block.padding
      return obj
    }

    case 'adaptiveAvgPool2d': {
      return { kind: 'adaptiveAvgPool2d', outputSize: block.outputSize }
    }

    case 'flatten': {
      const obj: Record<string, unknown> = { kind: 'flatten' }
      if (block.startDim !== 1) obj.startDim = block.startDim
      if (block.endDim !== -1) obj.endDim = block.endDim
      return obj
    }

    case 'embedding': {
      const obj: Record<string, unknown> = {
        kind: 'embedding',
        numEmbeddings: block.numEmbeddings,
        embeddingDim: block.embeddingDim,
      }
      if (block.paddingIdx !== undefined) obj.paddingIdx = block.paddingIdx
      return obj
    }

    case 'transformerEncoder': {
      const obj: Record<string, unknown> = {
        kind: 'transformerEncoder',
        nHead: block.nHead,
        numLayers: block.numLayers,
      }
      if (block.dimFeedforward !== undefined) obj.dimFeedforward = block.dimFeedforward
      if (block.dropout !== undefined) obj.dropout = block.dropout
      if (block.activation !== undefined) obj.activation = block.activation
      if (block.normFirst !== undefined) obj.normFirst = block.normFirst
      return obj
    }

    case 'heads': {
      const headsObj: Record<string, unknown> = {}
      for (const [name, headDef] of Object.entries(block.headDefs)) {
        headsObj[name] = { blocks: headDef.blocks.map((b) => serializeBlock(b)) }
      }
      const obj: Record<string, unknown> = { kind: 'heads', heads: headsObj }
      if (block.defaultHead !== undefined) obj.defaultHead = block.defaultHead
      return obj
    }
  }
}

class SequenceDefImpl implements SequenceDef {
  constructor(
    readonly inputDef: InputDef,
    readonly blocks: readonly AnyBlockDef[],
  ) {}

  init<Dev extends DeviceType>(device: DeviceContext<Dev>): Sequential<Shape, Shape, float32, Dev> {
    let state = shapeStateFromInput(this.inputDef.shape)

    // Check if last block is heads
    const lastBlock = this.blocks[this.blocks.length - 1]
    const hasHeads = lastBlock?.kind === 'heads'
    const regularBlocks = hasHeads ? this.blocks.slice(0, -1) : this.blocks

    // Process shared/regular blocks
    const sharedLayers: Module<any, any, float32, 'cpu'>[] = []
    for (const block of regularBlocks) {
      state = processBlock(block as Exclude<AnyBlockDef, HeadsBlockDef>, state, sharedLayers, this._buildFcLayers)
    }

    if (!hasHeads) {
      // No heads — return Sequential as before
      const cpuModel = new Sequential<Shape, Shape, float32, 'cpu'>(...sharedLayers)
      cpuModel._config = this.toJSON()
      return cpuModel.to(device.type) as unknown as Sequential<Shape, Shape, float32, Dev>
    }

    // Build headed model
    const headsBlock = lastBlock as HeadsBlockDef
    const sharedSeq =
      sharedLayers.length > 0
        ? new Sequential<Shape, Shape, float32, 'cpu'>(...sharedLayers)
        : new Sequential<Shape, Shape, float32, 'cpu'>(new Identity())

    const headSequentials: Record<string, Sequential<Shape, Shape, float32, 'cpu'>> = {}
    for (const [headName, headDef] of Object.entries(headsBlock.headDefs)) {
      const headLayers: Module<any, any, float32, 'cpu'>[] = []
      let headState = { ...state } as ShapeState
      for (const block of headDef.blocks) {
        headState = processBlock(
          block as Exclude<AnyBlockDef, HeadsBlockDef>,
          headState,
          headLayers,
          this._buildFcLayers,
        )
      }
      headSequentials[headName] = new Sequential<Shape, Shape, float32, 'cpu'>(...headLayers)
    }

    const cpuModel = new HeadedSequential<float32, 'cpu'>(sharedSeq, headSequentials, headsBlock.defaultHead)
    cpuModel._config = this.toJSON()
    return cpuModel.to(device.type) as unknown as Sequential<Shape, Shape, float32, Dev>
  }

  private _buildFcLayers(layers: Module<any, any, float32, 'cpu'>[], block: BlockDef, inFeatures: number): void {
    const canFuse =
      !block.useBatchNorm &&
      (block.activation === 'relu' || block.activation === 'sigmoid' || block.activation === 'tanh')

    if (canFuse) {
      layers.push(
        new FusedLinear(inFeatures, block.outFeatures, {
          bias: block.bias,
          init: block.initStrategy,
          activation: block.activation as 'relu' | 'sigmoid' | 'tanh',
        }) as Module<any, any, float32, 'cpu'>,
      )
    } else {
      layers.push(
        new Linear(inFeatures, block.outFeatures, {
          bias: block.bias,
          init: block.initStrategy,
        }),
      )

      if (block.useBatchNorm) {
        layers.push(new BatchNorm1d(block.outFeatures) as Module<any, any, float32, 'cpu'>)
      }

      if (block.activation) {
        layers.push(createActivation(block.activation, block.activationOptions) as Module<any, any, float32, 'cpu'>)
      }
    }

    if (block.dropoutP !== undefined && block.dropoutP > 0) {
      layers.push(new Dropout(block.dropoutP) as Module<any, any, float32, 'cpu'>)
    }
  }

  async load<Dev extends DeviceType>(
    device: DeviceContext<Dev>,
    directory: string,
  ): Promise<{ model: Sequential<Shape, Shape, float32, Dev>; metadata: Record<string, unknown> }> {
    const { join } = await import('node:path')
    const { loadSafetensors, deserializeMetadata } = await import('./safetensors.js')

    const model = this.init(device)
    const { tensors, metadata } = await loadSafetensors(join(directory, 'model.safetensors'))
    model.loadStateDict(tensors)
    return { model, metadata: deserializeMetadata(metadata) }
  }

  toJSON(): object {
    const hasHeads = this.blocks.some((b) => b.kind === 'heads')

    const blocks = this.blocks.map((block) => serializeBlock(block))

    return {
      format: 'ts-torch-sequence',
      version: hasHeads ? CONFIG_VERSION : 2,
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
 * @param outFeatures - Number of output features
 * @returns BlockDef that can be chained and used in `nn.sequence()`
 */
export function fc(outFeatures: number): BlockDef {
  return new BlockDefImpl(outFeatures)
}

/**
 * Create a Conv2d block definition.
 *
 * @param outChannels - Number of output channels
 * @param kernelSize - Size of the convolution kernel
 * @param options - Optional stride, padding, dilation, groups, bias
 */
export function conv2d(
  outChannels: number,
  kernelSize: number,
  options?: { stride?: number; padding?: number; dilation?: number; groups?: number; bias?: boolean },
): Conv2dBlockDef {
  return new Conv2dBlockDefImpl(
    outChannels,
    kernelSize,
    options?.stride,
    options?.padding,
    options?.dilation,
    options?.groups,
    options?.bias,
  )
}

/**
 * Create a MaxPool2d block definition.
 *
 * @param kernelSize - Pooling window size
 * @param options - Optional stride and padding
 */
export function maxPool2d(kernelSize: number, options?: { stride?: number; padding?: number }): PoolBlockDef {
  return {
    kind: 'maxPool2d',
    kernelSize,
    stride: options?.stride,
    padding: options?.padding,
    outputSize: undefined,
  }
}

/**
 * Create an AvgPool2d block definition.
 *
 * @param kernelSize - Pooling window size
 * @param options - Optional stride and padding
 */
export function avgPool2d(kernelSize: number, options?: { stride?: number; padding?: number }): PoolBlockDef {
  return {
    kind: 'avgPool2d',
    kernelSize,
    stride: options?.stride,
    padding: options?.padding,
    outputSize: undefined,
  }
}

/**
 * Create an AdaptiveAvgPool2d block definition.
 *
 * @param outputSize - Target output size
 */
export function adaptiveAvgPool2d(outputSize: number): PoolBlockDef {
  return {
    kind: 'adaptiveAvgPool2d',
    kernelSize: undefined,
    stride: undefined,
    padding: undefined,
    outputSize,
  }
}

/**
 * Create a Flatten block definition.
 *
 * @param startDim - First dim to flatten (default: 1)
 * @param endDim - Last dim to flatten (default: -1)
 */
export function flatten(startDim = 1, endDim = -1): FlattenBlockDef {
  return { kind: 'flatten', startDim, endDim }
}

/**
 * Create an Embedding block definition.
 *
 * @param numEmbeddings - Size of the embedding dictionary
 * @param embeddingDim - Size of each embedding vector
 * @param options - Optional paddingIdx
 */
export function embedding(
  numEmbeddings: number,
  embeddingDim: number,
  options?: { paddingIdx?: number },
): EmbeddingBlockDef {
  return {
    kind: 'embedding',
    numEmbeddings,
    embeddingDim,
    paddingIdx: options?.paddingIdx,
  }
}

/**
 * Create a TransformerEncoder block definition.
 *
 * @param nHead - Number of attention heads
 * @param numLayers - Number of encoder layers
 * @param options - Optional dimFeedforward, dropout, activation, normFirst
 */
export function transformerEncoder(
  nHead: number,
  numLayers: number,
  options?: { dimFeedforward?: number; dropout?: number; activation?: 'relu' | 'gelu'; normFirst?: boolean },
): TransformerEncoderBlockDef {
  return {
    kind: 'transformerEncoder',
    nHead,
    numLayers,
    dimFeedforward: options?.dimFeedforward,
    dropout: options?.dropout,
    activation: options?.activation,
    normFirst: options?.normFirst,
  }
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
 * When called without an InputDef as the first argument, creates a headless sequence
 * for use inside nn.heads().
 */
export function sequence(inputDef: InputDef, ...blocks: AnyBlockDef[]): SequenceDef
export function sequence(...blocks: AnyBlockDef[]): HeadlessSequenceDef
export function sequence(...args: (InputDef | AnyBlockDef)[]): SequenceDef | HeadlessSequenceDef {
  const first = args[0]
  if (first && typeof first === 'object' && 'kind' in first && first.kind === 'input') {
    // Full sequence with input
    const inputDef = first as InputDef
    const blocks = args.slice(1) as AnyBlockDef[]
    if (blocks.length === 0) {
      throw new Error('Sequence requires at least one block')
    }
    // Validate: heads block must be last
    for (let i = 0; i < blocks.length; i++) {
      if (blocks[i]!.kind === 'heads' && i !== blocks.length - 1) {
        throw new Error('nn.heads() must be the last block in a sequence')
      }
    }
    return new SequenceDefImpl(inputDef, blocks)
  }
  // Headless sequence (for use inside nn.heads())
  const blocks = args as AnyBlockDef[]
  if (blocks.length === 0) {
    throw new Error('Headless sequence requires at least one block')
  }
  // Heads inside heads not allowed
  for (const b of blocks) {
    if (b.kind === 'heads') {
      throw new Error('nn.heads() cannot be used inside a headless sequence')
    }
  }
  return { blocks }
}

/**
 * Create a heads block definition for multi-head models.
 * Must be the last block in nn.sequence().
 *
 * @param headDefs - Record of head name to headless sequence definition
 * @param options - Optional default head name
 */
export function heads(
  headDefs: Record<string, HeadlessSequenceDef>,
  options?: { defaultHead?: string },
): HeadsBlockDef {
  const keys = Object.keys(headDefs)
  if (keys.length === 0) {
    throw new Error('nn.heads() requires at least one head')
  }
  if (options?.defaultHead && !headDefs[options.defaultHead]) {
    throw new Error(`Default head "${options.defaultHead}" not found in head definitions`)
  }
  const result: HeadsBlockDef = { kind: 'heads', headDefs }
  if (options?.defaultHead) {
    ;(result as { defaultHead: string }).defaultHead = options.defaultHead
  }
  return result
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
const VALID_INIT_STRATEGIES = new Set<string>([
  'kaiming_uniform',
  'kaiming_normal',
  'xavier_uniform',
  'xavier_normal',
  'zeros',
])

/**
 * Parse an FC block from JSON
 */
function parseFcBlock(block: Record<string, unknown>, i: number): BlockDef {
  if (typeof block.outFeatures !== 'number' || !Number.isInteger(block.outFeatures) || block.outFeatures <= 0) {
    throw new Error(`Block ${i}: outFeatures must be a positive integer, got ${block.outFeatures}`)
  }
  if (block.bias !== undefined && typeof block.bias !== 'boolean') {
    throw new Error(`Block ${i}: bias must be a boolean if present`)
  }
  if (block.init !== undefined) {
    if (typeof block.init !== 'string' || !VALID_INIT_STRATEGIES.has(block.init)) {
      throw new Error(`Block ${i}: unknown init strategy "${block.init}"`)
    }
  }
  if (block.activation !== undefined) {
    if (typeof block.activation !== 'string' || !VALID_ACTIVATIONS.has(block.activation)) {
      throw new Error(`Block ${i}: unknown activation "${block.activation}"`)
    }
  }
  if (block.dropoutP !== undefined) {
    if (typeof block.dropoutP !== 'number' || block.dropoutP < 0 || block.dropoutP >= 1) {
      throw new Error(`Block ${i}: dropoutP must be a number in [0, 1), got ${block.dropoutP}`)
    }
  }

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
}

/**
 * Parse a block from JSON based on its kind discriminant
 */
function parseBlock(block: Record<string, unknown>, i: number, version: number): AnyBlockDef {
  // Version 1 blocks have no kind field — treat as FC
  const kind = version === 1 ? 'fc' : (block.kind as string)

  if (!kind) {
    throw new Error(`Block ${i}: missing "kind" field`)
  }

  switch (kind) {
    case 'fc':
      return parseFcBlock(block, i)

    case 'conv2d': {
      if (typeof block.outChannels !== 'number' || block.outChannels <= 0) {
        throw new Error(`Block ${i}: conv2d requires positive outChannels`)
      }
      if (typeof block.kernelSize !== 'number' || block.kernelSize <= 0) {
        throw new Error(`Block ${i}: conv2d requires positive kernelSize`)
      }
      let activationOptions: { negativeSlope?: number; dim?: number } | undefined
      if (block.activation === 'leaky_relu' && block.negativeSlope !== undefined) {
        activationOptions = { negativeSlope: block.negativeSlope as number }
      }
      return new Conv2dBlockDefImpl(
        block.outChannels as number,
        block.kernelSize as number,
        (block.stride as number) ?? 1,
        (block.padding as number) ?? 0,
        (block.dilation as number) ?? 1,
        (block.groups as number) ?? 1,
        (block.bias as boolean) ?? true,
        block.activation as ActivationType | undefined,
        activationOptions,
        block.dropoutP as number | undefined,
        (block.batchNorm as boolean) ?? undefined,
      )
    }

    case 'maxPool2d':
    case 'avgPool2d':
      return {
        kind: kind as 'maxPool2d' | 'avgPool2d',
        kernelSize: block.kernelSize as number,
        stride: block.stride as number | undefined,
        padding: block.padding as number | undefined,
        outputSize: undefined,
      }

    case 'adaptiveAvgPool2d':
      return {
        kind: 'adaptiveAvgPool2d',
        kernelSize: undefined,
        stride: undefined,
        padding: undefined,
        outputSize: block.outputSize as number,
      }

    case 'flatten':
      return {
        kind: 'flatten',
        startDim: (block.startDim as number) ?? 1,
        endDim: (block.endDim as number) ?? -1,
      }

    case 'embedding':
      return {
        kind: 'embedding',
        numEmbeddings: block.numEmbeddings as number,
        embeddingDim: block.embeddingDim as number,
        paddingIdx: block.paddingIdx as number | undefined,
      }

    case 'transformerEncoder':
      return {
        kind: 'transformerEncoder',
        nHead: block.nHead as number,
        numLayers: block.numLayers as number,
        dimFeedforward: block.dimFeedforward as number | undefined,
        dropout: block.dropout as number | undefined,
        activation: block.activation as 'relu' | 'gelu' | undefined,
        normFirst: block.normFirst as boolean | undefined,
      }

    case 'heads': {
      const headsRaw = block.heads
      if (!headsRaw || typeof headsRaw !== 'object') {
        throw new Error(`Block ${i}: heads block requires a "heads" object`)
      }
      const headEntries = Object.entries(headsRaw as Record<string, unknown>)
      if (headEntries.length === 0) {
        throw new Error(`Block ${i}: heads block requires at least one head`)
      }
      const headDefs: Record<string, HeadlessSequenceDef> = {}
      for (const [headName, headRaw] of headEntries) {
        if (!headRaw || typeof headRaw !== 'object') {
          throw new Error(`Block ${i}, head "${headName}": must be an object`)
        }
        const headObj = headRaw as Record<string, unknown>
        const headBlocks = headObj.blocks
        if (!Array.isArray(headBlocks) || headBlocks.length === 0) {
          throw new Error(`Block ${i}, head "${headName}": must have a non-empty "blocks" array`)
        }
        headDefs[headName] = {
          blocks: headBlocks.map((hb: unknown, hi: number) => {
            if (!hb || typeof hb !== 'object') {
              throw new Error(`Block ${i}, head "${headName}", block ${hi}: must be an object`)
            }
            return parseBlock(hb as Record<string, unknown>, hi, version)
          }),
        }
      }
      const headsResult: HeadsBlockDef = { kind: 'heads', headDefs }
      if (typeof block.defaultHead === 'string') {
        ;(headsResult as { defaultHead: string }).defaultHead = block.defaultHead
      }
      return headsResult
    }

    default:
      throw new Error(`Block ${i}: unknown block kind "${kind}"`)
  }
}

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
  if (
    !Array.isArray(shape) ||
    shape.length === 0 ||
    !shape.every((d: unknown) => typeof d === 'number' && Number.isInteger(d) && d > 0)
  ) {
    throw new Error(`Invalid input shape: ${JSON.stringify(shape)} (must be array of positive integers)`)
  }

  // Blocks validation
  const blocks = obj.blocks
  if (!Array.isArray(blocks) || blocks.length === 0) {
    throw new Error('Config must have a non-empty "blocks" array')
  }

  const blockDefs: AnyBlockDef[] = blocks.map((b: unknown, i: number) => {
    if (!b || typeof b !== 'object') {
      throw new Error(`Block ${i} must be an object`)
    }
    return parseBlock(b as Record<string, unknown>, i, version as number)
  })

  return new SequenceDefImpl({ kind: 'input', shape: shape as number[] }, blockDefs)
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

// ==================== Inspection ====================

/**
 * Result of inspecting a saved model directory
 */
export interface InspectResult {
  config: object
  parameters: Record<string, { shape: number[]; dtype: string }>
  metadata: Record<string, unknown>
  fileSize: string
  fileSizeBytes: number
}

/**
 * Format bytes into human-readable string
 */
function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`
}

/**
 * Inspect a saved model directory without loading tensor data.
 * Reads config.json, safetensors header, and file size.
 *
 * @param directory - Path to model directory
 * @returns Inspection result with config, parameter info, metadata, and file size
 */
async function inspect(directory: string): Promise<InspectResult> {
  const { readFile, stat } = await import('node:fs/promises')
  const { join } = await import('node:path')
  const { inspectSafetensorsHeader, deserializeMetadata } = await import('./safetensors.js')

  const configPath = join(directory, 'config.json')
  const safetensorsPath = join(directory, 'model.safetensors')

  const [config, headerResult, fileStat] = await Promise.all([
    readFile(configPath, 'utf-8').then(JSON.parse),
    inspectSafetensorsHeader(safetensorsPath),
    stat(safetensorsPath),
  ])

  return {
    config,
    parameters: headerResult.parameters,
    metadata: deserializeMetadata(headerResult.metadata),
    fileSize: formatBytes(fileStat.size),
    fileSizeBytes: fileStat.size,
  }
}

// ==================== nn Namespace ====================

/**
 * Neural network namespace - fluent model building
 */
export const nn = {
  Module,
  input,
  sequence,
  fc,
  conv2d,
  maxPool2d,
  avgPool2d,
  adaptiveAvgPool2d,
  flatten,
  embedding,
  transformerEncoder,
  heads,
  fromJSON,
  load,
  inspect,
}
