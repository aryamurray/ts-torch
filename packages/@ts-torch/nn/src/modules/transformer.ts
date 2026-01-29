/**
 * Transformer layers
 *
 * Implements TransformerEncoder, TransformerDecoder, and their component layers
 * following the architecture from "Attention Is All You Need".
 */

import { Module, type Tensor, type float32 } from '../module.js'
import { device, type DType, type DeviceType, type Shape } from '@ts-torch/core'
import { MultiheadAttention } from './attention.js'
import { Linear } from './linear.js'
import { LayerNorm } from './normalization.js'
import { Dropout } from './dropout.js'

/**
 * TransformerEncoderLayer options
 */
export interface TransformerEncoderLayerOptions {
  /**
   * Hidden dimension of the feedforward network (default: 2048)
   */
  dimFeedforward?: number

  /**
   * Dropout probability (default: 0.1)
   */
  dropout?: number

  /**
   * Activation function: 'relu' or 'gelu' (default: 'relu')
   */
  activation?: 'relu' | 'gelu'

  /**
   * Whether to use batch_first mode (default: false)
   * If true: input/output shape is [batch, seq, embed]
   * If false: input/output shape is [seq, batch, embed]
   */
  batchFirst?: boolean

  /**
   * Whether to use pre-norm (LayerNorm before attention/FFN) or post-norm
   * (LayerNorm after attention/FFN). Post-norm is the original transformer
   * architecture. Pre-norm is often more stable. (default: false = post-norm)
   */
  normFirst?: boolean
}

/**
 * Single Transformer Encoder Layer
 *
 * Consists of:
 * 1. Multi-head self-attention
 * 2. Feedforward network (two linear layers with activation)
 * 3. Residual connections and layer normalization
 *
 * @template D - Data type (default: float32)
 * @template Dev - Device type (default: 'cpu')
 *
 * @example
 * ```ts
 * // Create an encoder layer with 512-dim model and 8 attention heads
 * const layer = new TransformerEncoderLayer(512, 8);
 *
 * // Forward pass
 * const x = cpu.randn([10, 32, 512]); // [seq, batch, embed]
 * const output = layer.forward(x);
 * ```
 */
export class TransformerEncoderLayer<
  D extends DType<string> = float32,
  Dev extends DeviceType = 'cpu',
> extends Module<Shape, Shape, D, Dev> {
  readonly dModel: number
  readonly nHead: number
  readonly dimFeedforward: number
  readonly dropoutP: number
  readonly activation: 'relu' | 'gelu'
  readonly batchFirst: boolean
  readonly normFirst: boolean

  // Self-attention
  private selfAttn: MultiheadAttention<number, number, D, Dev>

  // Feedforward network
  private linear1: Linear<number, number, D, Dev>
  private linear2: Linear<number, number, D, Dev>

  // Layer normalization
  private norm1: LayerNorm<D>
  private norm2: LayerNorm<D>

  // Dropout
  private dropout: Dropout<Shape, D, Dev>
  private dropout1: Dropout<Shape, D, Dev>
  private dropout2: Dropout<Shape, D, Dev>

  /**
   * Create a new TransformerEncoderLayer
   *
   * @param dModel - Model dimension (embedding size)
   * @param nHead - Number of attention heads
   * @param options - Configuration options
   */
  constructor(
    dModel: number,
    nHead: number,
    options: TransformerEncoderLayerOptions = {},
  ) {
    super()

    if (dModel <= 0) {
      throw new Error(`dModel must be positive, got ${dModel}`)
    }
    if (nHead <= 0) {
      throw new Error(`nHead must be positive, got ${nHead}`)
    }
    if (dModel % nHead !== 0) {
      throw new Error(`dModel (${dModel}) must be divisible by nHead (${nHead})`)
    }

    this.dModel = dModel
    this.nHead = nHead
    this.dimFeedforward = options.dimFeedforward ?? 2048
    this.dropoutP = options.dropout ?? 0.1
    this.activation = options.activation ?? 'relu'
    this.batchFirst = options.batchFirst ?? false
    this.normFirst = options.normFirst ?? false

    // Self-attention
    this.selfAttn = new MultiheadAttention(dModel, nHead, {
      dropout: this.dropoutP,
      batchFirst: this.batchFirst,
    }) as unknown as MultiheadAttention<number, number, D, Dev>
    this.registerModule('self_attn', this.selfAttn as any)

    // Feedforward network
    this.linear1 = new Linear(dModel, this.dimFeedforward) as unknown as Linear<number, number, D, Dev>
    this.linear2 = new Linear(this.dimFeedforward, dModel) as unknown as Linear<number, number, D, Dev>
    this.registerModule('linear1', this.linear1 as any)
    this.registerModule('linear2', this.linear2 as any)

    // Layer normalization
    this.norm1 = new LayerNorm([dModel]) as unknown as LayerNorm<D>
    this.norm2 = new LayerNorm([dModel]) as unknown as LayerNorm<D>
    this.registerModule('norm1', this.norm1 as any)
    this.registerModule('norm2', this.norm2 as any)

    // Dropout layers
    this.dropout = new Dropout(this.dropoutP) as unknown as Dropout<Shape, D, Dev>
    this.dropout1 = new Dropout(this.dropoutP) as unknown as Dropout<Shape, D, Dev>
    this.dropout2 = new Dropout(this.dropoutP) as unknown as Dropout<Shape, D, Dev>
  }

  /**
   * Forward pass
   *
   * @param src - Source sequence [seq, batch, embed] or [batch, seq, embed] if batchFirst
   * @param options - Forward options
   * @returns Output sequence with same shape as input
   */
  forward(
    src: Tensor<Shape, D, Dev>,
    options: {
      /**
       * Attention mask [seq, seq] - True values are masked
       */
      srcMask?: Tensor<Shape, DType<'bool'>, Dev>

      /**
       * Key padding mask [batch, seq] - True values are padding
       */
      srcKeyPaddingMask?: Tensor<Shape, DType<'bool'>, Dev>
    } = {},
  ): Tensor<Shape, D, Dev> {
    let x = src

    if (this.normFirst) {
      // Pre-norm: LayerNorm before attention/FFN
      x = x.add(this.selfAttnBlock(this.norm1.forward(x as any) as Tensor<Shape, D, Dev>, options)) as Tensor<Shape, D, Dev>
      x = x.add(this.ffBlock(this.norm2.forward(x as any) as Tensor<Shape, D, Dev>)) as Tensor<Shape, D, Dev>
    } else {
      // Post-norm: LayerNorm after attention/FFN
      x = this.norm1.forward(x.add(this.selfAttnBlock(x, options)) as any) as Tensor<Shape, D, Dev>
      x = this.norm2.forward(x.add(this.ffBlock(x)) as any) as Tensor<Shape, D, Dev>
    }

    return x
  }

  /**
   * Self-attention block
   */
  private selfAttnBlock(
    x: Tensor<Shape, D, Dev>,
    options: {
      srcMask?: Tensor<Shape, DType<'bool'>, Dev>
      srcKeyPaddingMask?: Tensor<Shape, DType<'bool'>, Dev>
    },
  ): Tensor<Shape, D, Dev> {
    // Build attention options, only including defined values
    const attnOptions: {
      attnMask?: Tensor<Shape, DType<'bool'>, Dev>
      keyPaddingMask?: Tensor<Shape, DType<'bool'>, Dev>
      needWeights: boolean
    } = { needWeights: false }
    if (options.srcMask) attnOptions.attnMask = options.srcMask
    if (options.srcKeyPaddingMask) attnOptions.keyPaddingMask = options.srcKeyPaddingMask

    const [attnOutput] = this.selfAttn.forward(x, x, x, attnOptions)
    return this.dropout1.forward(attnOutput as any) as Tensor<Shape, D, Dev>
  }

  /**
   * Feedforward block
   */
  private ffBlock(x: Tensor<Shape, D, Dev>): Tensor<Shape, D, Dev> {
    // Flatten for linear layers
    const shape = x.shape as readonly number[]
    const flatShape = [-1, this.dModel] as const
    let flat = (x as any).reshape(flatShape) as Tensor<Shape, D, Dev>

    // Linear1 -> activation -> dropout -> Linear2 -> dropout
    flat = this.linear1.forward(flat as any) as unknown as Tensor<Shape, D, Dev>
    flat = this.applyActivation(flat)
    flat = this.dropout.forward(flat as any) as Tensor<Shape, D, Dev>
    flat = this.linear2.forward(flat as any) as unknown as Tensor<Shape, D, Dev>
    flat = this.dropout2.forward(flat as any) as Tensor<Shape, D, Dev>

    // Reshape back
    return (flat as any).reshape(shape as number[]) as Tensor<Shape, D, Dev>
  }

  /**
   * Apply the configured activation function
   */
  private applyActivation(x: Tensor<Shape, D, Dev>): Tensor<Shape, D, Dev> {
    if (this.activation === 'gelu') {
      // GELU approximation
      const x3 = x.mul(x as any).mul(x as any) as Tensor<Shape, D, Dev>
      const inner = x.add(x3.mulScalar(0.044715) as any).mulScalar(Math.sqrt(2 / Math.PI)) as Tensor<Shape, D, Dev>
      const tanh = inner.tanh() as Tensor<Shape, D, Dev>
      return x.mul(tanh.addScalar(1) as any).mulScalar(0.5) as Tensor<Shape, D, Dev>
    } else {
      // ReLU
      return x.relu() as Tensor<Shape, D, Dev>
    }
  }

  override toString(): string {
    return `TransformerEncoderLayer(d_model=${this.dModel}, nhead=${this.nHead}, dim_feedforward=${this.dimFeedforward}, dropout=${this.dropoutP})`
  }
}

/**
 * TransformerEncoder options
 */
export interface TransformerEncoderOptions<D extends DType<string> = float32> {
  /**
   * Optional final layer normalization
   */
  norm?: LayerNorm<D>
}

/**
 * Stack of Transformer Encoder Layers
 *
 * @template D - Data type (default: float32)
 * @template Dev - Device type (default: 'cpu')
 *
 * @example
 * ```ts
 * // Create encoder layer
 * const encoderLayer = new TransformerEncoderLayer(512, 8);
 *
 * // Create encoder with 6 layers
 * const encoder = new TransformerEncoder(encoderLayer, 6);
 *
 * // Forward pass
 * const x = cpu.randn([10, 32, 512]); // [seq, batch, embed]
 * const output = encoder.forward(x);
 * ```
 */
export class TransformerEncoder<
  D extends DType<string> = float32,
  Dev extends DeviceType = 'cpu',
> extends Module<Shape, Shape, D, Dev> {
  readonly numLayers: number

  // Encoder layers
  private layers: TransformerEncoderLayer<D, Dev>[]

  // Optional final layer normalization
  private norm: LayerNorm<D> | null

  /**
   * Create a new TransformerEncoder
   *
   * @param encoderLayer - Single encoder layer to clone
   * @param numLayers - Number of encoder layers
   * @param options - Configuration options
   */
  constructor(
    encoderLayer: TransformerEncoderLayer<D, Dev>,
    numLayers: number,
    options: TransformerEncoderOptions<D> = {},
  ) {
    super()

    if (numLayers <= 0) {
      throw new Error(`numLayers must be positive, got ${numLayers}`)
    }

    this.numLayers = numLayers
    this.norm = options.norm ?? null

    // Create layers (clone the provided layer for each)
    this.layers = []
    for (let i = 0; i < numLayers; i++) {
      // Create new instances with same configuration
      const layer = new TransformerEncoderLayer<D, Dev>(
        encoderLayer.dModel,
        encoderLayer.nHead,
        {
          dimFeedforward: encoderLayer.dimFeedforward,
          dropout: encoderLayer.dropoutP,
          activation: encoderLayer.activation,
          batchFirst: encoderLayer.batchFirst,
          normFirst: encoderLayer.normFirst,
        },
      )
      this.layers.push(layer)
      this.registerModule(`layers.${i}`, layer as any)
    }

    if (this.norm) {
      this.registerModule('norm', this.norm as any)
    }
  }

  /**
   * Forward pass through all encoder layers
   *
   * @param src - Source sequence
   * @param options - Forward options
   * @returns Encoded sequence
   */
  forward(
    src: Tensor<Shape, D, Dev>,
    options: {
      mask?: Tensor<Shape, DType<'bool'>, Dev>
      srcKeyPaddingMask?: Tensor<Shape, DType<'bool'>, Dev>
    } = {},
  ): Tensor<Shape, D, Dev> {
    let output = src

    // Build layer options, only including defined values
    const layerOptions: {
      srcMask?: Tensor<Shape, DType<'bool'>, Dev>
      srcKeyPaddingMask?: Tensor<Shape, DType<'bool'>, Dev>
    } = {}
    if (options.mask) layerOptions.srcMask = options.mask
    if (options.srcKeyPaddingMask) layerOptions.srcKeyPaddingMask = options.srcKeyPaddingMask

    for (const layer of this.layers) {
      output = layer.forward(output, layerOptions)
    }

    if (this.norm) {
      output = this.norm.forward(output as any) as Tensor<Shape, D, Dev>
    }

    return output
  }

  override toString(): string {
    return `TransformerEncoder(num_layers=${this.numLayers})`
  }
}

/**
 * TransformerDecoderLayer options
 */
export interface TransformerDecoderLayerOptions {
  /**
   * Hidden dimension of the feedforward network (default: 2048)
   */
  dimFeedforward?: number

  /**
   * Dropout probability (default: 0.1)
   */
  dropout?: number

  /**
   * Activation function: 'relu' or 'gelu' (default: 'relu')
   */
  activation?: 'relu' | 'gelu'

  /**
   * Whether to use batch_first mode (default: false)
   */
  batchFirst?: boolean

  /**
   * Whether to use pre-norm (default: false = post-norm)
   */
  normFirst?: boolean
}

/**
 * Single Transformer Decoder Layer
 *
 * Consists of:
 * 1. Masked multi-head self-attention
 * 2. Multi-head cross-attention (with encoder output)
 * 3. Feedforward network
 * 4. Residual connections and layer normalization
 *
 * @template D - Data type (default: float32)
 * @template Dev - Device type (default: 'cpu')
 *
 * @example
 * ```ts
 * // Create a decoder layer
 * const layer = new TransformerDecoderLayer(512, 8);
 *
 * // Forward pass
 * const tgt = cpu.randn([10, 32, 512]); // [tgt_seq, batch, embed]
 * const memory = cpu.randn([20, 32, 512]); // [src_seq, batch, embed]
 * const output = layer.forward(tgt, memory);
 * ```
 */
export class TransformerDecoderLayer<
  D extends DType<string> = float32,
  Dev extends DeviceType = 'cpu',
> extends Module<Shape, Shape, D, Dev> {
  readonly dModel: number
  readonly nHead: number
  readonly dimFeedforward: number
  readonly dropoutP: number
  readonly activation: 'relu' | 'gelu'
  readonly batchFirst: boolean
  readonly normFirst: boolean

  // Self-attention
  private selfAttn: MultiheadAttention<number, number, D, Dev>

  // Cross-attention (encoder-decoder attention)
  private multiheadAttn: MultiheadAttention<number, number, D, Dev>

  // Feedforward network
  private linear1: Linear<number, number, D, Dev>
  private linear2: Linear<number, number, D, Dev>

  // Layer normalization
  private norm1: LayerNorm<D>
  private norm2: LayerNorm<D>
  private norm3: LayerNorm<D>

  // Dropout
  private dropout: Dropout<Shape, D, Dev>
  private dropout1: Dropout<Shape, D, Dev>
  private dropout2: Dropout<Shape, D, Dev>
  private dropout3: Dropout<Shape, D, Dev>

  /**
   * Create a new TransformerDecoderLayer
   *
   * @param dModel - Model dimension (embedding size)
   * @param nHead - Number of attention heads
   * @param options - Configuration options
   */
  constructor(
    dModel: number,
    nHead: number,
    options: TransformerDecoderLayerOptions = {},
  ) {
    super()

    if (dModel <= 0) {
      throw new Error(`dModel must be positive, got ${dModel}`)
    }
    if (nHead <= 0) {
      throw new Error(`nHead must be positive, got ${nHead}`)
    }
    if (dModel % nHead !== 0) {
      throw new Error(`dModel (${dModel}) must be divisible by nHead (${nHead})`)
    }

    this.dModel = dModel
    this.nHead = nHead
    this.dimFeedforward = options.dimFeedforward ?? 2048
    this.dropoutP = options.dropout ?? 0.1
    this.activation = options.activation ?? 'relu'
    this.batchFirst = options.batchFirst ?? false
    this.normFirst = options.normFirst ?? false

    // Self-attention (masked)
    this.selfAttn = new MultiheadAttention(dModel, nHead, {
      dropout: this.dropoutP,
      batchFirst: this.batchFirst,
    }) as unknown as MultiheadAttention<number, number, D, Dev>
    this.registerModule('self_attn', this.selfAttn as any)

    // Cross-attention
    this.multiheadAttn = new MultiheadAttention(dModel, nHead, {
      dropout: this.dropoutP,
      batchFirst: this.batchFirst,
    }) as unknown as MultiheadAttention<number, number, D, Dev>
    this.registerModule('multihead_attn', this.multiheadAttn as any)

    // Feedforward network
    this.linear1 = new Linear(dModel, this.dimFeedforward) as unknown as Linear<number, number, D, Dev>
    this.linear2 = new Linear(this.dimFeedforward, dModel) as unknown as Linear<number, number, D, Dev>
    this.registerModule('linear1', this.linear1 as any)
    this.registerModule('linear2', this.linear2 as any)

    // Layer normalization
    this.norm1 = new LayerNorm([dModel]) as unknown as LayerNorm<D>
    this.norm2 = new LayerNorm([dModel]) as unknown as LayerNorm<D>
    this.norm3 = new LayerNorm([dModel]) as unknown as LayerNorm<D>
    this.registerModule('norm1', this.norm1 as any)
    this.registerModule('norm2', this.norm2 as any)
    this.registerModule('norm3', this.norm3 as any)

    // Dropout layers
    this.dropout = new Dropout(this.dropoutP) as unknown as Dropout<Shape, D, Dev>
    this.dropout1 = new Dropout(this.dropoutP) as unknown as Dropout<Shape, D, Dev>
    this.dropout2 = new Dropout(this.dropoutP) as unknown as Dropout<Shape, D, Dev>
    this.dropout3 = new Dropout(this.dropoutP) as unknown as Dropout<Shape, D, Dev>
  }

  /**
   * Forward pass
   *
   * @param tgt - Target sequence [tgt_seq, batch, embed] or [batch, tgt_seq, embed] if batchFirst
   * @param memory - Encoder output [src_seq, batch, embed] or [batch, src_seq, embed] if batchFirst
   * @param options - Forward options
   * @returns Output sequence with same shape as tgt
   */
  forward(
    tgt: Tensor<Shape, D, Dev>,
    memory: Tensor<Shape, D, Dev>,
    options: {
      /**
       * Target attention mask [tgt_seq, tgt_seq]
       */
      tgtMask?: Tensor<Shape, DType<'bool'>, Dev>

      /**
       * Memory attention mask [tgt_seq, src_seq]
       */
      memoryMask?: Tensor<Shape, DType<'bool'>, Dev>

      /**
       * Target key padding mask [batch, tgt_seq]
       */
      tgtKeyPaddingMask?: Tensor<Shape, DType<'bool'>, Dev>

      /**
       * Memory key padding mask [batch, src_seq]
       */
      memoryKeyPaddingMask?: Tensor<Shape, DType<'bool'>, Dev>
    } = {},
  ): Tensor<Shape, D, Dev> {
    let x = tgt

    if (this.normFirst) {
      // Pre-norm
      x = x.add(this.selfAttnBlock(this.norm1.forward(x as any) as Tensor<Shape, D, Dev>, options)) as Tensor<Shape, D, Dev>
      x = x.add(this.crossAttnBlock(this.norm2.forward(x as any) as Tensor<Shape, D, Dev>, memory, options)) as Tensor<Shape, D, Dev>
      x = x.add(this.ffBlock(this.norm3.forward(x as any) as Tensor<Shape, D, Dev>)) as Tensor<Shape, D, Dev>
    } else {
      // Post-norm
      x = this.norm1.forward(x.add(this.selfAttnBlock(x, options)) as any) as Tensor<Shape, D, Dev>
      x = this.norm2.forward(x.add(this.crossAttnBlock(x, memory, options)) as any) as Tensor<Shape, D, Dev>
      x = this.norm3.forward(x.add(this.ffBlock(x)) as any) as Tensor<Shape, D, Dev>
    }

    return x
  }

  /**
   * Self-attention block
   */
  private selfAttnBlock(
    x: Tensor<Shape, D, Dev>,
    options: {
      tgtMask?: Tensor<Shape, DType<'bool'>, Dev>
      tgtKeyPaddingMask?: Tensor<Shape, DType<'bool'>, Dev>
    },
  ): Tensor<Shape, D, Dev> {
    // Build attention options, only including defined values
    const attnOptions: {
      attnMask?: Tensor<Shape, DType<'bool'>, Dev>
      keyPaddingMask?: Tensor<Shape, DType<'bool'>, Dev>
      needWeights: boolean
    } = { needWeights: false }
    if (options.tgtMask) attnOptions.attnMask = options.tgtMask
    if (options.tgtKeyPaddingMask) attnOptions.keyPaddingMask = options.tgtKeyPaddingMask

    const [attnOutput] = this.selfAttn.forward(x, x, x, attnOptions)
    return this.dropout1.forward(attnOutput as any) as Tensor<Shape, D, Dev>
  }

  /**
   * Cross-attention block
   */
  private crossAttnBlock(
    x: Tensor<Shape, D, Dev>,
    memory: Tensor<Shape, D, Dev>,
    options: {
      memoryMask?: Tensor<Shape, DType<'bool'>, Dev>
      memoryKeyPaddingMask?: Tensor<Shape, DType<'bool'>, Dev>
    },
  ): Tensor<Shape, D, Dev> {
    // Build attention options, only including defined values
    const attnOptions: {
      attnMask?: Tensor<Shape, DType<'bool'>, Dev>
      keyPaddingMask?: Tensor<Shape, DType<'bool'>, Dev>
      needWeights: boolean
    } = { needWeights: false }
    if (options.memoryMask) attnOptions.attnMask = options.memoryMask
    if (options.memoryKeyPaddingMask) attnOptions.keyPaddingMask = options.memoryKeyPaddingMask

    const [attnOutput] = this.multiheadAttn.forward(x, memory, memory, attnOptions)
    return this.dropout2.forward(attnOutput as any) as Tensor<Shape, D, Dev>
  }

  /**
   * Feedforward block
   */
  private ffBlock(x: Tensor<Shape, D, Dev>): Tensor<Shape, D, Dev> {
    // Flatten for linear layers
    const shape = x.shape as readonly number[]
    const flatShape = [-1, this.dModel] as const
    let flat = (x as any).reshape(flatShape) as Tensor<Shape, D, Dev>

    // Linear1 -> activation -> dropout -> Linear2 -> dropout
    flat = this.linear1.forward(flat as any) as unknown as Tensor<Shape, D, Dev>
    flat = this.applyActivation(flat)
    flat = this.dropout.forward(flat as any) as Tensor<Shape, D, Dev>
    flat = this.linear2.forward(flat as any) as unknown as Tensor<Shape, D, Dev>
    flat = this.dropout3.forward(flat as any) as Tensor<Shape, D, Dev>

    // Reshape back
    return (flat as any).reshape(shape as number[]) as Tensor<Shape, D, Dev>
  }

  /**
   * Apply the configured activation function
   */
  private applyActivation(x: Tensor<Shape, D, Dev>): Tensor<Shape, D, Dev> {
    if (this.activation === 'gelu') {
      // GELU approximation
      const x3 = x.mul(x as any).mul(x as any) as Tensor<Shape, D, Dev>
      const inner = x.add(x3.mulScalar(0.044715) as any).mulScalar(Math.sqrt(2 / Math.PI)) as Tensor<Shape, D, Dev>
      const tanh = inner.tanh() as Tensor<Shape, D, Dev>
      return x.mul(tanh.addScalar(1) as any).mulScalar(0.5) as Tensor<Shape, D, Dev>
    } else {
      // ReLU
      return x.relu() as Tensor<Shape, D, Dev>
    }
  }

  override toString(): string {
    return `TransformerDecoderLayer(d_model=${this.dModel}, nhead=${this.nHead}, dim_feedforward=${this.dimFeedforward}, dropout=${this.dropoutP})`
  }
}

/**
 * TransformerDecoder options
 */
export interface TransformerDecoderOptions<D extends DType<string> = float32> {
  /**
   * Optional final layer normalization
   */
  norm?: LayerNorm<D>
}

/**
 * Stack of Transformer Decoder Layers
 *
 * @template D - Data type (default: float32)
 * @template Dev - Device type (default: 'cpu')
 *
 * @example
 * ```ts
 * // Create decoder layer
 * const decoderLayer = new TransformerDecoderLayer(512, 8);
 *
 * // Create decoder with 6 layers
 * const decoder = new TransformerDecoder(decoderLayer, 6);
 *
 * // Forward pass
 * const tgt = cpu.randn([10, 32, 512]); // [tgt_seq, batch, embed]
 * const memory = cpu.randn([20, 32, 512]); // [src_seq, batch, embed] from encoder
 * const output = decoder.forward(tgt, memory);
 * ```
 */
export class TransformerDecoder<
  D extends DType<string> = float32,
  Dev extends DeviceType = 'cpu',
> extends Module<Shape, Shape, D, Dev> {
  readonly numLayers: number

  // Decoder layers
  private layers: TransformerDecoderLayer<D, Dev>[]

  // Optional final layer normalization
  private norm: LayerNorm<D> | null

  /**
   * Create a new TransformerDecoder
   *
   * @param decoderLayer - Single decoder layer to clone
   * @param numLayers - Number of decoder layers
   * @param options - Configuration options
   */
  constructor(
    decoderLayer: TransformerDecoderLayer<D, Dev>,
    numLayers: number,
    options: TransformerDecoderOptions<D> = {},
  ) {
    super()

    if (numLayers <= 0) {
      throw new Error(`numLayers must be positive, got ${numLayers}`)
    }

    this.numLayers = numLayers
    this.norm = options.norm ?? null

    // Create layers (clone the provided layer for each)
    this.layers = []
    for (let i = 0; i < numLayers; i++) {
      const layer = new TransformerDecoderLayer<D, Dev>(
        decoderLayer.dModel,
        decoderLayer.nHead,
        {
          dimFeedforward: decoderLayer.dimFeedforward,
          dropout: decoderLayer.dropoutP,
          activation: decoderLayer.activation,
          batchFirst: decoderLayer.batchFirst,
          normFirst: decoderLayer.normFirst,
        },
      )
      this.layers.push(layer)
      this.registerModule(`layers.${i}`, layer as any)
    }

    if (this.norm) {
      this.registerModule('norm', this.norm as any)
    }
  }

  /**
   * Forward pass through all decoder layers
   *
   * @param tgt - Target sequence
   * @param memory - Encoder output
   * @param options - Forward options
   * @returns Decoded sequence
   */
  forward(
    tgt: Tensor<Shape, D, Dev>,
    memory: Tensor<Shape, D, Dev>,
    options: {
      tgtMask?: Tensor<Shape, DType<'bool'>, Dev>
      memoryMask?: Tensor<Shape, DType<'bool'>, Dev>
      tgtKeyPaddingMask?: Tensor<Shape, DType<'bool'>, Dev>
      memoryKeyPaddingMask?: Tensor<Shape, DType<'bool'>, Dev>
    } = {},
  ): Tensor<Shape, D, Dev> {
    let output = tgt

    for (const layer of this.layers) {
      output = layer.forward(output, memory, options)
    }

    if (this.norm) {
      output = this.norm.forward(output as any) as Tensor<Shape, D, Dev>
    }

    return output
  }

  override toString(): string {
    return `TransformerDecoder(num_layers=${this.numLayers})`
  }
}

/**
 * Generate a causal (autoregressive) mask for transformer decoder
 *
 * Creates an upper triangular mask where future positions are masked.
 *
 * @param size - Sequence length
 * @returns Boolean tensor [size, size] where upper triangle is True
 *
 * @example
 * ```ts
 * const mask = generateSquareSubsequentMask(10);
 * // Use with decoder:
 * const output = decoder.forward(tgt, memory, { tgtMask: mask });
 * ```
 */
export function generateSquareSubsequentMask<Dev extends DeviceType = 'cpu'>(
  size: number,
): Tensor<readonly [number, number], DType<'bool'>, Dev> {
  // Create upper triangular matrix (excluding diagonal)
  // True values = masked positions (future positions)
  const cpu = device.cpu()
  const mask = (cpu.ones([size, size]) as any).triu(1) as unknown as Tensor<readonly [number, number], DType<'bool'>, Dev>
  return mask
}
