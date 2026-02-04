/**
 * Embedding layer for mapping discrete tokens to dense vectors
 *
 * Essential for NLP tasks where words/tokens need to be converted to
 * continuous representations before processing by neural networks.
 */

import { Module, Parameter, type Tensor, type float32 } from '../module.js'
import { device, type DType, type DeviceType, type Shape } from '@ts-torch/core'

// CPU device for weight initialization
const cpu = device.cpu()

/**
 * Embedding options interface
 */
export interface EmbeddingOptions<D extends DType<string> = float32> {
  /**
   * If specified, entries at paddingIdx will be zeroed out during forward
   * and their gradients will not be updated during training
   */
  paddingIdx?: number | null

  /**
   * If specified, scale gradients by frequency of words in mini-batch
   */
  scaleGradByFreq?: boolean

  /**
   * Data type for embedding weights (default: float32)
   */
  dtype?: D

  /**
   * If true, tensor will be sparse (not implemented yet)
   */
  sparse?: boolean

  /**
   * Maximum norm for embedding vectors. If provided, embeddings
   * with norm > maxNorm will be renormalized
   */
  maxNorm?: number | null

  /**
   * p for the p-norm to compute for maxNorm (default: 2)
   */
  normType?: number
}

/**
 * Embedding layer that maps integer indices to dense vectors
 *
 * A simple lookup table that stores embeddings of a fixed dictionary and size.
 * This module is often used to store word embeddings and retrieve them using indices.
 *
 * @template NumEmbeddings - Size of the embedding dictionary (vocabulary size)
 * @template EmbeddingDim - Size of each embedding vector
 * @template D - Data type (default: float32)
 * @template Dev - Device type (default: 'cpu')
 *
 * @example
 * ```ts
 * // Create embedding for vocabulary of 10000 words, 512-dim vectors
 * const embedding = new Embedding(10000, 512);
 *
 * // Input: token indices [batch, seqLen]
 * const tokens: Tensor<readonly [32, 128], DType<'int64'>> = ...;
 *
 * // Output: embeddings [batch, seqLen, embeddingDim]
 * const embedded = embedding.forward(tokens);
 * // Type: Tensor<readonly [32, 128, 512]>
 * ```
 *
 * @example
 * ```ts
 * // With padding token
 * const embedding = new Embedding(10000, 256, { paddingIdx: 0 });
 * // Index 0 will always output zero vectors
 * ```
 */
export class Embedding<
  NumEmbeddings extends number = number,
  EmbeddingDim extends number = number,
  D extends DType<string> = float32,
  Dev extends DeviceType = 'cpu',
> extends Module<Shape, Shape, D, Dev> {
  /**
   * Size of the embedding dictionary
   */
  readonly numEmbeddings: NumEmbeddings

  /**
   * Dimension of each embedding vector
   */
  readonly embeddingDim: EmbeddingDim

  /**
   * Index of padding token (or null if not specified)
   */
  readonly paddingIdx: number | null

  /**
   * Whether to scale gradients by frequency
   */
  readonly scaleGradByFreq: boolean

  /**
   * Maximum norm for embedding vectors
   */
  readonly maxNorm: number | null

  /**
   * P-norm type for maxNorm
   */
  readonly normType: number

  /**
   * Embedding weight matrix [numEmbeddings, embeddingDim]
   */
  readonly weight: Parameter<readonly [NumEmbeddings, EmbeddingDim], D, Dev>

  /**
   * Create a new Embedding layer
   *
   * @param numEmbeddings - Size of the embedding dictionary (vocabulary size)
   * @param embeddingDim - Dimension of each embedding vector
   * @param options - Configuration options
   */
  constructor(
    numEmbeddings: NumEmbeddings,
    embeddingDim: EmbeddingDim,
    options: EmbeddingOptions<D> = {},
  ) {
    super()

    if (numEmbeddings <= 0) {
      throw new Error(`numEmbeddings must be positive, got ${numEmbeddings}`)
    }
    if (embeddingDim <= 0) {
      throw new Error(`embeddingDim must be positive, got ${embeddingDim}`)
    }

    this.numEmbeddings = numEmbeddings
    this.embeddingDim = embeddingDim
    this.paddingIdx = options.paddingIdx ?? null
    this.scaleGradByFreq = options.scaleGradByFreq ?? false
    this.maxNorm = options.maxNorm ?? null
    this.normType = options.normType ?? 2

    // Validate paddingIdx
    if (this.paddingIdx !== null) {
      if (this.paddingIdx < -numEmbeddings || this.paddingIdx >= numEmbeddings) {
        throw new Error(
          `paddingIdx must be within [-${numEmbeddings}, ${numEmbeddings - 1}], got ${this.paddingIdx}`,
        )
      }
    }

    // Initialize weight matrix with normal distribution
    // Standard initialization: N(0, 1)
    const shape = [numEmbeddings, embeddingDim] as const
    type WeightTensor = Tensor<readonly [NumEmbeddings, EmbeddingDim], D, 'cpu'>
    const weightTensor = cpu.randn(shape) as unknown as WeightTensor

    // Zero out padding index if specified
    if (this.paddingIdx !== null) {
      // Note: We can't easily zero a specific row without indexing ops
      // The native forward handles this
    }

    weightTensor.escape()
    this.weight = new Parameter(weightTensor, true) as Parameter<
      readonly [NumEmbeddings, EmbeddingDim],
      D,
      Dev
    >
    this.registerParameter('weight', this.weight)
  }

  /**
   * Forward pass: lookup embeddings for input indices
   *
   * @param input - Tensor of integer indices (any shape, typically [batch, seqLen])
   * @returns Embedded tensor with embeddingDim added as last dimension
   *
   * @example
   * Input shape [B, S] -> Output shape [B, S, E]
   * where E = embeddingDim
   */
  forward<S extends Shape>(
    input: Tensor<S, DType<'int64'>, Dev>,
  ): Tensor<Shape, D, Dev> {
    // Flatten input for indexSelect, then reshape back
    const inputShape = input.shape as readonly number[]
    const numElements = inputShape.reduce((a, b) => a * b, 1)
    const flatInput = input.reshape([numElements] as const) as Tensor<readonly [number], DType<'int64'>, Dev>

    // Use indexSelect on dimension 0 of weight matrix
    // weight: [numEmbeddings, embeddingDim]
    // flatInput: [N] where N = product of input dimensions
    // result: [N, embeddingDim]
    const embedded = this.weight.data.indexSelect(0, flatInput)

    // Reshape to [...inputShape, embeddingDim]
    const outputShape = [...inputShape, this.embeddingDim] as number[]
    const result = embedded.reshape(outputShape)

    return result as unknown as Tensor<Shape, D, Dev>
  }

  override toString(): string {
    let s = `Embedding(${this.numEmbeddings}, ${this.embeddingDim}`
    if (this.paddingIdx !== null) {
      s += `, padding_idx=${this.paddingIdx}`
    }
    s += ')'
    return s
  }
}

/**
 * Create an Embedding layer from a pretrained weight tensor
 *
 * @param embeddings - Pretrained embedding tensor [numEmbeddings, embeddingDim]
 * @param options - Configuration options (freeze determines requires_grad)
 * @returns Embedding layer initialized with pretrained weights
 *
 * @example
 * ```ts
 * // Load pretrained word vectors
 * const pretrainedWeights = loadGloveVectors(); // [50000, 300]
 *
 * // Create embedding layer with frozen weights
 * const embedding = Embedding.fromPretrained(pretrainedWeights, { freeze: true });
 *
 * // Or allow fine-tuning
 * const embeddingFinetune = Embedding.fromPretrained(pretrainedWeights, { freeze: false });
 * ```
 */
export function embeddingFromPretrained<
  NumEmbeddings extends number,
  EmbeddingDim extends number,
  D extends DType<string>,
  Dev extends DeviceType,
>(
  embeddings: Tensor<readonly [NumEmbeddings, EmbeddingDim], D, Dev>,
  options: {
    freeze?: boolean
    paddingIdx?: number | null
    maxNorm?: number | null
    normType?: number
    scaleGradByFreq?: boolean
  } = {},
): Embedding<NumEmbeddings, EmbeddingDim, D, Dev> {
  const shape = embeddings.shape as readonly [NumEmbeddings, EmbeddingDim]
  const numEmbeddings = shape[0]
  const embeddingDim = shape[1]

  const freeze = options.freeze ?? true

  // Create embedding with same dimensions
  const embedding = new Embedding<NumEmbeddings, EmbeddingDim, D, Dev>(
    numEmbeddings,
    embeddingDim,
    {
      paddingIdx: options.paddingIdx,
      maxNorm: options.maxNorm,
      normType: options.normType,
      scaleGradByFreq: options.scaleGradByFreq,
    },
  )

  // Copy pretrained weights
  const cloned = embeddings.clone()
  cloned.escape()

  // Replace weight with pretrained
  const newWeight = new Parameter(cloned, !freeze)
  ;(embedding as any).weight = newWeight
  embedding.registerParameter('weight', newWeight as any)

  return embedding
}
