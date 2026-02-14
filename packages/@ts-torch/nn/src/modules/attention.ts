/**
 * Attention mechanisms for transformer models
 *
 * Implements MultiheadAttention as used in "Attention Is All You Need"
 */

import { Module, type Tensor, type float32 } from '../module.js'
import { type DType, type DeviceType, type Shape } from '@ts-torch/core'
import { Linear } from './linear.js'
import { Dropout } from './dropout.js'

/**
 * MultiheadAttention options
 */
export interface MultiheadAttentionOptions<D extends DType<string> = float32> {
  /**
   * Dropout probability on attention weights (default: 0.0)
   */
  dropout?: number

  /**
   * Whether to include bias in projections (default: true)
   */
  bias?: boolean

  /**
   * Whether to add bias to key/value projections (default: true)
   */
  addBiasKv?: boolean

  /**
   * Whether to add zero attention to key/value (default: false)
   */
  addZeroAttn?: boolean

  /**
   * Key dimension if different from embedDim (default: embedDim)
   */
  kdim?: number

  /**
   * Value dimension if different from embedDim (default: embedDim)
   */
  vdim?: number

  /**
   * Whether to enable batch_first mode (default: false)
   * If true: input/output shape is [batch, seq, embed]
   * If false: input/output shape is [seq, batch, embed]
   */
  batchFirst?: boolean

  /**
   * Data type for weights (default: float32)
   */
  dtype?: D
}

/**
 * Multi-head attention mechanism
 *
 * Allows the model to jointly attend to information from different
 * representation subspaces at different positions.
 *
 * MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
 * where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
 *
 * @template EmbedDim - Embedding dimension
 * @template NumHeads - Number of attention heads
 * @template D - Data type (default: float32)
 * @template Dev - Device type (default: 'cpu')
 *
 * @example
 * ```ts
 * // Create attention with 512-dim embeddings and 8 heads
 * const attention = new MultiheadAttention(512, 8);
 *
 * // Self-attention: query = key = value
 * const [output, weights] = attention.forward(x, x, x);
 *
 * // Cross-attention with key_padding_mask
 * const [output, weights] = attention.forward(query, key, value, {
 *   keyPaddingMask: paddingMask,
 *   needWeights: true,
 * });
 * ```
 */
export class MultiheadAttention<
  EmbedDim extends number = number,
  NumHeads extends number = number,
  D extends DType<string> = float32,
  Dev extends DeviceType = 'cpu',
> extends Module<Shape, Shape, D, Dev> {
  /**
   * Total embedding dimension
   */
  readonly embedDim: EmbedDim

  /**
   * Number of attention heads
   */
  readonly numHeads: NumHeads

  /**
   * Dimension of each head (embedDim / numHeads)
   */
  readonly headDim: number

  /**
   * Dropout probability
   */
  readonly dropoutP: number

  /**
   * Whether using batch_first mode
   */
  readonly batchFirst: boolean

  /**
   * Key dimension
   */
  readonly kdim: number

  /**
   * Value dimension
   */
  readonly vdim: number

  /**
   * Linear projection for query, key, value (combined for efficiency)
   */
  private inProj: Linear<number, number, D, Dev>

  /**
   * Output projection
   */
  private outProj: Linear<number, number, D, Dev>

  /**
   * Attention dropout
   */
  private dropout: Dropout<Shape, D, Dev>

  /**
   * Scaling factor for attention scores
   */
  private scale: number

  /**
   * Create a new MultiheadAttention layer
   *
   * @param embedDim - Total dimension of the model
   * @param numHeads - Number of parallel attention heads
   * @param options - Configuration options
   */
  constructor(embedDim: EmbedDim, numHeads: NumHeads, options: MultiheadAttentionOptions<D> = {}) {
    super()

    if (embedDim <= 0) {
      throw new Error(`embedDim must be positive, got ${embedDim}`)
    }
    if (numHeads <= 0) {
      throw new Error(`numHeads must be positive, got ${numHeads}`)
    }
    if (embedDim % numHeads !== 0) {
      throw new Error(`embedDim (${embedDim}) must be divisible by numHeads (${numHeads})`)
    }

    this.embedDim = embedDim
    this.numHeads = numHeads
    this.headDim = embedDim / numHeads
    this.dropoutP = options.dropout ?? 0.0
    this.batchFirst = options.batchFirst ?? false
    this.kdim = options.kdim ?? embedDim
    this.vdim = options.vdim ?? embedDim
    this.scale = Math.sqrt(this.headDim)

    const bias = options.bias ?? true

    // In-projection: projects query, key, value all at once
    // For self-attention where kdim = vdim = embedDim, we use a combined projection
    // Input: embedDim, Output: 3 * embedDim (for Q, K, V)
    this.inProj = new Linear(embedDim, embedDim * 3, { bias }) as unknown as Linear<number, number, D, Dev>
    this.registerModule('in_proj', this.inProj as any)

    // Output projection: embedDim -> embedDim
    this.outProj = new Linear(embedDim, embedDim, { bias }) as unknown as Linear<number, number, D, Dev>
    this.registerModule('out_proj', this.outProj as any)

    // Dropout for attention weights
    this.dropout = new Dropout(this.dropoutP) as unknown as Dropout<Shape, D, Dev>
  }

  /**
   * Forward pass for multi-head attention
   *
   * @param query - Query tensor [L, N, E] or [N, L, E] if batchFirst
   * @param key - Key tensor [S, N, E] or [N, S, E] if batchFirst
   * @param value - Value tensor [S, N, E] or [N, S, E] if batchFirst
   * @param options - Forward options
   * @returns Tuple of [output, attention_weights?]
   *   - output: [L, N, E] or [N, L, E] if batchFirst
   *   - attention_weights: [N, L, S] (only if needWeights=true)
   *
   * Where:
   * - L = target sequence length
   * - S = source sequence length
   * - N = batch size
   * - E = embedding dimension
   */
  forward(
    query: Tensor<Shape, D, Dev>,
    key: Tensor<Shape, D, Dev>,
    value: Tensor<Shape, D, Dev>,
    options: {
      /**
       * Mask to prevent attention to certain positions [L, S]
       * True values are positions that will be masked (filled with -inf)
       */
      attnMask?: Tensor<Shape, DType<'bool'>, Dev>

      /**
       * Mask for padded positions in key [N, S]
       * True values indicate padding positions to ignore
       */
      keyPaddingMask?: Tensor<Shape, DType<'bool'>, Dev>

      /**
       * Whether to return attention weights (default: true)
       */
      needWeights?: boolean

      /**
       * Whether to average attention weights across heads (default: true)
       * Only used when needWeights=true
       */
      averageAttnWeights?: boolean
    } = {},
  ): [Tensor<Shape, D, Dev>, Tensor<Shape, D, Dev> | null] {
    const needWeights = options.needWeights ?? true
    const averageAttnWeights = options.averageAttnWeights ?? true

    const queryHandle = (query as any)._handle
    const keyHandle = (key as any)._handle
    const valueHandle = (value as any)._handle
    if (queryHandle !== keyHandle || queryHandle !== valueHandle) {
      throw new Error('Cross-attention is not implemented yet: query, key, and value must reference the same tensor')
    }

    // Get dimensions
    const queryShape = query.shape as readonly number[]
    let batchSize: number
    let tgtLen: number
    let srcLen: number

    if (this.batchFirst) {
      // [N, L, E] format
      batchSize = queryShape[0] ?? 1
      tgtLen = queryShape[1] ?? 1
      srcLen = (key.shape as readonly number[])[1] ?? 1
    } else {
      // [L, N, E] format
      tgtLen = queryShape[0] ?? 1
      batchSize = queryShape[1] ?? 1
      srcLen = (key.shape as readonly number[])[0] ?? 1
    }

    // Project query, key, value using in_proj
    // For self-attention, all three come from the same source
    let qkv: Tensor<Shape, D, Dev>

    // Reshape for projection: [L*N, E] or [N*L, E]
    const flatSize = tgtLen * batchSize
    const flatQuery = query.reshape([flatSize, this.embedDim]) as Tensor<Shape, D, Dev>

    // Project to get Q, K, V combined
    qkv = this.inProj.forward(flatQuery as any) as unknown as Tensor<Shape, D, Dev>

    // Split into Q, K, V - each of shape [L*N, E]
    // qkv shape: [L*N, 3*E]
    const qkvShape = qkv.shape as readonly number[]
    const qkvFlat = qkv.reshape([qkvShape[0] ?? flatSize, 3, this.embedDim]) as Tensor<Shape, D, Dev>

    // For self-attention, we use the same projected tensor for Q, K, V
    // Each has shape [batchSize * numHeads, seqLen, headDim]

    // Reshape Q, K, V for multi-head attention
    // [L, N, E] -> [N, numHeads, L, headDim]
    const Q = qkvFlat
      .narrow(1, 0, 1)
      .squeeze(1) // [L*N, E] - narrow keeps the dimension, squeeze removes it
      .reshape([tgtLen, batchSize, this.numHeads, this.headDim])
      .transpose(0, 2) // [numHeads, batchSize, tgtLen, headDim]
      .transpose(1, 0) // [batchSize, numHeads, tgtLen, headDim]
      .reshape([batchSize * this.numHeads, tgtLen, this.headDim]) as Tensor<Shape, D, Dev>

    const K = qkvFlat
      .narrow(1, 1, 1)
      .squeeze(1) // [L*N, E]
      .reshape([tgtLen, batchSize, this.numHeads, this.headDim])
      .transpose(0, 2)
      .transpose(1, 0)
      .reshape([batchSize * this.numHeads, tgtLen, this.headDim]) as Tensor<Shape, D, Dev>

    const V = qkvFlat
      .narrow(1, 2, 1)
      .squeeze(1) // [L*N, E]
      .reshape([tgtLen, batchSize, this.numHeads, this.headDim])
      .transpose(0, 2)
      .transpose(1, 0)
      .reshape([batchSize * this.numHeads, tgtLen, this.headDim]) as Tensor<Shape, D, Dev>

    // Compute attention scores: Q @ K^T / sqrt(d_k)
    // Q: [B*H, L, D], K: [B*H, S, D] -> K^T: [B*H, D, S]
    const Kt = K.transpose(1, 2) as Tensor<readonly [number, number, number], D, Dev>
    let attnScores = Q.bmm(Kt as any) as Tensor<Shape, D, Dev>
    attnScores = attnScores.divScalar(this.scale) as Tensor<Shape, D, Dev>

    // Apply attention mask if provided
    if (options.attnMask) {
      // Expand mask to [B*H, L, S]
      const mask = options.attnMask
      attnScores = attnScores.maskedFill(mask as any, -Infinity) as Tensor<Shape, D, Dev>
    }

    // Apply key padding mask if provided
    if (options.keyPaddingMask) {
      // keyPaddingMask: [N, S] -> expand to [N*H, 1, S]
      const mask = options.keyPaddingMask
        .reshape([batchSize, 1, 1, srcLen])
        .expand([batchSize, this.numHeads, 1, srcLen])
        .reshape([batchSize * this.numHeads, 1, srcLen]) as Tensor<Shape, DType<'bool'>, Dev>
      attnScores = attnScores.maskedFill(mask as any, -Infinity) as Tensor<Shape, D, Dev>
    }

    // Softmax over source dimension
    let attnWeights = attnScores.softmax(-1) as Tensor<Shape, D, Dev>

    // Apply dropout to attention weights
    if (this._training && this.dropoutP > 0) {
      attnWeights = this.dropout.forward(attnWeights as any) as unknown as Tensor<Shape, D, Dev>
    }

    // Apply attention to values: attn_weights @ V
    // [B*H, L, S] @ [B*H, S, D] -> [B*H, L, D]
    let attnOutput = attnWeights.bmm(V as any) as Tensor<Shape, D, Dev>

    // Reshape back: [B*H, L, D] -> [L, N, E]
    // First reshape from [B*H, L, D] to [B, H, L, D]
    // Then transpose to [B, L, H, D], clone for contiguous memory, and reshape to [B, L, E]
    attnOutput = attnOutput
      .reshape([batchSize, this.numHeads, tgtLen, this.headDim])
      .transpose(1, 2) // [N, L, H, D]
      .clone() // Ensure contiguous memory layout before reshape
      .reshape([batchSize, tgtLen, this.embedDim]) as Tensor<Shape, D, Dev>

    if (!this.batchFirst) {
      attnOutput = attnOutput.transpose(0, 1) as Tensor<Shape, D, Dev> // [L, N, E]
    }

    // Output projection
    const outputFlat = attnOutput.reshape([flatSize, this.embedDim]) as Tensor<Shape, D, Dev>
    let output = this.outProj.forward(outputFlat as any) as unknown as Tensor<Shape, D, Dev>

    // Reshape to original format
    if (this.batchFirst) {
      output = output.reshape([batchSize, tgtLen, this.embedDim]) as Tensor<Shape, D, Dev>
    } else {
      output = output.reshape([tgtLen, batchSize, this.embedDim]) as Tensor<Shape, D, Dev>
    }

    // Return attention weights if requested
    let attnWeightsOut: Tensor<Shape, D, Dev> | null = null
    if (needWeights) {
      // Reshape weights: [B*H, L, S] -> [N, H, L, S]
      attnWeightsOut = attnWeights.reshape([batchSize, this.numHeads, tgtLen, srcLen]) as Tensor<Shape, D, Dev>

      if (averageAttnWeights) {
        // Average over heads: [N, H, L, S] -> [N, L, S]
        attnWeightsOut = attnWeightsOut.meanDim(1) as Tensor<Shape, D, Dev>
      }
    }

    return [output, attnWeightsOut]
  }

  override toString(): string {
    return `MultiheadAttention(embed_dim=${this.embedDim}, num_heads=${this.numHeads}, dropout=${this.dropoutP}, batch_first=${this.batchFirst})`
  }
}

/**
 * Scaled dot-product attention function
 *
 * Computes attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
 *
 * @param query - Query tensor [batch, ..., seq_q, d]
 * @param key - Key tensor [batch, ..., seq_k, d]
 * @param value - Value tensor [batch, ..., seq_k, d_v]
 * @param options - Attention options
 * @returns Tuple of [output, attention_weights]
 */
export function scaledDotProductAttention<S extends Shape, D extends DType<string>, Dev extends DeviceType>(
  query: Tensor<S, D, Dev>,
  key: Tensor<S, D, Dev>,
  value: Tensor<S, D, Dev>,
  options: {
    attnMask?: Tensor<Shape, DType<'bool'>, Dev>
    dropoutP?: number
    isCausal?: boolean
    scale?: number
  } = {},
): Tensor<S, D, Dev> {
  const queryShape = query.shape as readonly number[]
  const keyShape = key.shape as readonly number[]
  const d = queryShape[queryShape.length - 1] ?? 1
  const scale = options.scale ?? Math.sqrt(d)

  // Q @ K^T - convert negative indices to positive
  const ndim = keyShape.length
  const dim0 = ndim - 2 // -2 in positive form
  const dim1 = ndim - 1 // -1 in positive form
  const kt = key.transpose(dim0, dim1) as Tensor<Shape, D, Dev>
  let attnWeights = query.matmul(kt as any) as Tensor<Shape, D, Dev>
  attnWeights = attnWeights.divScalar(scale) as Tensor<Shape, D, Dev>

  // Apply causal mask if requested
  if (options.isCausal) {
    const seqLen = queryShape[queryShape.length - 2]
    // Create causal mask: upper triangle (excluding diagonal) = True
    const causalMask = (query as any).ones([seqLen, seqLen]).triu(1) as Tensor<Shape, DType<'bool'>, Dev>
    attnWeights = attnWeights.maskedFill(causalMask as any, -Infinity) as Tensor<Shape, D, Dev>
  }

  // Apply provided mask
  if (options.attnMask) {
    attnWeights = attnWeights.maskedFill(options.attnMask as any, -Infinity) as Tensor<Shape, D, Dev>
  }

  // Softmax
  attnWeights = attnWeights.softmax(-1) as Tensor<Shape, D, Dev>

  // attn @ V
  const output = attnWeights.matmul(value as any) as Tensor<S, D, Dev>

  return output
}
