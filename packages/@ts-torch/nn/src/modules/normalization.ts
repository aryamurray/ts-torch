/**
 * Normalization layers
 */

import { Module, Parameter, type Tensor, type float32 } from '../module.js'
import { device, validateNormParams, type DType, type Shape } from '@ts-torch/core'

// CPU device for weight initialization
const cpu = device.cpu()

/**
 * Batch normalization layer for 2D inputs (4D tensor: N, C, H, W)
 *
 * Applies Batch Normalization as described in the paper
 * "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"
 *
 * y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
 *
 * During training, this layer keeps running estimates of its computed mean and variance,
 * which are then used during evaluation.
 *
 * @example
 * ```ts
 * // Create a batch normalization layer for 64 channels
 * const bn = new BatchNorm2d(64);
 *
 * // Forward pass
 * const input: Tensor<readonly [32, 64, 28, 28]> = ...;
 * const output = bn.forward(input);
 * ```
 */
export class BatchNorm2d<D extends DType<string> = float32> extends Module<
  readonly [number, number, number, number],
  readonly [number, number, number, number],
  D
> {
  readonly numFeatures: number
  readonly eps: number
  readonly momentum: number
  readonly affine: boolean
  readonly trackRunningStats: boolean

  /**
   * Learnable scale parameter (gamma)
   */
  readonly weight: Parameter<readonly [number], D> | null

  /**
   * Learnable shift parameter (beta)
   */
  readonly biasParam: Parameter<readonly [number], D> | null

  /**
   * Running mean for inference
   */
  private runningMean: Tensor<readonly [number], D> | null

  /**
   * Running variance for inference
   */
  private runningVar: Tensor<readonly [number], D> | null

  /**
   * Create a new BatchNorm2d layer
   *
   * @param numFeatures - Number of features (channels)
   * @param options - Configuration options
   */
  constructor(
    numFeatures: number,
    options: {
      eps?: number
      momentum?: number
      affine?: boolean
      trackRunningStats?: boolean
    } = {},
  ) {
    super()

    this.numFeatures = numFeatures
    this.eps = options.eps ?? 1e-5
    this.momentum = options.momentum ?? 0.1
    this.affine = options.affine ?? true
    this.trackRunningStats = options.trackRunningStats ?? true

    validateNormParams({
      numFeatures,
      eps: this.eps,
      momentum: this.momentum,
    })

    const shape = [numFeatures] as const

    // Initialize learnable parameters
    if (this.affine) {
      // gamma (weight) initialized to 1
      const weightTensor = cpu.ones(shape) as unknown as Tensor<readonly [number], D>
      weightTensor.escape()
      this.weight = new Parameter(weightTensor, true)
      this.registerParameter('weight', this.weight)

      // beta (bias) initialized to 0
      const biasTensor = cpu.zeros(shape) as unknown as Tensor<readonly [number], D>
      biasTensor.escape()
      this.biasParam = new Parameter(biasTensor, true)
      this.registerParameter('bias', this.biasParam)
    } else {
      this.weight = null
      this.biasParam = null
    }

    // Initialize running statistics
    if (this.trackRunningStats) {
      const runningMeanTensor = cpu.zeros(shape) as unknown as Tensor<readonly [number], D>
      runningMeanTensor.escape()
      this.runningMean = runningMeanTensor

      const runningVarTensor = cpu.ones(shape) as unknown as Tensor<readonly [number], D>
      runningVarTensor.escape()
      this.runningVar = runningVarTensor
    } else {
      this.runningMean = null
      this.runningVar = null
    }
  }

  /**
   * Forward pass: applies batch normalization
   *
   * @param input - Input tensor with shape [N, C, H, W]
   * @returns Normalized tensor with same shape
   */
  forward(
    input: Tensor<readonly [number, number, number, number], D>,
  ): Tensor<readonly [number, number, number, number], D> {
    const result = (input as any).batchNorm(
      this.weight?.data ?? null,
      this.biasParam?.data ?? null,
      this.runningMean,
      this.runningVar,
      this._training,
      this.momentum,
      this.eps,
    )

    return result as Tensor<readonly [number, number, number, number], D>
  }

  override toString(): string {
    return `BatchNorm2d(${this.numFeatures}, eps=${this.eps}, momentum=${this.momentum}, affine=${this.affine})`
  }
}

/**
 * Layer normalization
 *
 * Applies Layer Normalization over a mini-batch of inputs as described in the paper
 * "Layer Normalization".
 *
 * y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
 *
 * Unlike Batch Normalization, Layer Normalization normalizes across the feature dimension
 * rather than the batch dimension, making it suitable for RNNs and Transformers.
 *
 * @example
 * ```ts
 * // Normalize over the last dimension
 * const ln = new LayerNorm([768]);
 *
 * // Normalize over multiple dimensions
 * const ln2 = new LayerNorm([32, 32]);
 *
 * // Forward pass
 * const input: Tensor<readonly [32, 128, 768]> = ...;
 * const output = ln.forward(input);
 * ```
 */
export class LayerNorm<D extends DType<string> = float32> extends Module<Shape, Shape, D> {
  readonly normalizedShape: readonly number[]
  readonly eps: number
  readonly elementwiseAffine: boolean

  /**
   * Learnable scale parameter (gamma)
   */
  readonly weight: Parameter<Shape, D> | null

  /**
   * Learnable shift parameter (beta)
   */
  readonly biasParam: Parameter<Shape, D> | null

  /**
   * Create a new LayerNorm layer
   *
   * @param normalizedShape - Shape over which to normalize (last N dimensions)
   * @param options - Configuration options
   */
  constructor(
    normalizedShape: number | readonly number[],
    options: {
      eps?: number
      elementwiseAffine?: boolean
    } = {},
  ) {
    super()

    this.normalizedShape =
      typeof normalizedShape === 'number' ? ([normalizedShape] as const) : normalizedShape
    this.eps = options.eps ?? 1e-5
    this.elementwiseAffine = options.elementwiseAffine ?? true

    validateNormParams({
      normalizedShape: this.normalizedShape,
      eps: this.eps,
    })

    // Initialize learnable parameters
    if (this.elementwiseAffine) {
      // gamma (weight) initialized to 1
      const weightTensor = cpu.ones(this.normalizedShape as number[]) as unknown as Tensor<Shape, D>
      weightTensor.escape()
      this.weight = new Parameter(weightTensor, true)
      this.registerParameter('weight', this.weight)

      // beta (bias) initialized to 0
      const biasTensor = cpu.zeros(this.normalizedShape as number[]) as unknown as Tensor<Shape, D>
      biasTensor.escape()
      this.biasParam = new Parameter(biasTensor, true)
      this.registerParameter('bias', this.biasParam)
    } else {
      this.weight = null
      this.biasParam = null
    }
  }

  /**
   * Forward pass: applies layer normalization
   *
   * @param input - Input tensor
   * @returns Normalized tensor with same shape
   */
  forward(input: Tensor<Shape, D>): Tensor<Shape, D> {
    const result = (input as any).layerNorm(
      this.normalizedShape,
      this.weight?.data ?? null,
      this.biasParam?.data ?? null,
      this.eps,
    )

    return result as Tensor<Shape, D>
  }

  override toString(): string {
    return `LayerNorm(${JSON.stringify(this.normalizedShape)}, eps=${this.eps}, elementwise_affine=${this.elementwiseAffine})`
  }
}

/**
 * 1D Batch normalization layer (for 3D tensor: N, C, L)
 *
 * Applies Batch Normalization over a 3D input.
 *
 * @example
 * ```ts
 * const bn = new BatchNorm1d(128);
 * const input: Tensor<readonly [32, 128, 50]> = ...;
 * const output = bn.forward(input);
 * ```
 */
export class BatchNorm1d<D extends DType<string> = float32> extends Module<
  readonly [number, number, number],
  readonly [number, number, number],
  D
> {
  readonly numFeatures: number
  readonly eps: number
  readonly momentum: number
  readonly affine: boolean
  readonly trackRunningStats: boolean

  readonly weight: Parameter<readonly [number], D> | null
  readonly biasParam: Parameter<readonly [number], D> | null
  private runningMean: Tensor<readonly [number], D> | null
  private runningVar: Tensor<readonly [number], D> | null

  constructor(
    numFeatures: number,
    options: {
      eps?: number
      momentum?: number
      affine?: boolean
      trackRunningStats?: boolean
    } = {},
  ) {
    super()

    this.numFeatures = numFeatures
    this.eps = options.eps ?? 1e-5
    this.momentum = options.momentum ?? 0.1
    this.affine = options.affine ?? true
    this.trackRunningStats = options.trackRunningStats ?? true

    validateNormParams({
      numFeatures,
      eps: this.eps,
      momentum: this.momentum,
    })

    const shape = [numFeatures] as const

    if (this.affine) {
      const weightTensor = cpu.ones(shape) as unknown as Tensor<readonly [number], D>
      weightTensor.escape()
      this.weight = new Parameter(weightTensor, true)
      this.registerParameter('weight', this.weight)

      const biasTensor = cpu.zeros(shape) as unknown as Tensor<readonly [number], D>
      biasTensor.escape()
      this.biasParam = new Parameter(biasTensor, true)
      this.registerParameter('bias', this.biasParam)
    } else {
      this.weight = null
      this.biasParam = null
    }

    if (this.trackRunningStats) {
      const runningMeanTensor = cpu.zeros(shape) as unknown as Tensor<readonly [number], D>
      runningMeanTensor.escape()
      this.runningMean = runningMeanTensor

      const runningVarTensor = cpu.ones(shape) as unknown as Tensor<readonly [number], D>
      runningVarTensor.escape()
      this.runningVar = runningVarTensor
    } else {
      this.runningMean = null
      this.runningVar = null
    }
  }

  forward(
    input: Tensor<readonly [number, number, number], D>,
  ): Tensor<readonly [number, number, number], D> {
    const result = (input as any).batchNorm(
      this.weight?.data ?? null,
      this.biasParam?.data ?? null,
      this.runningMean,
      this.runningVar,
      this._training,
      this.momentum,
      this.eps,
    )

    return result as Tensor<readonly [number, number, number], D>
  }

  override toString(): string {
    return `BatchNorm1d(${this.numFeatures}, eps=${this.eps}, momentum=${this.momentum}, affine=${this.affine})`
  }
}
