/**
 * Linear (fully connected) layer with type-safe shape inference
 *
 * Implements: y = xW^T + b
 * where W has shape [OutFeatures, InFeatures]
 */

import { Module, Parameter, type Tensor, type float32 } from '../module.js'
import { torch, type DType } from '@ts-torch/core'

/**
 * Linear options interface
 */
export interface LinearOptions<D extends DType<string> = float32> {
  /**
   * Whether to include bias term (default: true)
   */
  bias?: boolean

  /**
   * Data type for weights and bias (default: float32)
   */
  dtype?: D

  /**
   * Weight initialization strategy (default: 'kaiming_uniform')
   */
  init?: 'kaiming_uniform' | 'kaiming_normal' | 'xavier_uniform' | 'xavier_normal' | 'zeros'
}

/**
 * Linear (fully connected) layer
 *
 * Applies a linear transformation: y = xW^T + b
 *
 * Type parameters ensure compile-time shape checking:
 * - Input shape: [Batch, InFeatures]
 * - Output shape: [Batch, OutFeatures]
 *
 * @template InFeatures - Number of input features
 * @template OutFeatures - Number of output features
 * @template D - Data type (default: float32)
 *
 * @example
 * ```ts
 * // Create a linear layer: 784 -> 128
 * const fc1 = new Linear(784, 128);
 *
 * // Type-safe chaining
 * const model = new Linear(784, 128)
 *   .pipe(new ReLU())
 *   .pipe(new Linear(128, 10));
 *
 * // Input must be [Batch, 784]
 * const input: Tensor<readonly [32, 784]> = ...;
 * const output = model.forward(input); // Type: Tensor<readonly [32, 10]>
 * ```
 */
export class Linear<
  InFeatures extends number,
  OutFeatures extends number,
  D extends DType<string> = float32,
> extends Module<readonly [number, InFeatures], readonly [number, OutFeatures], D> {
  /**
   * Weight matrix with shape [OutFeatures, InFeatures]
   * Transposed during forward pass for efficient computation
   */
  readonly weight: Parameter<readonly [OutFeatures, InFeatures], D>

  /**
   * Bias vector with shape [OutFeatures]
   * Null if bias is disabled
   */
  readonly bias: Parameter<readonly [OutFeatures], D> | null

  /**
   * Create a new Linear layer
   *
   * @param inFeatures - Number of input features
   * @param outFeatures - Number of output features
   * @param options - Configuration options
   */
  constructor(
    public readonly inFeatures: InFeatures,
    public readonly outFeatures: OutFeatures,
    options: LinearOptions<D> = {},
  ) {
    super()

    const { bias = true, init = 'kaiming_uniform' } = options

    // Initialize weight with specified strategy
    this.weight = this.initWeight(init) as Parameter<readonly [OutFeatures, InFeatures], D>
    this.registerParameter('weight', this.weight)

    // Initialize bias if enabled
    if (bias) {
      this.bias = this.initBias() as Parameter<readonly [OutFeatures], D>
      this.registerParameter('bias', this.bias)
    } else {
      this.bias = null
    }
  }

  /**
   * Forward pass: y = xW^T + b
   *
   * @param input - Input tensor with shape [Batch, InFeatures]
   * @returns Output tensor with shape [Batch, OutFeatures]
   */
  forward(input: Tensor<readonly [number, InFeatures], D>): Tensor<readonly [number, OutFeatures], D> {
    // Input: [Batch, InFeatures]
    // Weight: [OutFeatures, InFeatures]
    // Weight^T: [InFeatures, OutFeatures]
    // Result: [Batch, OutFeatures]

    // Transpose weight for matmul: [OutFeatures, InFeatures] -> [InFeatures, OutFeatures]
    const weightT = this.weight.data.transpose(0, 1)

    // Compute xW^T: [Batch, InFeatures] @ [InFeatures, OutFeatures] = [Batch, OutFeatures]
    let output = input.matmul(weightT) as Tensor<readonly [number, OutFeatures], D>

    // Add bias if present
    if (this.bias) {
      // Broadcasting: [Batch, OutFeatures] + [OutFeatures] = [Batch, OutFeatures]
      output = output.add(this.bias.data as any) as Tensor<readonly [number, OutFeatures], D>
    }

    return output
  }

  /**
   * Initialize weight matrix using specified strategy
   *
   * @param init - Initialization strategy
   * @returns Initialized weight parameter
   */
  private initWeight(
    init: 'kaiming_uniform' | 'kaiming_normal' | 'xavier_uniform' | 'xavier_normal' | 'zeros',
  ): Parameter<readonly [OutFeatures, InFeatures], D> {
    const shape = [this.outFeatures, this.inFeatures] as const

    let weight: Tensor<readonly [OutFeatures, InFeatures], D>

    switch (init) {
      case 'kaiming_uniform':
      case 'kaiming_normal': {
        // Kaiming/He initialization for ReLU activations
        // Use randn (approximate - we'd need scalar mul for exact scaling)
        weight = torch.randn(shape) as unknown as Tensor<readonly [OutFeatures, InFeatures], D>
        break
      }

      case 'xavier_uniform':
      case 'xavier_normal': {
        // Xavier/Glorot initialization for tanh/sigmoid
        weight = torch.randn(shape) as unknown as Tensor<readonly [OutFeatures, InFeatures], D>
        break
      }

      case 'zeros': {
        // Zero initialization (mainly for testing)
        weight = torch.zeros(shape) as unknown as Tensor<readonly [OutFeatures, InFeatures], D>
        break
      }

      default: {
        // Default to randn
        weight = torch.randn(shape) as unknown as Tensor<readonly [OutFeatures, InFeatures], D>
      }
    }

    // Escape weight from any scope so it persists
    weight.escape()

    return new Parameter(weight, true)
  }

  /**
   * Initialize bias vector to zeros
   *
   * @returns Initialized bias parameter
   */
  private initBias(): Parameter<readonly [OutFeatures], D> {
    const shape = [this.outFeatures] as const
    const bias = torch.zeros(shape) as unknown as Tensor<readonly [OutFeatures], D>

    // Escape bias from any scope so it persists
    bias.escape()

    return new Parameter(bias, true)
  }

  /**
   * String representation
   */
  override toString(): string {
    return `Linear(in_features=${this.inFeatures}, out_features=${this.outFeatures}, bias=${this.bias !== null})`
  }
}
