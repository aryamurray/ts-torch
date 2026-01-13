/**
 * Linear (fully connected) layer with type-safe shape inference
 *
 * Implements: y = xW^T + b
 * where W has shape [OutFeatures, InFeatures]
 *
 * @remarks
 * Parameters are initialized on CPU. Use `.to(device)` to move to target device.
 * The `nn.sequence()` builder handles this automatically.
 *
 * @template InFeatures - Number of input features
 * @template OutFeatures - Number of output features
 * @template D - Data type (default: float32)
 * @template Dev - Device type (default: 'cpu' - honest about initialization)
 */

import { Module, Parameter, type Tensor, type float32 } from '../module.js'
import { device, type DType, type DeviceType, type Shape, validateLinearParams } from '@ts-torch/core'

// CPU device for weight initialization
const cpu = device.cpu()

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
 * @template Dev - Device type (default: any device)
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
  Dev extends DeviceType = 'cpu',
> extends Module<readonly [number, InFeatures], readonly [number, OutFeatures], D, Dev> {
  /**
   * Weight matrix with shape [OutFeatures, InFeatures]
   * Transposed during forward pass for efficient computation
   */
  readonly weight: Parameter<readonly [OutFeatures, InFeatures], D, Dev>

  /**
   * Bias vector with shape [OutFeatures]
   * Null if bias is disabled
   */
  readonly bias: Parameter<readonly [OutFeatures], D, Dev> | null

  /**
   * Create a new Linear layer
   *
   * @param inFeatures - Number of input features
   * @param outFeatures - Number of output features
   * @param options - Configuration options
   *
   * @remarks
   * Parameters are initialized on CPU. When Dev is 'cpu' (the default), types are exact.
   * When using with `nn.sequence(device, [...])`, the builder calls `.to()` which
   * moves parameters and updates the type. Direct construction with Dev != 'cpu'
   * requires calling `.to()` for the type to be honest.
   */
  constructor(
    public readonly inFeatures: InFeatures,
    public readonly outFeatures: OutFeatures,
    options: LinearOptions<D> = {},
  ) {
    super()
    validateLinearParams(inFeatures, outFeatures)

    const { bias = true, init = 'kaiming_uniform' } = options

    // Initialize weight with specified strategy (always starts on CPU)
    // Cast is safe: Dev defaults to 'cpu', and .to() handles device movement
    this.weight = this.initWeight(init) as Parameter<readonly [OutFeatures, InFeatures], D, Dev>
    this.registerParameter('weight', this.weight)

    // Initialize bias if enabled (always starts on CPU)
    if (bias) {
      this.bias = this.initBias() as Parameter<readonly [OutFeatures], D, Dev>
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
  forward(input: Tensor<readonly [number, InFeatures], D, Dev>): Tensor<readonly [number, OutFeatures], D, Dev> {
    // Input: [Batch, InFeatures]
    // Weight: [OutFeatures, InFeatures]
    // Weight^T: [InFeatures, OutFeatures]
    // Result: [Batch, OutFeatures]

    // Transpose weight for matmul: [OutFeatures, InFeatures] -> [InFeatures, OutFeatures]
    const weightT = this.weight.data.transpose(0, 1)

    // Compute xW^T: [Batch, InFeatures] @ [InFeatures, OutFeatures] = [Batch, OutFeatures]
    let output = input.matmul(weightT) as Tensor<readonly [number, OutFeatures], D, Dev>

    // Add bias if present
    if (this.bias) {
      // Broadcasting: [Batch, OutFeatures] + [OutFeatures] = [Batch, OutFeatures]
      // The bias tensor has shape [OutFeatures] but same D and Dev types, so device safety is maintained
      output = output.add(this.bias.data as Tensor<Shape, D, Dev>) as Tensor<readonly [number, OutFeatures], D, Dev>
    }

    return output
  }

  /**
   * Initialize weight matrix using specified strategy
   *
   * @param init - Initialization strategy
   * @returns Initialized weight parameter (on CPU)
   * @internal
   *
   * @remarks
   * Weights are always initialized on CPU. The `.to(device)` method moves them
   * to the target device and updates the type accordingly.
   */
  private initWeight(
    init: 'kaiming_uniform' | 'kaiming_normal' | 'xavier_uniform' | 'xavier_normal' | 'zeros',
  ): Parameter<readonly [OutFeatures, InFeatures], D, 'cpu'> {
    const shape = [this.outFeatures, this.inFeatures] as const
    const fanIn = this.inFeatures

    // Note: cpu.randn/zeros return DType<'float32'>, cast to D is safe as runtime dtype matches
    // The 'cpu' device type is honest - these tensors are actually on CPU
    type WeightTensor = Tensor<readonly [OutFeatures, InFeatures], D, 'cpu'>
    let weight: WeightTensor

    switch (init) {
      case 'kaiming_uniform':
      case 'kaiming_normal': {
        // Kaiming/He initialization for ReLU activations
        // std = sqrt(2 / fan_in) for ReLU
        const std = Math.sqrt(2.0 / fanIn)
        const randWeight = cpu.randn(shape)
        weight = (randWeight as any).mulScalar(std) as WeightTensor
        break
      }

      case 'xavier_uniform':
      case 'xavier_normal': {
        // Xavier/Glorot initialization for tanh/sigmoid
        // std = sqrt(2 / (fan_in + fan_out))
        const std = Math.sqrt(2.0 / (fanIn + this.outFeatures))
        const randWeight = cpu.randn(shape)
        weight = (randWeight as any).mulScalar(std) as WeightTensor
        break
      }

      case 'zeros': {
        // Zero initialization (mainly for testing)
        weight = cpu.zeros(shape) as unknown as WeightTensor
        break
      }

      default: {
        // Default to Kaiming normal
        const std = Math.sqrt(2.0 / fanIn)
        const randWeight = cpu.randn(shape)
        weight = (randWeight as any).mulScalar(std) as WeightTensor
      }
    }

    // Escape weight from any scope so it persists
    weight.escape()

    return new Parameter(weight, true)
  }

  /**
   * Initialize bias vector to zeros
   *
   * @returns Initialized bias parameter (on CPU)
   * @internal
   *
   * @remarks
   * Bias is always initialized on CPU. The `.to(device)` method moves it
   * to the target device and updates the type accordingly.
   */
  private initBias(): Parameter<readonly [OutFeatures], D, 'cpu'> {
    const shape = [this.outFeatures] as const
    // Note: cpu.zeros returns DType<'float32'>, cast to D is safe as runtime dtype matches
    // The 'cpu' device type is honest - this tensor is actually on CPU
    type BiasTensor = Tensor<readonly [OutFeatures], D, 'cpu'>
    const bias = cpu.zeros(shape) as unknown as BiasTensor

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
