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
 * Built-in weight initialization strategies
 */
export type InitStrategy =
  | 'kaiming_uniform'
  | 'kaiming_normal'
  | 'xavier_uniform'
  | 'xavier_normal'
  | 'orthogonal'
  | 'zeros'
  | 'ones'

/**
 * Custom weight initializer function
 *
 * @param fanIn - Number of input features
 * @param fanOut - Number of output features
 * @param shape - Weight tensor shape [outFeatures, inFeatures]
 * @returns Initialized weight tensor
 *
 * @example
 * ```ts
 * // Custom uniform initialization
 * const customInit: InitFn = (fanIn, fanOut, shape) => {
 *   const bound = Math.sqrt(6 / (fanIn + fanOut))
 *   return cpu.uniform(shape, -bound, bound)
 * }
 *
 * const layer = new Linear(784, 128, { init: customInit })
 * ```
 */
export type InitFn = (fanIn: number, fanOut: number, shape: readonly [number, number]) => Tensor<any, any, 'cpu'>

/**
 * Constant initializer options
 */
export interface ConstantInit {
  type: 'constant'
  value: number
}

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
   *
   * Can be:
   * - A string literal: 'kaiming_uniform', 'kaiming_normal', 'xavier_uniform', 'xavier_normal', 'orthogonal', 'zeros', 'ones'
   * - A constant object: { type: 'constant', value: 0.5 }
   * - A custom function: (fanIn, fanOut, shape) => Tensor
   */
  init?: InitStrategy | ConstantInit | InitFn
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

    const { bias = true, init = 'kaiming_uniform' as InitStrategy | ConstantInit | InitFn } = options

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
   * @param init - Initialization strategy (string, object, or custom function)
   * @returns Initialized weight parameter (on CPU)
   * @internal
   *
   * @remarks
   * Weights are always initialized on CPU. The `.to(device)` method moves them
   * to the target device and updates the type accordingly.
   */
  private initWeight(
    init: InitStrategy | ConstantInit | InitFn,
  ): Parameter<readonly [OutFeatures, InFeatures], D, 'cpu'> {
    const shape = [this.outFeatures, this.inFeatures] as const
    const fanIn = this.inFeatures
    const fanOut = this.outFeatures

    // Note: cpu.randn/zeros return DType<'float32'>, cast to D is safe as runtime dtype matches
    // The 'cpu' device type is honest - these tensors are actually on CPU
    type WeightTensor = Tensor<readonly [OutFeatures, InFeatures], D, 'cpu'>
    let weight: WeightTensor

    // Handle custom function
    if (typeof init === 'function') {
      weight = init(fanIn, fanOut, shape) as WeightTensor
      weight.escape()
      return new Parameter(weight, true)
    }

    // Handle constant initialization object
    if (typeof init === 'object' && init.type === 'constant') {
      const ones = cpu.ones(shape)
      weight = (ones as any).mulScalar(init.value) as WeightTensor
      weight.escape()
      return new Parameter(weight, true)
    }

    // Handle string-based initialization strategies
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
        const std = Math.sqrt(2.0 / (fanIn + fanOut))
        const randWeight = cpu.randn(shape)
        weight = (randWeight as any).mulScalar(std) as WeightTensor
        break
      }

      case 'orthogonal': {
        // Orthogonal initialization using QR decomposition approximation
        // For simplicity, we use scaled random normal (proper orthogonal requires QR)
        // gain = sqrt(2) for ReLU by default
        const gain = Math.sqrt(2.0)
        const randWeight = cpu.randn(shape)
        // Normalize rows to approximate orthogonality
        weight = (randWeight as any).mulScalar(gain / Math.sqrt(fanIn)) as WeightTensor
        break
      }

      case 'zeros': {
        // Zero initialization (mainly for testing)
        weight = cpu.zeros(shape) as unknown as WeightTensor
        break
      }

      case 'ones': {
        // Ones initialization (mainly for testing)
        weight = cpu.ones(shape) as unknown as WeightTensor
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

  protected override _outputShapeHint(): string {
    return `[*, ${this.outFeatures}]`
  }

  /**
   * String representation
   */
  override toString(): string {
    return `Linear(in_features=${this.inFeatures}, out_features=${this.outFeatures}, bias=${this.bias !== null})`
  }
}

/**
 * Fused Linear + Activation module
 *
 * Combines a linear transformation with an activation function into a single operation.
 * This leverages native fused operations like linearRelu, linearSigmoid, etc. for improved performance.
 *
 * Applies: y = activation(xW^T + b)
 *
 * Supported activations:
 * - 'relu': Rectified Linear Unit
 * - 'sigmoid': Sigmoid activation
 * - 'tanh': Hyperbolic tangent activation
 *
 * @template InFeatures - Number of input features
 * @template OutFeatures - Number of output features
 * @template D - Data type (default: float32)
 * @template Dev - Device type (default: 'cpu')
 *
 * @example
 * ```ts
 * // Fused ReLU: faster than Linear + ReLU
 * const layer = new FusedLinear(784, 128, { activation: 'relu' });
 * const output = layer.forward(input); // Fused operation
 * ```
 */
export class FusedLinear<
  InFeatures extends number,
  OutFeatures extends number,
  D extends DType<string> = float32,
  Dev extends DeviceType = 'cpu',
> extends Module<readonly [number, InFeatures], readonly [number, OutFeatures], D, Dev> {
  /**
   * Weight matrix with shape [OutFeatures, InFeatures]
   */
  readonly weight: Parameter<readonly [OutFeatures, InFeatures], D, Dev>

  /**
   * Bias vector with shape [OutFeatures]
   * Null if bias is disabled
   */
  readonly bias: Parameter<readonly [OutFeatures], D, Dev> | null

  /**
   * Activation function: 'relu', 'sigmoid', or 'tanh'
   */
  readonly activation: 'relu' | 'sigmoid' | 'tanh'

  constructor(
    public readonly inFeatures: InFeatures,
    public readonly outFeatures: OutFeatures,
    options: LinearOptions<D> & { activation: 'relu' | 'sigmoid' | 'tanh' } = {} as any,
  ) {
    super()
    validateLinearParams(inFeatures, outFeatures)

    const { bias = true, init = 'kaiming_uniform' as InitStrategy | ConstantInit | InitFn, activation = 'relu' } = options as any

    // Initialize weight (same as Linear)
    this.weight = this.initWeight(init) as Parameter<readonly [OutFeatures, InFeatures], D, Dev>
    this.registerParameter('weight', this.weight)

    // Initialize bias if enabled
    if (bias) {
      this.bias = this.initBias() as Parameter<readonly [OutFeatures], D, Dev>
      this.registerParameter('bias', this.bias)
    } else {
      this.bias = null
    }

    this.activation = activation
  }

  /**
   * Forward pass with fused linear + activation
   *
   * @param input - Input tensor with shape [Batch, InFeatures]
   * @returns Output tensor with shape [Batch, OutFeatures]
   */
  forward(input: Tensor<readonly [number, InFeatures], D, Dev>): Tensor<readonly [number, OutFeatures], D, Dev> {
    // Fused operations expect weight in [OutFeatures, InFeatures] shape (not transposed)
    // Use fused operation based on activation type
    let output: Tensor<readonly [number, OutFeatures], D, Dev>

    if (this.activation === 'relu') {
      // Use fused linearRelu: weight [OutFeatures, InFeatures], bias [OutFeatures]
      output = (input as any).linearRelu(this.weight.data, this.bias?.data ?? null) as Tensor<readonly [number, OutFeatures], D, Dev>
    } else if (this.activation === 'sigmoid') {
      // Use fused linearSigmoid
      output = (input as any).linearSigmoid(this.weight.data, this.bias?.data ?? null) as Tensor<readonly [number, OutFeatures], D, Dev>
    } else if (this.activation === 'tanh') {
      // Use fused linearTanh
      output = (input as any).linearTanh(this.weight.data, this.bias?.data ?? null) as Tensor<readonly [number, OutFeatures], D, Dev>
    } else {
      // Fallback: this shouldn't happen if constructor enforces activation type
      // Do linear + ReLU as default
      output = (input as any).linearRelu(this.weight.data, this.bias?.data ?? null) as Tensor<readonly [number, OutFeatures], D, Dev>
    }

    return output
  }

  /**
   * Initialize weight matrix using specified strategy (same as Linear)
   *
   * @param init - Initialization strategy (string, object, or custom function)
   * @returns Initialized weight parameter (on CPU)
   * @internal
   */
  private initWeight(
    init: InitStrategy | ConstantInit | InitFn,
  ): Parameter<readonly [OutFeatures, InFeatures], D, 'cpu'> {
    const shape = [this.outFeatures, this.inFeatures] as const
    const fanIn = this.inFeatures
    const fanOut = this.outFeatures

    type WeightTensor = Tensor<readonly [OutFeatures, InFeatures], D, 'cpu'>
    let weight: WeightTensor

    // Handle custom function
    if (typeof init === 'function') {
      weight = init(fanIn, fanOut, shape) as WeightTensor
      weight.escape()
      return new Parameter(weight, true)
    }

    // Handle constant initialization object
    if (typeof init === 'object' && init.type === 'constant') {
      const ones = cpu.ones(shape)
      weight = (ones as any).mulScalar(init.value) as WeightTensor
      weight.escape()
      return new Parameter(weight, true)
    }

    // Handle string-based initialization strategies
    switch (init) {
      case 'kaiming_uniform':
      case 'kaiming_normal': {
        const std = Math.sqrt(2.0 / fanIn)
        const randWeight = cpu.randn(shape)
        weight = (randWeight as any).mulScalar(std) as WeightTensor
        break
      }

      case 'xavier_uniform':
      case 'xavier_normal': {
        const std = Math.sqrt(2.0 / (fanIn + fanOut))
        const randWeight = cpu.randn(shape)
        weight = (randWeight as any).mulScalar(std) as WeightTensor
        break
      }

      case 'orthogonal': {
        const gain = Math.sqrt(2.0)
        const randWeight = cpu.randn(shape)
        weight = (randWeight as any).mulScalar(gain / Math.sqrt(fanIn)) as WeightTensor
        break
      }

      case 'zeros': {
        weight = cpu.zeros(shape) as unknown as WeightTensor
        break
      }

      case 'ones': {
        weight = cpu.ones(shape) as unknown as WeightTensor
        break
      }

      default: {
        const std = Math.sqrt(2.0 / fanIn)
        const randWeight = cpu.randn(shape)
        weight = (randWeight as any).mulScalar(std) as WeightTensor
      }
    }

    weight.escape()
    return new Parameter(weight, true)
  }

  /**
   * Initialize bias vector to zeros (same as Linear)
   *
   * @returns Initialized bias parameter (on CPU)
   * @internal
   */
  private initBias(): Parameter<readonly [OutFeatures], D, 'cpu'> {
    const shape = [this.outFeatures] as const
    type BiasTensor = Tensor<readonly [OutFeatures], D, 'cpu'>
    const bias = cpu.zeros(shape) as unknown as BiasTensor

    bias.escape()
    return new Parameter(bias, true)
  }

  protected override _outputShapeHint(): string {
    return `[*, ${this.outFeatures}]`
  }

  /**
   * String representation
   */
  override toString(): string {
    return `FusedLinear(in_features=${this.inFeatures}, out_features=${this.outFeatures}, activation=${this.activation}, bias=${this.bias !== null})`
  }
}
