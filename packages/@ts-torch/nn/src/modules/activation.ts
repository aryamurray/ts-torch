/**
 * Activation function modules with type-safe shape preservation
 *
 * Activation functions maintain input shape, making them easy to compose.
 */

import { Module, Parameter, type Tensor, type float32 } from '../module.js'
import { device, type Shape, type DType, type DeviceType } from '@ts-torch/core'

/**
 * Rectified Linear Unit activation: ReLU(x) = max(0, x)
 *
 * Shape-preserving: input and output have identical shapes.
 *
 * @template S - Tensor shape (preserved through activation)
 * @template D - Data type (default: float32)
 * @template Dev - Device type (default: any device)
 *
 * @example
 * ```ts
 * const relu = new ReLU<readonly [32, 128]>();
 * const input: Tensor<readonly [32, 128]> = ...;
 * const output = relu.forward(input); // Type: Tensor<readonly [32, 128]>
 *
 * // Use in pipeline
 * const model = new Linear(784, 128)
 *   .pipe(new ReLU())
 *   .pipe(new Linear(128, 10));
 * ```
 */
export class ReLU<S extends Shape = Shape, D extends DType<string> = float32, Dev extends DeviceType = DeviceType> extends Module<S, S, D, Dev> {
  /**
   * Create a new ReLU activation
   *
   * @param inplace - Whether to modify input tensor in-place (default: false)
   */
  constructor(public readonly inplace: boolean = false) {
    super()
  }

  /**
   * Forward pass: ReLU(x) = max(0, x)
   *
   * @param input - Input tensor
   * @returns Output tensor with same shape, all negative values set to 0
   */
  forward(input: Tensor<S, D, Dev>): Tensor<S, D, Dev> {
    return input.relu()
  }

  override toString(): string {
    return `ReLU(inplace=${this.inplace})`
  }
}

/**
 * Sigmoid activation: σ(x) = 1 / (1 + e^(-x))
 *
 * Shape-preserving: input and output have identical shapes.
 * Output values are in range (0, 1).
 *
 * @template S - Tensor shape (preserved through activation)
 * @template D - Data type (default: float32)
 * @template Dev - Device type (default: any device)
 *
 * @example
 * ```ts
 * const sigmoid = new Sigmoid<readonly [32, 10]>();
 * const logits: Tensor<readonly [32, 10]> = ...;
 * const probs = sigmoid.forward(logits); // Type: Tensor<readonly [32, 10]>
 * ```
 */
export class Sigmoid<S extends Shape = Shape, D extends DType<string> = float32, Dev extends DeviceType = DeviceType> extends Module<S, S, D, Dev> {
  /**
   * Forward pass: σ(x) = 1 / (1 + e^(-x))
   *
   * @param input - Input tensor
   * @returns Output tensor with same shape, values in (0, 1)
   */
  forward(input: Tensor<S, D, Dev>): Tensor<S, D, Dev> {
    return input.sigmoid()
  }

  override toString(): string {
    return 'Sigmoid()'
  }
}

/**
 * Hyperbolic tangent activation: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
 *
 * Shape-preserving: input and output have identical shapes.
 * Output values are in range (-1, 1).
 *
 * @template S - Tensor shape (preserved through activation)
 * @template D - Data type (default: float32)
 * @template Dev - Device type (default: any device)
 *
 * @example
 * ```ts
 * const tanh = new Tanh<readonly [32, 64]>();
 * const input: Tensor<readonly [32, 64]> = ...;
 * const output = tanh.forward(input); // Type: Tensor<readonly [32, 64]>
 * ```
 */
export class Tanh<S extends Shape = Shape, D extends DType<string> = float32, Dev extends DeviceType = DeviceType> extends Module<S, S, D, Dev> {
  /**
   * Forward pass: tanh(x)
   *
   * @param input - Input tensor
   * @returns Output tensor with same shape, values in (-1, 1)
   */
  forward(input: Tensor<S, D, Dev>): Tensor<S, D, Dev> {
    return input.tanh()
  }

  override toString(): string {
    return 'Tanh()'
  }
}

/**
 * Softmax activation: softmax(x_i) = e^(x_i) / Σ(e^(x_j))
 *
 * Shape-preserving: input and output have identical shapes.
 * Output values sum to 1 along the specified dimension.
 *
 * @template S - Tensor shape (preserved through activation)
 * @template D - Data type (default: float32)
 * @template Dev - Device type (default: any device)
 *
 * @example
 * ```ts
 * // Apply softmax along last dimension (class probabilities)
 * const softmax = new Softmax<readonly [32, 10]>(-1);
 * const logits: Tensor<readonly [32, 10]> = ...;
 * const probs = softmax.forward(logits); // Type: Tensor<readonly [32, 10]>
 * // Each row sums to 1
 * ```
 */
export class Softmax<S extends Shape = Shape, D extends DType<string> = float32, Dev extends DeviceType = DeviceType> extends Module<S, S, D, Dev> {
  /**
   * Create a new Softmax activation
   *
   * @param dim - Dimension along which to apply softmax (default: -1, last dimension)
   */
  constructor(public readonly dim: number = -1) {
    super()
  }

  /**
   * Forward pass: softmax(x) along specified dimension
   *
   * @param input - Input tensor
   * @returns Output tensor with same shape, values sum to 1 along dim
   */
  forward(input: Tensor<S, D, Dev>): Tensor<S, D, Dev> {
    return input.softmax(this.dim)
  }

  override toString(): string {
    return `Softmax(dim=${this.dim})`
  }
}

/**
 * Leaky ReLU activation: LeakyReLU(x) = max(αx, x)
 *
 * Shape-preserving: input and output have identical shapes.
 * Allows small negative values instead of zeroing them.
 *
 * @template S - Tensor shape (preserved through activation)
 * @template D - Data type (default: float32)
 * @template Dev - Device type (default: any device)
 *
 * @example
 * ```ts
 * const leaky = new LeakyReLU<readonly [32, 128]>(0.01);
 * const input: Tensor<readonly [32, 128]> = ...;
 * const output = leaky.forward(input); // Type: Tensor<readonly [32, 128]>
 * ```
 */
export class LeakyReLU<S extends Shape = Shape, D extends DType<string> = float32, Dev extends DeviceType = DeviceType> extends Module<S, S, D, Dev> {
  /**
   * Create a new Leaky ReLU activation
   *
   * @param negativeSlope - Slope for negative values (default: 0.01)
   * @param inplace - Whether to modify input tensor in-place (default: false)
   */
  constructor(
    public readonly negativeSlope: number = 0.01,
    public readonly inplace: boolean = false,
  ) {
    super()
  }

  /**
   * Forward pass: LeakyReLU(x) = max(αx, x)
   *
   * @param input - Input tensor
   * @returns Output tensor with same shape
   */
  forward(input: Tensor<S, D, Dev>): Tensor<S, D, Dev> {
    const negative = input.mulScalar(this.negativeSlope)
    return input.maximum(negative)
  }

  override toString(): string {
    return `LeakyReLU(negative_slope=${this.negativeSlope}, inplace=${this.inplace})`
  }
}

/**
 * GELU (Gaussian Error Linear Unit) activation
 *
 * GELU(x) = x * Φ(x), where Φ(x) is the CDF of standard normal distribution.
 * Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))
 *
 * Shape-preserving: input and output have identical shapes.
 * Popular in transformer architectures.
 *
 * @template S - Tensor shape (preserved through activation)
 * @template D - Data type (default: float32)
 * @template Dev - Device type (default: any device)
 *
 * @example
 * ```ts
 * const gelu = new GELU<readonly [32, 768]>();
 * const input: Tensor<readonly [32, 768]> = ...;
 * const output = gelu.forward(input); // Type: Tensor<readonly [32, 768]>
 * ```
 */
export class GELU<S extends Shape = Shape, D extends DType<string> = float32, Dev extends DeviceType = DeviceType> extends Module<S, S, D, Dev> {
  /**
   * Forward pass: GELU(x)
   *
   * @param input - Input tensor
   * @returns Output tensor with same shape
   */
  forward(input: Tensor<S, D, Dev>): Tensor<S, D, Dev> {
    const x3 = input.mul(input).mul(input)
    const inner = input.add(x3.mulScalar(0.044715)).mulScalar(Math.sqrt(2 / Math.PI))
    const tanh = inner.tanh()
    return input.mul(tanh.addScalar(1)).mulScalar(0.5)
  }

  override toString(): string {
    return 'GELU()'
  }
}

/**
 * ELU (Exponential Linear Unit) activation
 *
 * ELU(x) = x if x > 0
 *        = alpha * (exp(x) - 1) if x <= 0
 *
 * Helps with the vanishing gradient problem while allowing negative outputs.
 *
 * @template S - Tensor shape (preserved through activation)
 * @template D - Data type (default: float32)
 * @template Dev - Device type (default: any device)
 */
export class ELU<S extends Shape = Shape, D extends DType<string> = float32, Dev extends DeviceType = DeviceType> extends Module<S, S, D, Dev> {
  /**
   * Create a new ELU activation
   *
   * @param alpha - Scale for negative outputs (default: 1.0)
   * @param inplace - Whether to modify input in-place (default: false)
   */
  constructor(
    public readonly alpha: number = 1.0,
    public readonly inplace: boolean = false,
  ) {
    super()
  }

  forward(input: Tensor<S, D, Dev>): Tensor<S, D, Dev> {
    // ELU(x) = x if x > 0, alpha * (exp(x) - 1) otherwise
    // Approximate using: max(0, x) + min(0, alpha * (exp(x) - 1))
    const positive = input.relu()
    const expX = input.exp()
    const negative = expX.addScalar(-1).mulScalar(this.alpha).minimum(input.mulScalar(0))
    return positive.add(negative)
  }

  override toString(): string {
    return `ELU(alpha=${this.alpha})`
  }
}

/**
 * SELU (Scaled Exponential Linear Unit) activation
 *
 * Self-normalizing activation that maintains mean 0 and variance 1.
 * SELU(x) = scale * (max(0, x) + min(0, alpha * (exp(x) - 1)))
 *
 * where alpha ≈ 1.6733 and scale ≈ 1.0507
 */
export class SELU<S extends Shape = Shape, D extends DType<string> = float32, Dev extends DeviceType = DeviceType> extends Module<S, S, D, Dev> {
  private readonly alpha = 1.673263242354377
  private readonly scale = 1.050700987355480

  constructor(public readonly inplace: boolean = false) {
    super()
  }

  forward(input: Tensor<S, D, Dev>): Tensor<S, D, Dev> {
    const positive = input.relu()
    const expX = input.exp()
    const negative = expX.addScalar(-1).mulScalar(this.alpha).minimum(input.mulScalar(0))
    return positive.add(negative).mulScalar(this.scale)
  }

  override toString(): string {
    return 'SELU()'
  }
}

/**
 * SiLU (Swish) activation: x * sigmoid(x)
 *
 * Smooth, non-monotonic activation used in EfficientNet and modern architectures.
 * Also known as Swish activation.
 */
export class SiLU<S extends Shape = Shape, D extends DType<string> = float32, Dev extends DeviceType = DeviceType> extends Module<S, S, D, Dev> {
  constructor(public readonly inplace: boolean = false) {
    super()
  }

  forward(input: Tensor<S, D, Dev>): Tensor<S, D, Dev> {
    return input.mul(input.sigmoid())
  }

  override toString(): string {
    return 'SiLU()'
  }
}

/**
 * Alias for SiLU activation (same as Swish)
 */
export const Swish = SiLU

/**
 * Mish activation: x * tanh(softplus(x))
 *
 * Smooth, self-regularizing activation function.
 * softplus(x) = log(1 + exp(x))
 */
export class Mish<S extends Shape = Shape, D extends DType<string> = float32, Dev extends DeviceType = DeviceType> extends Module<S, S, D, Dev> {
  constructor(public readonly inplace: boolean = false) {
    super()
  }

  forward(input: Tensor<S, D, Dev>): Tensor<S, D, Dev> {
    // softplus(x) = log(1 + exp(x))
    const softplus = input.exp().addScalar(1).log()
    return input.mul(softplus.tanh())
  }

  override toString(): string {
    return 'Mish()'
  }
}

/**
 * Hardswish activation: x * hardSigmoid(x)
 *
 * Mobile-optimized version of Swish, used in MobileNetV3.
 * hardswish(x) = x * ReLU6(x + 3) / 6
 */
export class Hardswish<S extends Shape = Shape, D extends DType<string> = float32, Dev extends DeviceType = DeviceType> extends Module<S, S, D, Dev> {
  constructor(public readonly inplace: boolean = false) {
    super()
  }

  forward(input: Tensor<S, D, Dev>): Tensor<S, D, Dev> {
    // hardswish(x) = x * min(max(x + 3, 0), 6) / 6
    const relu6 = input.addScalar(3).clamp(0, 6)
    return input.mul(relu6).divScalar(6)
  }

  override toString(): string {
    return 'Hardswish()'
  }
}

/**
 * Hardsigmoid activation
 *
 * Mobile-optimized approximation of sigmoid.
 * hardsigmoid(x) = ReLU6(x + 3) / 6
 */
export class Hardsigmoid<S extends Shape = Shape, D extends DType<string> = float32, Dev extends DeviceType = DeviceType> extends Module<S, S, D, Dev> {
  constructor(public readonly inplace: boolean = false) {
    super()
  }

  forward(input: Tensor<S, D, Dev>): Tensor<S, D, Dev> {
    return input.addScalar(3).clamp(0, 6).divScalar(6)
  }

  override toString(): string {
    return 'Hardsigmoid()'
  }
}

/**
 * Hardtanh activation
 *
 * Piecewise linear approximation of tanh.
 * hardtanh(x) = max(min_val, min(max_val, x))
 */
export class Hardtanh<S extends Shape = Shape, D extends DType<string> = float32, Dev extends DeviceType = DeviceType> extends Module<S, S, D, Dev> {
  constructor(
    public readonly minVal: number = -1.0,
    public readonly maxVal: number = 1.0,
    public readonly inplace: boolean = false,
  ) {
    super()
  }

  forward(input: Tensor<S, D, Dev>): Tensor<S, D, Dev> {
    return input.clamp(this.minVal, this.maxVal)
  }

  override toString(): string {
    return `Hardtanh(min_val=${this.minVal}, max_val=${this.maxVal})`
  }
}

/**
 * ReLU6 activation: min(max(0, x), 6)
 *
 * Commonly used in mobile architectures.
 */
export class ReLU6<S extends Shape = Shape, D extends DType<string> = float32, Dev extends DeviceType = DeviceType> extends Module<S, S, D, Dev> {
  constructor(public readonly inplace: boolean = false) {
    super()
  }

  forward(input: Tensor<S, D, Dev>): Tensor<S, D, Dev> {
    return input.clamp(0, 6)
  }

  override toString(): string {
    return 'ReLU6()'
  }
}

/**
 * PReLU (Parametric Rectified Linear Unit)
 *
 * PReLU(x) = max(0, x) + a * min(0, x)
 *
 * where a is a learnable parameter.
 */
export class PReLU<S extends Shape = Shape, D extends DType<string> = float32, Dev extends DeviceType = DeviceType> extends Module<S, S, D, Dev> {
  /**
   * Learnable slope parameter
   */
  readonly weight: import('../module.js').Parameter<readonly [number], D, Dev>

  /**
   * Create PReLU activation
   *
   * @param numParameters - Number of learnable parameters (1 for shared, or number of channels)
   * @param init - Initial value for negative slope (default: 0.25)
   */
  constructor(
    public readonly numParameters: number = 1,
    init: number = 0.25,
  ) {
    super()

    const cpu = device.cpu()

    // Initialize weight parameter
    const weightTensor = cpu.ones([numParameters]).mulScalar(init)
    weightTensor.escape()
    this.weight = new Parameter(weightTensor, true)
    this.registerParameter('weight', this.weight as any)
  }

  forward(input: Tensor<S, D, Dev>): Tensor<S, D, Dev> {
    // PReLU(x) = max(0, x) + weight * min(0, x)
    const positive = input.relu()
    const negative = input.minimum(input.mulScalar(0))

    // Broadcast weight appropriately
    // For single parameter, broadcast across all elements
    // For per-channel, need to reshape weight
    let scaledNegative: Tensor<S, D, Dev>
    if (this.numParameters === 1) {
      const weightValue = (this.weight.data as any).toArray()[0] as number
      scaledNegative = negative.mulScalar(weightValue)
    } else {
      // Per-channel PReLU - need to broadcast weight across spatial dimensions
      scaledNegative = negative.mul(this.weight.data as any)
    }

    return positive.add(scaledNegative)
  }

  override toString(): string {
    return `PReLU(num_parameters=${this.numParameters})`
  }
}

/**
 * Softplus activation: log(1 + exp(x))
 *
 * Smooth approximation to ReLU.
 */
export class Softplus<S extends Shape = Shape, D extends DType<string> = float32, Dev extends DeviceType = DeviceType> extends Module<S, S, D, Dev> {
  /**
   * @param beta - Scaling factor (default: 1)
   * @param threshold - Above this, switch to linear (default: 20)
   */
  constructor(
    public readonly beta: number = 1,
    public readonly threshold: number = 20,
  ) {
    super()
  }

  forward(input: Tensor<S, D, Dev>): Tensor<S, D, Dev> {
    // softplus(x) = (1/beta) * log(1 + exp(beta * x))
    // For numerical stability, when beta*x > threshold, return x
    const scaled = input.mulScalar(this.beta)
    const softplus = scaled.exp().addScalar(1).log().divScalar(this.beta)
    // Note: Would need where() for proper thresholding
    return softplus
  }

  override toString(): string {
    return `Softplus(beta=${this.beta}, threshold=${this.threshold})`
  }
}

/**
 * Softsign activation: x / (1 + |x|)
 *
 * Smooth approximation to sign function.
 */
export class Softsign<S extends Shape = Shape, D extends DType<string> = float32, Dev extends DeviceType = DeviceType> extends Module<S, S, D, Dev> {
  forward(input: Tensor<S, D, Dev>): Tensor<S, D, Dev> {
    const absX = (input as any).abs()
    return input.div(absX.addScalar(1))
  }

  override toString(): string {
    return 'Softsign()'
  }
}

/**
 * Log Softmax activation
 *
 * More numerically stable than computing log(softmax(x)) separately.
 */
export class LogSoftmax<S extends Shape = Shape, D extends DType<string> = float32, Dev extends DeviceType = DeviceType> extends Module<S, S, D, Dev> {
  constructor(public readonly dim: number = -1) {
    super()
  }

  forward(input: Tensor<S, D, Dev>): Tensor<S, D, Dev> {
    return input.logSoftmax(this.dim)
  }

  override toString(): string {
    return `LogSoftmax(dim=${this.dim})`
  }
}
