/**
 * Activation function modules with type-safe shape preservation
 *
 * Activation functions maintain input shape, making them easy to compose.
 */

import { Module, type Tensor, type float32 } from '../module.js';
import type { Shape, DType } from '@ts-torch/core';

/**
 * Rectified Linear Unit activation: ReLU(x) = max(0, x)
 *
 * Shape-preserving: input and output have identical shapes.
 *
 * @template S - Tensor shape (preserved through activation)
 * @template D - Data type (default: float32)
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
export class ReLU<
  S extends Shape = Shape,
  D extends DType<string> = float32
> extends Module<S, S, D> {
  /**
   * Create a new ReLU activation
   *
   * @param inplace - Whether to modify input tensor in-place (default: false)
   */
  constructor(public readonly inplace: boolean = false) {
    super();
  }

  /**
   * Forward pass: ReLU(x) = max(0, x)
   *
   * @param input - Input tensor
   * @returns Output tensor with same shape, all negative values set to 0
   */
  forward(input: Tensor<S, D>): Tensor<S, D> {
    // TODO: Implement actual ReLU when Tensor ops are ready
    // return input.relu();
    // or: return input.clamp(0, Infinity);

    return input as any; // Placeholder - maintains type safety
  }

  override toString(): string {
    return `ReLU(inplace=${this.inplace})`;
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
 *
 * @example
 * ```ts
 * const sigmoid = new Sigmoid<readonly [32, 10]>();
 * const logits: Tensor<readonly [32, 10]> = ...;
 * const probs = sigmoid.forward(logits); // Type: Tensor<readonly [32, 10]>
 * ```
 */
export class Sigmoid<
  S extends Shape = Shape,
  D extends DType<string> = float32
> extends Module<S, S, D> {
  /**
   * Forward pass: σ(x) = 1 / (1 + e^(-x))
   *
   * @param input - Input tensor
   * @returns Output tensor with same shape, values in (0, 1)
   */
  forward(input: Tensor<S, D>): Tensor<S, D> {
    // TODO: Implement actual Sigmoid when Tensor ops are ready
    // return input.sigmoid();
    // or: return 1 / (1 + input.neg().exp());

    return input as any; // Placeholder - maintains type safety
  }

  override toString(): string {
    return 'Sigmoid()';
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
 *
 * @example
 * ```ts
 * const tanh = new Tanh<readonly [32, 64]>();
 * const input: Tensor<readonly [32, 64]> = ...;
 * const output = tanh.forward(input); // Type: Tensor<readonly [32, 64]>
 * ```
 */
export class Tanh<
  S extends Shape = Shape,
  D extends DType<string> = float32
> extends Module<S, S, D> {
  /**
   * Forward pass: tanh(x)
   *
   * @param input - Input tensor
   * @returns Output tensor with same shape, values in (-1, 1)
   */
  forward(input: Tensor<S, D>): Tensor<S, D> {
    // TODO: Implement actual Tanh when Tensor ops are ready
    // return input.tanh();

    return input as any; // Placeholder - maintains type safety
  }

  override toString(): string {
    return 'Tanh()';
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
export class Softmax<
  S extends Shape = Shape,
  D extends DType<string> = float32
> extends Module<S, S, D> {
  /**
   * Create a new Softmax activation
   *
   * @param dim - Dimension along which to apply softmax (default: -1, last dimension)
   */
  constructor(public readonly dim: number = -1) {
    super();
  }

  /**
   * Forward pass: softmax(x) along specified dimension
   *
   * @param input - Input tensor
   * @returns Output tensor with same shape, values sum to 1 along dim
   */
  forward(input: Tensor<S, D>): Tensor<S, D> {
    // TODO: Implement actual Softmax when Tensor ops are ready
    // return input.softmax(this.dim);
    //
    // Numerically stable implementation:
    // const maxVals = input.max(dim=this.dim, keepdim=true);
    // const exp = input.sub(maxVals).exp();
    // return exp.div(exp.sum(dim=this.dim, keepdim=true));

    return input as any; // Placeholder - maintains type safety
  }

  override toString(): string {
    return `Softmax(dim=${this.dim})`;
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
 *
 * @example
 * ```ts
 * const leaky = new LeakyReLU<readonly [32, 128]>(0.01);
 * const input: Tensor<readonly [32, 128]> = ...;
 * const output = leaky.forward(input); // Type: Tensor<readonly [32, 128]>
 * ```
 */
export class LeakyReLU<
  S extends Shape = Shape,
  D extends DType<string> = float32
> extends Module<S, S, D> {
  /**
   * Create a new Leaky ReLU activation
   *
   * @param negativeSlope - Slope for negative values (default: 0.01)
   * @param inplace - Whether to modify input tensor in-place (default: false)
   */
  constructor(
    public readonly negativeSlope: number = 0.01,
    public readonly inplace: boolean = false
  ) {
    super();
  }

  /**
   * Forward pass: LeakyReLU(x) = max(αx, x)
   *
   * @param input - Input tensor
   * @returns Output tensor with same shape
   */
  forward(input: Tensor<S, D>): Tensor<S, D> {
    // TODO: Implement actual LeakyReLU when Tensor ops are ready
    // return input.leaky_relu(this.negativeSlope);
    // or: return input.maximum(input.mul(this.negativeSlope));

    return input as any; // Placeholder - maintains type safety
  }

  override toString(): string {
    return `LeakyReLU(negative_slope=${this.negativeSlope}, inplace=${this.inplace})`;
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
 *
 * @example
 * ```ts
 * const gelu = new GELU<readonly [32, 768]>();
 * const input: Tensor<readonly [32, 768]> = ...;
 * const output = gelu.forward(input); // Type: Tensor<readonly [32, 768]>
 * ```
 */
export class GELU<
  S extends Shape = Shape,
  D extends DType<string> = float32
> extends Module<S, S, D> {
  /**
   * Forward pass: GELU(x)
   *
   * @param input - Input tensor
   * @returns Output tensor with same shape
   */
  forward(input: Tensor<S, D>): Tensor<S, D> {
    // TODO: Implement actual GELU when Tensor ops are ready
    // return input.gelu();
    //
    // Tanh approximation:
    // const x3 = input.pow(3);
    // const inner = Math.sqrt(2/Math.PI) * (input + 0.044715 * x3);
    // return 0.5 * input * (1 + inner.tanh());

    return input as any; // Placeholder - maintains type safety
  }

  override toString(): string {
    return 'GELU()';
  }
}
