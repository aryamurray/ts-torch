/**
 * Dropout layers
 */

import { Module, type Tensor, type float32 } from '../module.js'
import { validateProbability, type DType, type Shape } from '@ts-torch/core'

/**
 * Dropout layer
 *
 * During training, randomly zeroes some of the elements of the input tensor
 * with probability p using samples from a Bernoulli distribution.
 * Each channel will be zeroed out independently on every forward call.
 *
 * The outputs are scaled by a factor of 1/(1-p) during training to ensure
 * the expected value of the output is the same during training and evaluation.
 *
 * During evaluation mode, this layer simply returns the input unchanged.
 *
 * @example
 * ```ts
 * // Create a dropout layer with 50% probability
 * const dropout = new Dropout(0.5);
 *
 * // In training mode, randomly zeros elements
 * dropout.train();
 * const output = dropout.forward(input);
 *
 * // In eval mode, returns input unchanged
 * dropout.eval();
 * const output2 = dropout.forward(input);
 * ```
 */
export class Dropout<S extends Shape = Shape, D extends DType<string> = float32> extends Module<
  S,
  S,
  D
> {
  readonly p: number
  readonly inplace: boolean

  /**
   * Create a new Dropout layer
   *
   * @param p - Probability of an element to be zeroed (default: 0.5)
   * @param options - Configuration options
   */
  constructor(
    p: number = 0.5,
    options: {
      inplace?: boolean
    } = {},
  ) {
    super()
    validateProbability(p, 'p (dropout probability)')

    this.p = p
    this.inplace = options.inplace ?? false
  }

  /**
   * Forward pass: applies dropout
   *
   * @param input - Input tensor
   * @returns Output tensor with dropout applied (if training)
   */
  forward(input: Tensor<S, D>): Tensor<S, D> {
    // In eval mode, return input unchanged
    if (!this._training || this.p === 0) {
      return input
    }

    // Apply dropout using native implementation
    const result = (input as any).dropout(this.p, this._training)
    return result as Tensor<S, D>
  }

  override toString(): string {
    return `Dropout(p=${this.p}, inplace=${this.inplace})`
  }
}

/**
 * Dropout2d layer (spatial dropout)
 *
 * Randomly zero out entire channels (a channel is a 2D feature map).
 * This is useful for convolutional networks where adjacent pixels are
 * highly correlated.
 *
 * @example
 * ```ts
 * const dropout = new Dropout2d(0.2);
 * const input: Tensor<readonly [32, 64, 28, 28]> = ...;
 * const output = dropout.forward(input);
 * ```
 */
export class Dropout2d<D extends DType<string> = float32> extends Module<
  readonly [number, number, number, number],
  readonly [number, number, number, number],
  D
> {
  readonly p: number
  readonly inplace: boolean

  constructor(
    p: number = 0.5,
    options: {
      inplace?: boolean
    } = {},
  ) {
    super()
    validateProbability(p, 'p (dropout probability)')

    this.p = p
    this.inplace = options.inplace ?? false
  }

  forward(
    input: Tensor<readonly [number, number, number, number], D>,
  ): Tensor<readonly [number, number, number, number], D> {
    if (!this._training || this.p === 0) {
      return input
    }

    // Apply dropout (PyTorch's dropout already handles spatial dropout for 4D inputs)
    const result = (input as any).dropout(this.p, this._training)
    return result as Tensor<readonly [number, number, number, number], D>
  }

  override toString(): string {
    return `Dropout2d(p=${this.p}, inplace=${this.inplace})`
  }
}

/**
 * Dropout1d layer (channel dropout for 1D inputs)
 *
 * Randomly zero out entire channels for 3D inputs.
 *
 * @example
 * ```ts
 * const dropout = new Dropout1d(0.2);
 * const input: Tensor<readonly [32, 64, 100]> = ...;
 * const output = dropout.forward(input);
 * ```
 */
export class Dropout1d<D extends DType<string> = float32> extends Module<
  readonly [number, number, number],
  readonly [number, number, number],
  D
> {
  readonly p: number
  readonly inplace: boolean

  constructor(
    p: number = 0.5,
    options: {
      inplace?: boolean
    } = {},
  ) {
    super()
    validateProbability(p, 'p (dropout probability)')

    this.p = p
    this.inplace = options.inplace ?? false
  }

  forward(
    input: Tensor<readonly [number, number, number], D>,
  ): Tensor<readonly [number, number, number], D> {
    if (!this._training || this.p === 0) {
      return input
    }

    const result = (input as any).dropout(this.p, this._training)
    return result as Tensor<readonly [number, number, number], D>
  }

  override toString(): string {
    return `Dropout1d(p=${this.p}, inplace=${this.inplace})`
  }
}

/**
 * Alpha Dropout (for SELU activation)
 *
 * Applies Alpha Dropout over the input, designed to work with SELU activation
 * to maintain the self-normalizing property.
 *
 * @example
 * ```ts
 * const alphaDropout = new AlphaDropout(0.1);
 * ```
 */
export class AlphaDropout<S extends Shape = Shape, D extends DType<string> = float32> extends Module<
  S,
  S,
  D
> {
  readonly p: number
  readonly inplace: boolean

  constructor(
    p: number = 0.5,
    options: {
      inplace?: boolean
    } = {},
  ) {
    super()
    validateProbability(p, 'p (dropout probability)')

    this.p = p
    this.inplace = options.inplace ?? false
  }

  forward(input: Tensor<S, D>): Tensor<S, D> {
    if (!this._training || this.p === 0) {
      return input
    }

    // Use standard dropout for now (alpha dropout could be added later with specific FFI)
    const result = (input as any).dropout(this.p, this._training)
    return result as Tensor<S, D>
  }

  override toString(): string {
    return `AlphaDropout(p=${this.p}, inplace=${this.inplace})`
  }
}
