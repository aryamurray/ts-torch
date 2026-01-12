/**
 * Pooling layers
 */

import { Module, type Tensor, type float32 } from '../module.js'
import { validatePoolingParams, validatePositiveInt, type DType } from '@ts-torch/core'

/**
 * Helper to normalize kernel_size, stride, padding to tuple form
 */
function normalizePair(value: number | [number, number]): [number, number] {
  if (typeof value === 'number') {
    return [value, value]
  }
  return value
}

/**
 * 2D Max pooling layer
 *
 * Applies 2D max pooling over an input signal composed of several input planes.
 * Returns the maximum value in each pooling window.
 *
 * Input shape: [N, C, H, W]
 * Output shape: [N, C, H_out, W_out]
 *
 * @example
 * ```ts
 * // Create a max pooling layer with 2x2 kernel
 * const pool = new MaxPool2d(2);
 *
 * // With stride and padding
 * const pool2 = new MaxPool2d([3, 3], { stride: 2, padding: 1 });
 *
 * // Forward pass
 * const input: Tensor<readonly [1, 64, 32, 32]> = ...;
 * const output = pool.forward(input); // [1, 64, 16, 16]
 * ```
 */
export class MaxPool2d<D extends DType<string> = float32> extends Module<
  readonly [number, number, number, number],
  readonly [number, number, number, number],
  D
> {
  readonly kernelSize: [number, number]
  readonly stride: [number, number]
  readonly padding: [number, number]

  /**
   * Create a new MaxPool2d layer
   *
   * @param kernelSize - Size of the pooling window
   * @param options - Configuration options
   */
  constructor(
    kernelSize: number | [number, number],
    options: {
      stride?: number | [number, number]
      padding?: number | [number, number]
    } = {},
  ) {
    super()

    this.kernelSize = normalizePair(kernelSize)
    // Default stride is same as kernel size
    this.stride = options.stride !== undefined ? normalizePair(options.stride) : this.kernelSize
    this.padding = normalizePair(options.padding ?? 0)

    validatePoolingParams({
      kernelSize: this.kernelSize,
      stride: this.stride,
      padding: this.padding,
    })
  }

  /**
   * Forward pass: applies 2D max pooling
   *
   * @param input - Input tensor with shape [N, C, H, W]
   * @returns Output tensor with shape [N, C, H_out, W_out]
   */
  forward(
    input: Tensor<readonly [number, number, number, number], D>,
  ): Tensor<readonly [number, number, number, number], D> {
    const result = (input as any).maxPool2d(this.kernelSize, this.stride, this.padding)
    return result as Tensor<readonly [number, number, number, number], D>
  }

  override toString(): string {
    return `MaxPool2d(kernel_size=${JSON.stringify(this.kernelSize)}, stride=${JSON.stringify(this.stride)}, padding=${JSON.stringify(this.padding)})`
  }
}

/**
 * 2D Average pooling layer
 *
 * Applies 2D average pooling over an input signal composed of several input planes.
 * Returns the average value in each pooling window.
 *
 * Input shape: [N, C, H, W]
 * Output shape: [N, C, H_out, W_out]
 *
 * @example
 * ```ts
 * // Create an average pooling layer with 2x2 kernel
 * const pool = new AvgPool2d(2);
 *
 * // With stride and padding
 * const pool2 = new AvgPool2d([3, 3], { stride: 2, padding: 1 });
 *
 * // Forward pass
 * const input: Tensor<readonly [1, 64, 32, 32]> = ...;
 * const output = pool.forward(input); // [1, 64, 16, 16]
 * ```
 */
export class AvgPool2d<D extends DType<string> = float32> extends Module<
  readonly [number, number, number, number],
  readonly [number, number, number, number],
  D
> {
  readonly kernelSize: [number, number]
  readonly stride: [number, number]
  readonly padding: [number, number]

  /**
   * Create a new AvgPool2d layer
   *
   * @param kernelSize - Size of the pooling window
   * @param options - Configuration options
   */
  constructor(
    kernelSize: number | [number, number],
    options: {
      stride?: number | [number, number]
      padding?: number | [number, number]
    } = {},
  ) {
    super()

    this.kernelSize = normalizePair(kernelSize)
    // Default stride is same as kernel size
    this.stride = options.stride !== undefined ? normalizePair(options.stride) : this.kernelSize
    this.padding = normalizePair(options.padding ?? 0)

    validatePoolingParams({
      kernelSize: this.kernelSize,
      stride: this.stride,
      padding: this.padding,
    })
  }

  /**
   * Forward pass: applies 2D average pooling
   *
   * @param input - Input tensor with shape [N, C, H, W]
   * @returns Output tensor with shape [N, C, H_out, W_out]
   */
  forward(
    input: Tensor<readonly [number, number, number, number], D>,
  ): Tensor<readonly [number, number, number, number], D> {
    const result = (input as any).avgPool2d(this.kernelSize, this.stride, this.padding)
    return result as Tensor<readonly [number, number, number, number], D>
  }

  override toString(): string {
    return `AvgPool2d(kernel_size=${JSON.stringify(this.kernelSize)}, stride=${JSON.stringify(this.stride)}, padding=${JSON.stringify(this.padding)})`
  }
}

/**
 * Adaptive 2D Average pooling layer
 *
 * Applies 2D adaptive average pooling to produce a fixed output size
 * regardless of input size.
 *
 * Input shape: [N, C, H, W]
 * Output shape: [N, C, output_size[0], output_size[1]]
 *
 * @example
 * ```ts
 * // Pool to 1x1 output (global average pooling)
 * const gap = new AdaptiveAvgPool2d([1, 1]);
 *
 * // Pool to 7x7 output
 * const pool = new AdaptiveAvgPool2d([7, 7]);
 * ```
 */
export class AdaptiveAvgPool2d<D extends DType<string> = float32> extends Module<
  readonly [number, number, number, number],
  readonly [number, number, number, number],
  D
> {
  readonly outputSize: [number, number]

  /**
   * Create a new AdaptiveAvgPool2d layer
   *
   * @param outputSize - Target output size [H, W]
   */
  constructor(outputSize: number | [number, number]) {
    super()
    this.outputSize = typeof outputSize === 'number' ? [outputSize, outputSize] : outputSize
    validatePositiveInt(this.outputSize[0], 'outputSize[0]')
    validatePositiveInt(this.outputSize[1], 'outputSize[1]')
  }

  /**
   * Forward pass: applies adaptive 2D average pooling
   *
   * @param input - Input tensor with shape [N, C, H, W]
   * @returns Output tensor with shape [N, C, output_H, output_W]
   */
  forward(
    input: Tensor<readonly [number, number, number, number], D>,
  ): Tensor<readonly [number, number, number, number], D> {
    // Compute kernel size and stride to achieve target output size
    const H_in = (input as any).shape[2] as number
    const W_in = (input as any).shape[3] as number
    const [H_out, W_out] = this.outputSize

    // Kernel size is ceil(input_size / output_size)
    const kernelH = Math.ceil(H_in / H_out)
    const kernelW = Math.ceil(W_in / W_out)

    // Stride is floor(input_size / output_size)
    const strideH = Math.floor(H_in / H_out)
    const strideW = Math.floor(W_in / W_out)

    const result = (input as any).avgPool2d([kernelH, kernelW], [strideH, strideW], [0, 0])
    return result as Tensor<readonly [number, number, number, number], D>
  }

  override toString(): string {
    return `AdaptiveAvgPool2d(output_size=${JSON.stringify(this.outputSize)})`
  }
}
