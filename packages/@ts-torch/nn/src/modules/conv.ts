/**
 * Convolutional layers
 */

import { Module, Parameter, type Tensor, type float32 } from '../module.js'
import { device, type DType, validateConv2dParams } from '@ts-torch/core'

// CPU device for weight initialization
const cpu = device.cpu()

/**
 * Helper to normalize kernel_size, stride, padding, dilation to tuple form
 */
function normalizePair(value: number | [number, number]): [number, number] {
  if (typeof value === 'number') {
    return [value, value]
  }
  return value
}

/**
 * Conv2d options interface
 */
export interface Conv2dOptions<D extends DType<string> = float32> {
  /**
   * Stride of the convolution (default: 1)
   */
  stride?: number | [number, number]

  /**
   * Padding added to input (default: 0)
   */
  padding?: number | [number, number]

  /**
   * Dilation of the kernel (default: 1)
   */
  dilation?: number | [number, number]

  /**
   * Number of groups for grouped convolution (default: 1)
   */
  groups?: number

  /**
   * Whether to include bias term (default: true)
   */
  bias?: boolean

  /**
   * Data type for weights and bias (default: float32)
   */
  dtype?: D
}

/**
 * 2D Convolutional layer
 *
 * Applies a 2D convolution over an input signal composed of several input planes.
 *
 * Input shape: [N, C_in, H, W]
 * Output shape: [N, C_out, H_out, W_out]
 *
 * @template InChannels - Number of input channels
 * @template OutChannels - Number of output channels
 * @template D - Data type (default: float32)
 *
 * @example
 * ```ts
 * // Create a conv layer: 3 input channels, 64 output channels, 3x3 kernel
 * const conv = new Conv2d(3, 64, 3);
 *
 * // With options
 * const conv2 = new Conv2d(64, 128, 3, { stride: 2, padding: 1 });
 *
 * // Forward pass
 * const input: Tensor<readonly [1, 3, 224, 224]> = ...;
 * const output = conv.forward(input);
 * ```
 */
export class Conv2d<
  InChannels extends number = number,
  OutChannels extends number = number,
  D extends DType<string> = float32,
> extends Module<
  readonly [number, InChannels, number, number],
  readonly [number, OutChannels, number, number],
  D
> {
  readonly inChannels: InChannels
  readonly outChannels: OutChannels
  readonly kernelSize: [number, number]
  readonly stride: [number, number]
  readonly padding: [number, number]
  readonly dilation: [number, number]
  readonly groups: number

  /**
   * Weight tensor with shape [OutChannels, InChannels/groups, KernelH, KernelW]
   */
  readonly weight: Parameter<readonly [OutChannels, number, number, number], D>

  /**
   * Bias vector with shape [OutChannels]
   * Null if bias is disabled
   */
  readonly biasParam: Parameter<readonly [OutChannels], D> | null

  /**
   * Create a new Conv2d layer
   *
   * @param inChannels - Number of input channels
   * @param outChannels - Number of output channels
   * @param kernelSize - Size of the convolving kernel
   * @param options - Configuration options
   */
  constructor(
    inChannels: InChannels,
    outChannels: OutChannels,
    kernelSize: number | [number, number],
    options: Conv2dOptions<D> = {},
  ) {
    super()

    this.inChannels = inChannels
    this.outChannels = outChannels
    this.kernelSize = normalizePair(kernelSize)
    this.stride = normalizePair(options.stride ?? 1)
    this.padding = normalizePair(options.padding ?? 0)
    this.dilation = normalizePair(options.dilation ?? 1)
    this.groups = options.groups ?? 1

    const { bias = true } = options

    // Validate all Conv2d parameters
    validateConv2dParams({
      inChannels,
      outChannels,
      kernelSize: this.kernelSize,
      stride: this.stride,
      padding: this.padding,
      dilation: this.dilation,
      groups: this.groups,
    })

    // Initialize weight: [OutChannels, InChannels/groups, KernelH, KernelW]
    const inChannelsPerGroup = inChannels / this.groups
    const weightShape = [outChannels, inChannelsPerGroup, this.kernelSize[0], this.kernelSize[1]] as const

    // Kaiming initialization for conv layers
    const fanIn = inChannelsPerGroup * this.kernelSize[0] * this.kernelSize[1]
    const std = Math.sqrt(2.0 / fanIn)
    const randWeight = cpu.randn(weightShape)
    const weightTensor = (randWeight as any).mulScalar(std) as Tensor<
      readonly [OutChannels, number, number, number],
      D
    >
    weightTensor.escape()

    this.weight = new Parameter(weightTensor, true)
    this.registerParameter('weight', this.weight)

    // Initialize bias if enabled
    if (bias) {
      const biasShape = [outChannels] as const
      const biasTensor = cpu.zeros(biasShape) as unknown as Tensor<readonly [OutChannels], D>
      biasTensor.escape()

      this.biasParam = new Parameter(biasTensor, true)
      this.registerParameter('bias', this.biasParam)
    } else {
      this.biasParam = null
    }
  }

  /**
   * Forward pass: applies 2D convolution
   *
   * @param input - Input tensor with shape [N, C_in, H, W]
   * @returns Output tensor with shape [N, C_out, H_out, W_out]
   */
  forward(
    input: Tensor<readonly [number, InChannels, number, number], D>,
  ): Tensor<readonly [number, OutChannels, number, number], D> {
    // Use native conv2d operation
    const result = (input as any).conv2d(
      this.weight.data,
      this.biasParam?.data ?? null,
      this.stride,
      this.padding,
      this.dilation,
      this.groups,
    )

    return result as Tensor<readonly [number, OutChannels, number, number], D>
  }

  override toString(): string {
    return `Conv2d(${this.inChannels}, ${this.outChannels}, kernel_size=${JSON.stringify(this.kernelSize)}, stride=${JSON.stringify(this.stride)}, padding=${JSON.stringify(this.padding)})`
  }
}
