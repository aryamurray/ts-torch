import type { DType, DeviceType, Shape } from '@ts-torch/core'
import { Module, type Tensor, type float32 } from '../module.js'

/**
 * Flattens a contiguous range of dims into a single dim.
 *
 * @example
 * ```ts
 * const flatten = new Flatten()          // default: startDim=1, endDim=-1
 * const out = flatten.forward(input)     // [B, C, H, W] -> [B, C*H*W]
 * ```
 */
export class Flatten<D extends DType<string> = float32, Dev extends DeviceType = DeviceType> extends Module<
  Shape,
  Shape,
  D,
  Dev
> {
  readonly startDim: number
  readonly endDim: number

  constructor(startDim = 1, endDim = -1) {
    super()
    this.startDim = startDim
    this.endDim = endDim
  }

  forward(input: Tensor<Shape, D, Dev>): Tensor<Shape, D, Dev> {
    return (input as any).flatten(this.startDim, this.endDim)
  }

  override toString(): string {
    return `Flatten(start_dim=${this.startDim}, end_dim=${this.endDim})`
  }
}
