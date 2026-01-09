/**
 * Convolutional layers
 */

import { Module, type Tensor } from '../module.js';

/**
 * 2D Convolutional layer
 */
export class Conv2d extends Module {
  constructor(
    public inChannels: number,
    public outChannels: number,
    public kernelSize: number | [number, number],
    public stride: number | [number, number] = 1,
    public padding: number | [number, number] = 0,
    public dilation: number | [number, number] = 1,
    public groups: number = 1,
    public bias: boolean = true
  ) {
    super();
    // TODO: Initialize conv weights and bias
  }

  forward(_input: Tensor): Tensor {
    // TODO: Implement 2D convolution
    throw new Error('Conv2d.forward not yet implemented');
  }

  override toString(): string {
    return `Conv2d(${this.inChannels}, ${this.outChannels}, kernel_size=${JSON.stringify(this.kernelSize)})`;
  }
}
