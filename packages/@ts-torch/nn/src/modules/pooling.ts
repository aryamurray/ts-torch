/**
 * Pooling layers
 */

import { Module, type Tensor } from "../module.js";

/**
 * 2D Max pooling layer
 */
export class MaxPool2d extends Module {
  constructor(
    public kernelSize: number | [number, number],
    public stride?: number | [number, number],
    public padding: number | [number, number] = 0,
  ) {
    super();
  }

  forward(_input: Tensor): Tensor {
    // TODO: Implement 2D max pooling
    throw new Error("MaxPool2d.forward not yet implemented");
  }

  override toString(): string {
    return `MaxPool2d(kernel_size=${JSON.stringify(this.kernelSize)})`;
  }
}

/**
 * 2D Average pooling layer
 */
export class AvgPool2d extends Module {
  constructor(
    public kernelSize: number | [number, number],
    public stride?: number | [number, number],
    public padding: number | [number, number] = 0,
  ) {
    super();
  }

  forward(_input: Tensor): Tensor {
    // TODO: Implement 2D average pooling
    throw new Error("AvgPool2d.forward not yet implemented");
  }

  override toString(): string {
    return `AvgPool2d(kernel_size=${JSON.stringify(this.kernelSize)})`;
  }
}
