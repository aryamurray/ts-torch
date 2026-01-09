/**
 * Normalization layers
 */

import { Module, type Tensor } from "../module.js";

/**
 * Batch normalization layer
 */
export class BatchNorm2d extends Module {
  constructor(
    public numFeatures: number,
    public eps: number = 1e-5,
    public momentum: number = 0.1,
    public affine: boolean = true,
    public trackRunningStats: boolean = true,
  ) {
    super();
    // TODO: Initialize running mean/var and learnable affine parameters
  }

  forward(_input: Tensor): Tensor {
    // TODO: Implement batch normalization
    throw new Error("BatchNorm2d.forward not yet implemented");
  }

  override toString(): string {
    return `BatchNorm2d(${this.numFeatures}, eps=${this.eps}, momentum=${this.momentum})`;
  }
}

/**
 * Layer normalization
 */
export class LayerNorm extends Module {
  constructor(
    public normalizedShape: number | number[],
    public eps: number = 1e-5,
    public elementwiseAffine: boolean = true,
  ) {
    super();
    // TODO: Initialize learnable affine parameters
  }

  forward(_input: Tensor): Tensor {
    // TODO: Implement layer normalization
    throw new Error("LayerNorm.forward not yet implemented");
  }

  override toString(): string {
    return `LayerNorm(${JSON.stringify(this.normalizedShape)}, eps=${this.eps})`;
  }
}
