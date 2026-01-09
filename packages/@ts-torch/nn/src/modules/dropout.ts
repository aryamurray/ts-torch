/**
 * Dropout layers
 */

import { Module, type Tensor } from '../module.js'

/**
 * Dropout layer
 */
export class Dropout extends Module {
  constructor(
    public p: number = 0.5,
    public inplace: boolean = false,
  ) {
    super()

    if (p < 0 || p > 1) {
      throw new Error(`Dropout probability must be between 0 and 1, got ${p}`)
    }
  }

  forward(_input: Tensor): Tensor {
    if (!this.training) {
      return _input
    }

    // TODO: Implement dropout
    // During training, randomly zero elements with probability p
    // and scale remaining elements by 1/(1-p)
    throw new Error('Dropout.forward not yet implemented')
  }

  override toString(): string {
    return `Dropout(p=${this.p}, inplace=${this.inplace})`
  }
}
