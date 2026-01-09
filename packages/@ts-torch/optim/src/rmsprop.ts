/**
 * RMSprop optimizer
 */

import type { Tensor } from '@ts-torch/core';
import { Optimizer, type ParameterGroup, type OptimizerOptions } from './optimizer.js';

/**
 * RMSprop optimizer options
 */
export interface RMSpropOptions extends OptimizerOptions {
  lr: number;
  alpha?: number;
  eps?: number;
  weightDecay?: number;
  momentum?: number;
  centered?: boolean;
}

/**
 * RMSprop optimizer
 *
 * Implements RMSprop (Root Mean Square Propagation) algorithm.
 * Maintains a moving average of squared gradients and divides the gradient by the root of this average.
 *
 * Reference: Lecture 6.5 - RMSprop, COURSERA: Neural Networks for Machine Learning
 * by Tieleman and Hinton (2012)
 *
 * @example
 * ```typescript
 * const optimizer = new RMSprop(model.parameters(), {
 *   lr: 0.01,
 *   alpha: 0.99,
 *   eps: 1e-8,
 *   weightDecay: 0,
 *   momentum: 0
 * });
 * ```
 */
export class RMSprop extends Optimizer {
  declare defaults: RMSpropOptions;

  constructor(
    params: Tensor[] | ParameterGroup[],
    options: RMSpropOptions
  ) {
    const defaults: RMSpropOptions = {
      lr: options.lr,
      alpha: options.alpha ?? 0.99,
      eps: options.eps ?? 1e-8,
      weightDecay: options.weightDecay ?? 0,
      momentum: options.momentum ?? 0,
      centered: options.centered ?? false,
    };

    super(params, defaults);
  }

  step(): void {
    for (const group of this.paramGroups) {
      const lr = (group.lr ?? this.defaults.lr);
      const alpha = (group.alpha as number | undefined) ?? this.defaults.alpha ?? 0.99;
      const eps = (group.eps as number | undefined) ?? this.defaults.eps ?? 1e-8;
      const weightDecay = (group.weightDecay as number | undefined) ?? this.defaults.weightDecay ?? 0;
      const momentum = (group.momentum as number | undefined) ?? this.defaults.momentum ?? 0;
      const centered = (group.centered as boolean | undefined) ?? this.defaults.centered ?? false;

      for (const param of group.params) {
        // TODO: Implement RMSprop update
        // if (param.grad === null) continue;
        //
        // const paramState = this.state.get(param) ?? {
        //   square_avg: zeros_like(param),
        // };
        //
        // const state = paramState as {
        //   square_avg: Tensor;
        //   grad_avg?: Tensor;
        //   momentum_buffer?: Tensor;
        // };
        //
        // let grad = param.grad;
        //
        // // Apply weight decay
        // if (weightDecay !== 0) {
        //   grad = grad.add(param.mul(weightDecay));
        // }
        //
        // // Update accumulator
        // state.square_avg = state.square_avg.mul(alpha).add(grad.pow(2).mul(1 - alpha));
        //
        // let avg: Tensor;
        // if (centered) {
        //   if (!state.grad_avg) {
        //     state.grad_avg = zeros_like(param);
        //   }
        //   state.grad_avg = state.grad_avg.mul(alpha).add(grad.mul(1 - alpha));
        //   avg = state.square_avg.sub(state.grad_avg.pow(2)).sqrt().add(eps);
        // } else {
        //   avg = state.square_avg.sqrt().add(eps);
        // }
        //
        // if (momentum > 0) {
        //   if (!state.momentum_buffer) {
        //     state.momentum_buffer = grad.div(avg).clone();
        //   } else {
        //     state.momentum_buffer = state.momentum_buffer.mul(momentum).add(grad.div(avg));
        //   }
        //   param.data = param.sub(state.momentum_buffer.mul(lr));
        // } else {
        //   param.data = param.sub(grad.div(avg).mul(lr));
        // }
        //
        // this.state.set(param, state);
      }
    }
  }

  override toString(): string {
    return `RMSprop(lr=${this.defaults.lr}, alpha=${this.defaults.alpha}, eps=${this.defaults.eps})`;
  }
}
