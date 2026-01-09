/**
 * AdamW optimizer
 */

import type { Tensor } from '@ts-torch/core';
import { Optimizer, type ParameterGroup, type OptimizerOptions } from './optimizer.js';

/**
 * AdamW optimizer options
 */
export interface AdamWOptions extends OptimizerOptions {
  lr: number;
  betas?: [number, number];
  eps?: number;
  weightDecay?: number;
  amsgrad?: boolean;
}

/**
 * AdamW optimizer
 *
 * Implements AdamW algorithm (Adam with decoupled weight decay).
 * This is a variant of Adam that decouples the weight decay from the gradient-based update.
 *
 * Reference: "Decoupled Weight Decay Regularization"
 * https://arxiv.org/abs/1711.05101
 *
 * @example
 * ```typescript
 * const optimizer = new AdamW(model.parameters(), {
 *   lr: 0.001,
 *   betas: [0.9, 0.999],
 *   eps: 1e-8,
 *   weightDecay: 0.01
 * });
 * ```
 */
export class AdamW extends Optimizer {
  declare defaults: AdamWOptions;

  constructor(
    params: Tensor[] | ParameterGroup[],
    options: AdamWOptions
  ) {
    const defaults: AdamWOptions = {
      lr: options.lr,
      betas: options.betas ?? [0.9, 0.999],
      eps: options.eps ?? 1e-8,
      weightDecay: options.weightDecay ?? 0.01,
      amsgrad: options.amsgrad ?? false,
    };

    super(params, defaults);
  }

  step(): void {
    for (const group of this.paramGroups) {
      const lr = (group.lr ?? this.defaults.lr);
      const [beta1, beta2] = (group.betas as [number, number] | undefined) ?? this.defaults.betas ?? [0.9, 0.999];
      const eps = (group.eps as number | undefined) ?? this.defaults.eps ?? 1e-8;
      const weightDecay = (group.weightDecay as number | undefined) ?? this.defaults.weightDecay ?? 0.01;
      const amsgrad = (group.amsgrad as boolean | undefined) ?? this.defaults.amsgrad ?? false;

      for (const param of group.params) {
        // TODO: Implement AdamW update
        // if (param.grad === null) continue;
        //
        // const paramState = this.state.get(param) ?? {
        //   step: 0,
        //   exp_avg: zeros_like(param),
        //   exp_avg_sq: zeros_like(param),
        // };
        //
        // const state = paramState as {
        //   step: number;
        //   exp_avg: Tensor;
        //   exp_avg_sq: Tensor;
        //   max_exp_avg_sq?: Tensor;
        // };
        //
        // state.step += 1;
        //
        // // Perform weight decay (decoupled from gradient update)
        // param.data = param.mul(1 - lr * weightDecay);
        //
        // // Update biased first moment estimate
        // state.exp_avg = state.exp_avg.mul(beta1).add(param.grad.mul(1 - beta1));
        //
        // // Update biased second raw moment estimate
        // state.exp_avg_sq = state.exp_avg_sq.mul(beta2).add(param.grad.pow(2).mul(1 - beta2));
        //
        // // Compute bias-corrected moment estimates
        // const bias_correction1 = 1 - Math.pow(beta1, state.step);
        // const bias_correction2 = 1 - Math.pow(beta2, state.step);
        //
        // let denom: Tensor;
        // if (amsgrad) {
        //   if (!state.max_exp_avg_sq) {
        //     state.max_exp_avg_sq = state.exp_avg_sq.clone();
        //   }
        //   state.max_exp_avg_sq = state.max_exp_avg_sq.maximum(state.exp_avg_sq);
        //   denom = state.max_exp_avg_sq.sqrt().div(Math.sqrt(bias_correction2)).add(eps);
        // } else {
        //   denom = state.exp_avg_sq.sqrt().div(Math.sqrt(bias_correction2)).add(eps);
        // }
        //
        // const step_size = lr / bias_correction1;
        // param.data = param.sub(state.exp_avg.div(denom).mul(step_size));
        //
        // this.state.set(param, state);
      }
    }
  }

  override toString(): string {
    return `AdamW(lr=${this.defaults.lr}, betas=${JSON.stringify(this.defaults.betas)}, eps=${this.defaults.eps}, weight_decay=${this.defaults.weightDecay})`;
  }
}
