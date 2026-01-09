/**
 * Adam optimizer
 */

import type { Tensor } from '@ts-torch/core';
import { Optimizer, type ParameterGroup, type OptimizerOptions } from './optimizer.js';

/**
 * Adam optimizer options
 */
export interface AdamOptions extends OptimizerOptions {
  lr: number;
  betas?: [number, number];
  eps?: number;
  weightDecay?: number;
  amsgrad?: boolean;
}

/**
 * Adam optimizer
 *
 * Implements Adam algorithm (Adaptive Moment Estimation).
 *
 * Reference: "Adam: A Method for Stochastic Optimization"
 * https://arxiv.org/abs/1412.6980
 *
 * @example
 * ```typescript
 * const optimizer = new Adam(model.parameters(), {
 *   lr: 0.001,
 *   betas: [0.9, 0.999],
 *   eps: 1e-8,
 *   weightDecay: 0
 * });
 * ```
 */
export class Adam extends Optimizer {
  declare defaults: AdamOptions;

  constructor(
    params: Tensor[] | ParameterGroup[],
    options: AdamOptions
  ) {
    const defaults: AdamOptions = {
      lr: options.lr,
      betas: options.betas ?? [0.9, 0.999],
      eps: options.eps ?? 1e-8,
      weightDecay: options.weightDecay ?? 0,
      amsgrad: options.amsgrad ?? false,
    };

    super(params, defaults);
  }

  step(): void {
    for (const group of this.paramGroups) {
      const lr = (group.lr ?? this.defaults.lr);
      const [beta1, beta2] = (group.betas as [number, number] | undefined) ?? this.defaults.betas ?? [0.9, 0.999];
      const eps = (group.eps as number | undefined) ?? this.defaults.eps ?? 1e-8;
      const weightDecay = (group.weightDecay as number | undefined) ?? this.defaults.weightDecay ?? 0;
      const amsgrad = (group.amsgrad as boolean | undefined) ?? this.defaults.amsgrad ?? false;

      for (const param of group.params) {
        if (!('grad' in param) || param.grad === null || param.grad === undefined) continue;

        // Initialize state for this parameter
        const paramState = this.state.get(param) ?? {
          step: 0,
          exp_avg: null as Tensor | null,
          exp_avg_sq: null as Tensor | null,
        };

        const state = paramState as {
          step: number;
          exp_avg: Tensor | null;
          exp_avg_sq: Tensor | null;
          max_exp_avg_sq?: Tensor;
        };

        // Increment step counter
        state.step += 1;

        let grad = param.grad as Tensor;

        // Apply weight decay (L2 regularization)
        if (weightDecay !== 0) {
          if ('add' in grad && 'mul' in param &&
              typeof grad.add === 'function' && typeof (param as { mul?: Function }).mul === 'function') {
            const weighted = (param.mul as any)(weightDecay) as Tensor;
            grad = grad.add(weighted) as Tensor;
          }
        }

        // Initialize exponential moving averages if needed
        if (state.exp_avg === null) {
          // Create zero tensor with same shape as gradient
          if ('mul' in grad && typeof grad.mul === 'function') {
            state.exp_avg = (grad.mul as any)(0) as Tensor;
            state.exp_avg_sq = (grad.mul as any)(0) as Tensor;
          } else {
            state.exp_avg = grad;
            state.exp_avg_sq = grad;
          }
        }

        // Update biased first moment estimate: m_t = beta1 * m_{t-1} + (1 - beta1) * grad
        if (state.exp_avg && 'mul' in state.exp_avg && 'add' in state.exp_avg && 'mul' in grad &&
            typeof state.exp_avg.mul === 'function' && typeof state.exp_avg.add === 'function' && typeof grad.mul === 'function') {
          const m1 = (state.exp_avg.mul as any)(beta1) as Tensor;
          const m2 = (grad.mul as any)(1 - beta1) as Tensor;
          state.exp_avg = m1.add(m2) as Tensor;
        }

        // Update biased second raw moment estimate: v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2
        if (state.exp_avg_sq && 'mul' in state.exp_avg_sq && 'add' in state.exp_avg_sq && 'mul' in grad && 'pow' in grad &&
            typeof state.exp_avg_sq.mul === 'function' && typeof state.exp_avg_sq.add === 'function' &&
            typeof grad.mul === 'function' && typeof grad.pow === 'function') {
          const v1 = (state.exp_avg_sq.mul as any)(beta2) as Tensor;
          const gradSq = (grad.pow as any)(2) as Tensor;
          const v2 = ('mul' in gradSq && typeof gradSq.mul === 'function') ? (gradSq.mul as any)(1 - beta2) as Tensor : gradSq;
          state.exp_avg_sq = v1.add(v2) as Tensor;
        }

        // Compute bias-corrected moment estimates
        const bias_correction1 = 1 - Math.pow(beta1, state.step);
        const bias_correction2 = 1 - Math.pow(beta2, state.step);

        // Compute denominator
        let denom: Tensor | null = null;
        if (amsgrad) {
          // AMSGrad variant: use max of all v_t
          if (!state.max_exp_avg_sq && state.exp_avg_sq) {
            if ('clone' in state.exp_avg_sq && typeof state.exp_avg_sq.clone === 'function') {
              state.max_exp_avg_sq = state.exp_avg_sq.clone() as Tensor;
            } else {
              state.max_exp_avg_sq = state.exp_avg_sq;
            }
          }

          if (state.max_exp_avg_sq && state.exp_avg_sq && 'maximum' in state.max_exp_avg_sq &&
              typeof (state.max_exp_avg_sq as { maximum?: Function }).maximum === 'function') {
            state.max_exp_avg_sq = ((state.max_exp_avg_sq as { maximum: (x: Tensor) => Tensor }).maximum(state.exp_avg_sq)) as Tensor;
          }

          if (state.max_exp_avg_sq && 'sqrt' in state.max_exp_avg_sq && 'div' in state.max_exp_avg_sq && 'add' in state.max_exp_avg_sq &&
              typeof (state.max_exp_avg_sq as { sqrt?: Function }).sqrt === 'function' &&
              typeof (state.max_exp_avg_sq as { div?: Function }).div === 'function' &&
              typeof (state.max_exp_avg_sq as { add?: Function }).add === 'function') {
            const sqrtTerm = ((state.max_exp_avg_sq as { sqrt: () => Tensor }).sqrt()) as Tensor;
            const divTerm = ('div' in sqrtTerm && typeof (sqrtTerm as { div?: Function }).div === 'function')
              ? (sqrtTerm.div as any)(Math.sqrt(bias_correction2)) as Tensor
              : sqrtTerm;
            denom = ('add' in divTerm && typeof (divTerm as { add?: Function }).add === 'function')
              ? (divTerm.add as any)(eps) as Tensor
              : divTerm;
          }
        } else {
          // Standard Adam
          if (state.exp_avg_sq && 'sqrt' in state.exp_avg_sq && 'div' in state.exp_avg_sq && 'add' in state.exp_avg_sq &&
              typeof (state.exp_avg_sq as { sqrt?: Function }).sqrt === 'function' &&
              typeof (state.exp_avg_sq as { div?: Function }).div === 'function' &&
              typeof (state.exp_avg_sq as { add?: Function }).add === 'function') {
            const sqrtTerm = ((state.exp_avg_sq as { sqrt: () => Tensor }).sqrt()) as Tensor;
            const divTerm = ('div' in sqrtTerm && typeof (sqrtTerm as { div?: Function }).div === 'function')
              ? (sqrtTerm.div as any)(Math.sqrt(bias_correction2)) as Tensor
              : sqrtTerm;
            denom = ('add' in divTerm && typeof (divTerm as { add?: Function }).add === 'function')
              ? (divTerm.add as any)(eps) as Tensor
              : divTerm;
          }
        }

        // Compute step size
        const step_size = lr / bias_correction1;

        // Update parameters: param = param - step_size * m_t / denom
        if (state.exp_avg && denom && 'div' in state.exp_avg && 'mul' in state.exp_avg && 'sub' in param &&
            typeof (state.exp_avg as { div?: Function }).div === 'function' &&
            typeof (state.exp_avg as { mul?: Function }).mul === 'function' &&
            typeof (param as { sub?: Function }).sub === 'function') {
          const update_num = ((state.exp_avg as { div: (x: Tensor) => Tensor }).div(denom)) as Tensor;
          const update = ('mul' in update_num && typeof (update_num as { mul?: Function }).mul === 'function')
            ? (update_num.mul as any)(step_size) as Tensor
            : update_num;
          const newParam = ((param as { sub: (x: Tensor) => Tensor }).sub(update)) as Tensor;

          // Update param data in place
          if ('data' in param) {
            (param as { data: unknown }).data = ('data' in newParam) ? newParam.data : newParam;
          }
        }

        this.state.set(param, state);
      }
    }
  }

  override toString(): string {
    return `Adam(lr=${this.defaults.lr}, betas=${JSON.stringify(this.defaults.betas)}, eps=${this.defaults.eps})`;
  }
}
