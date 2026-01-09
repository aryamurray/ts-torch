/**
 * Stochastic Gradient Descent optimizer
 */

import type { Tensor } from '@ts-torch/core';
import { Optimizer, type ParameterGroup, type OptimizerOptions } from './optimizer.js';

/**
 * SGD optimizer options
 */
export interface SGDOptions extends OptimizerOptions {
  lr: number;
  momentum?: number;
  dampening?: number;
  weightDecay?: number;
  nesterov?: boolean;
}

/**
 * Stochastic Gradient Descent (SGD) optimizer
 *
 * Implements SGD with optional momentum, weight decay, and Nesterov momentum.
 *
 * @example
 * ```typescript
 * const optimizer = new SGD(model.parameters(), {
 *   lr: 0.01,
 *   momentum: 0.9,
 *   weightDecay: 1e-4
 * });
 * ```
 */
export class SGD extends Optimizer {
  declare defaults: SGDOptions;

  constructor(
    params: Tensor[] | ParameterGroup[],
    options: SGDOptions
  ) {
    const defaults: SGDOptions = {
      lr: options.lr,
      momentum: options.momentum ?? 0,
      dampening: options.dampening ?? 0,
      weightDecay: options.weightDecay ?? 0,
      nesterov: options.nesterov ?? false,
    };

    // Validate nesterov
    if (defaults.nesterov && ((defaults.momentum ?? 0) <= 0 || defaults.dampening !== 0)) {
      throw new Error('Nesterov momentum requires a momentum and zero dampening');
    }

    super(params, defaults);
  }

  step(): void {
    for (const group of this.paramGroups) {
      const lr = (group.lr ?? this.defaults.lr);
      const momentum = (group.momentum as number | undefined) ?? this.defaults.momentum ?? 0;
      const dampening = (group.dampening as number | undefined) ?? this.defaults.dampening ?? 0;
      const weightDecay = (group.weightDecay as number | undefined) ?? this.defaults.weightDecay ?? 0;
      const nesterov = (group.nesterov as boolean | undefined) ?? this.defaults.nesterov ?? false;

      for (const param of group.params) {
        if (!('grad' in param) || param.grad === null || param.grad === undefined) continue;

        // Get gradient
        let d_p = param.grad as Tensor;

        // Apply weight decay (L2 regularization)
        // gradient = gradient + weight_decay * param
        if (weightDecay !== 0) {
          if ('add' in d_p && 'mul' in param && typeof d_p.add === 'function' && typeof (param as { mul?: Function }).mul === 'function') {
            const weighted = (param.mul as any)(weightDecay) as Tensor;
            d_p = d_p.add(weighted) as Tensor;
          }
        }

        // Apply momentum
        if (momentum !== 0) {
          const paramState = this.state.get(param) ?? {};

          if (!paramState.momentum_buffer) {
            // Initialize momentum buffer with current gradient
            if ('clone' in d_p && typeof d_p.clone === 'function') {
              paramState.momentum_buffer = d_p.clone();
            } else {
              paramState.momentum_buffer = d_p;
            }
          } else {
            const buf = paramState.momentum_buffer as Tensor;
            // velocity = momentum * velocity + (1 - dampening) * gradient
            if ('mul' in buf && 'add' in buf && 'mul' in d_p &&
                typeof buf.mul === 'function' && typeof buf.add === 'function' && typeof d_p.mul === 'function') {
              const momentumTerm = (buf.mul as any)(momentum) as Tensor;
              const gradientTerm = (d_p.mul as any)(1 - dampening) as Tensor;
              paramState.momentum_buffer = momentumTerm.add(gradientTerm);
            }
          }

          if (nesterov) {
            // Nesterov momentum: gradient = gradient + momentum * velocity
            const buf = paramState.momentum_buffer as Tensor;
            if ('add' in d_p && 'mul' in buf && typeof d_p.add === 'function' && typeof buf.mul === 'function') {
              const momentumTerm = (buf.mul as any)(momentum) as Tensor;
              d_p = d_p.add(momentumTerm) as Tensor;
            }
          } else {
            // Standard momentum: use velocity as gradient
            d_p = paramState.momentum_buffer as Tensor;
          }

          this.state.set(param, paramState);
        }

        // Update parameters: param = param - lr * gradient
        if ('sub' in param && 'mul' in d_p &&
            typeof (param as { sub?: Function }).sub === 'function' && typeof d_p.mul === 'function') {
          const update = (d_p.mul as any)(lr) as Tensor;
          const newParam = ((param as { sub: (x: Tensor) => Tensor }).sub(update)) as Tensor;

          // Update param data in place
          if ('data' in param) {
            (param as { data: unknown }).data = ('data' in newParam) ? newParam.data : newParam;
          }
        }
      }
    }
  }

  override toString(): string {
    return `SGD(lr=${this.defaults.lr}, momentum=${this.defaults.momentum}, weight_decay=${this.defaults.weightDecay})`;
  }
}
