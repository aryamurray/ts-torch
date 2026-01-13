/**
 * Adam optimizer
 */

import { type Tensor, validateAdamParams } from '@ts-torch/core'
import { Optimizer, type ParameterGroup, type OptimizerOptions } from './optimizer.js'

export interface AdamOptions extends OptimizerOptions {
  lr: number
  betas?: [number, number]
  eps?: number
  weightDecay?: number
  amsgrad?: boolean
}

/**
 * Adam optimizer (Adaptive Moment Estimation)
 *
 * @example
 * ```typescript
 * const optimizer = new Adam(model.parameters(), { lr: 0.001 });
 * ```
 */
export class Adam extends Optimizer {
  declare defaults: AdamOptions

  constructor(params: Tensor[] | ParameterGroup[], options: AdamOptions) {
    const betas = options.betas ?? [0.9, 0.999]
    const eps = options.eps ?? 1e-8
    const weightDecay = options.weightDecay ?? 0
    const amsgrad = options.amsgrad ?? false

    validateAdamParams({
      lr: options.lr,
      betas,
      eps,
      weightDecay,
    })

    super(params, {
      lr: options.lr,
      betas,
      eps,
      weightDecay,
      amsgrad,
    })
  }

  step(): void {
    const lr = this.defaults.lr
    const [beta1, beta2] = this.defaults.betas!
    const eps = this.defaults.eps!
    const weightDecay = this.defaults.weightDecay ?? 0
    const amsgrad = this.defaults.amsgrad ?? false

    for (const group of this.paramGroups) {
      for (const param of group.params) {
        const grad = (param as any).grad as Tensor | null
        if (!grad) continue

        // Get or initialize state
        let state = this.state.get(param) as
          | {
              step: number
              exp_avg: Tensor
              exp_avg_sq: Tensor
              max_exp_avg_sq?: Tensor
            }
          | undefined

        if (!state) {
          // Initialize moment estimates to zeros (same shape as param)
          // Escape from scope since we store them across epochs
          const exp_avg = (grad as any).mulScalar(0) as Tensor
          const exp_avg_sq = (grad as any).mulScalar(0) as Tensor
          if ('escape' in exp_avg) (exp_avg as any).escape()
          if ('escape' in exp_avg_sq) (exp_avg_sq as any).escape()

          state = { step: 0, exp_avg, exp_avg_sq }

          // Initialize max_exp_avg_sq for AMSGrad
          if (amsgrad) {
            const max_exp_avg_sq = (grad as any).mulScalar(0) as Tensor
            if ('escape' in max_exp_avg_sq) (max_exp_avg_sq as any).escape()
            state.max_exp_avg_sq = max_exp_avg_sq
          }

          this.state.set(param, state as any)
        }

        state.step++

        // Apply weight decay to gradient if needed
        let g = grad
        if (weightDecay !== 0) {
          g = (grad as any).add((param as any).mulScalar(weightDecay)) as Tensor
        }

        // Update biased first moment: exp_avg = beta1 * exp_avg + (1 - beta1) * g
        const old_exp_avg = state.exp_avg
        const m_scaled = (old_exp_avg as any).mulScalar(beta1) as Tensor
        const g_scaled = (g as any).mulScalar(1 - beta1) as Tensor
        const new_exp_avg = (m_scaled as any).add(g_scaled) as Tensor
        if ('escape' in new_exp_avg) (new_exp_avg as any).escape()
        if ('free' in old_exp_avg) (old_exp_avg as any).free()
        state.exp_avg = new_exp_avg

        // Update biased second moment: exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * g^2
        const old_exp_avg_sq = state.exp_avg_sq
        const v_scaled = (old_exp_avg_sq as any).mulScalar(beta2) as Tensor
        const g_sq = (g as any).mul(g) as Tensor
        const g_sq_scaled = (g_sq as any).mulScalar(1 - beta2) as Tensor
        const new_exp_avg_sq = (v_scaled as any).add(g_sq_scaled) as Tensor
        if ('escape' in new_exp_avg_sq) (new_exp_avg_sq as any).escape()
        if ('free' in old_exp_avg_sq) (old_exp_avg_sq as any).free()
        state.exp_avg_sq = new_exp_avg_sq

        // AMSGrad: maintain max of all exp_avg_sq
        let denom_sq = state.exp_avg_sq
        if (amsgrad) {
          // Compute element-wise max(old_max, exp_avg_sq) using: max(a,b) = b + ReLU(a - b)
          const old_max = state.max_exp_avg_sq!
          const diff = (old_max as any).sub(state.exp_avg_sq) as Tensor
          const relu_diff = (diff as any).relu() as Tensor
          const new_max = (state.exp_avg_sq as any).add(relu_diff) as Tensor
          if ('escape' in new_max) (new_max as any).escape()
          if ('free' in old_max) (old_max as any).free()
          state.max_exp_avg_sq = new_max
          denom_sq = new_max
        }

        // Bias correction
        const bc1 = 1 - Math.pow(beta1, state.step)
        const bc2 = 1 - Math.pow(beta2, state.step)

        // Compute update: lr * m_hat / (sqrt(v_hat) + eps)
        // m_hat = exp_avg / bc1, v_hat = denom_sq / bc2
        const m_hat = (state.exp_avg as any).divScalar(bc1) as Tensor
        const v_hat = (denom_sq as any).divScalar(bc2) as Tensor
        const v_sqrt = (v_hat as any).sqrt() as Tensor
        const denom = (v_sqrt as any).addScalar(eps) as Tensor
        const update = (m_hat as any).div(denom) as Tensor

        // param -= lr * update (in-place)
        ;(param as any).addScaledInplace(update, -lr)
      }
    }
  }

  override zeroGrad(): void {
    for (const group of this.paramGroups) {
      for (const param of group.params) {
        if ('zeroGrad' in param && typeof (param as any).zeroGrad === 'function') {
          ;(param as any).zeroGrad()
        }
      }
    }
  }

  override toString(): string {
    return `Adam(lr=${this.defaults.lr}, betas=[${this.defaults.betas}], eps=${this.defaults.eps})`
  }
}
