/**
 * Adam optimizer
 */

import type { Tensor } from '@ts-torch/core'
import { Optimizer, type ParameterGroup, type OptimizerOptions } from './optimizer.js'

export interface AdamOptions extends OptimizerOptions {
  lr: number
  betas?: [number, number]
  eps?: number
  weightDecay?: number
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

  // State for each parameter: { step, m (first moment), v (second moment) }
  private adamState = new Map<Tensor, { step: number; m: Tensor; v: Tensor }>()

  constructor(params: Tensor[] | ParameterGroup[], options: AdamOptions) {
    super(params, {
      lr: options.lr,
      betas: options.betas ?? [0.9, 0.999],
      eps: options.eps ?? 1e-8,
      weightDecay: options.weightDecay ?? 0,
    })
  }

  step(): void {
    const lr = this.defaults.lr
    const [beta1, beta2] = this.defaults.betas!
    const eps = this.defaults.eps!
    const weightDecay = this.defaults.weightDecay ?? 0

    for (const group of this.paramGroups) {
      for (const param of group.params) {
        const grad = (param as any).grad as Tensor | null
        if (!grad) continue

        // Get or initialize state
        let state = this.adamState.get(param)
        if (!state) {
          // Initialize moment estimates to zeros (same shape as param)
          // Escape from scope since we store them across epochs
          const m = (grad as any).mulScalar(0) as Tensor
          const v = (grad as any).mulScalar(0) as Tensor
          if ('escape' in m) (m as any).escape()
          if ('escape' in v) (v as any).escape()
          state = { step: 0, m, v }
          this.adamState.set(param, state)
        }

        state.step++

        // Apply weight decay to gradient if needed
        let g = grad
        if (weightDecay !== 0) {
          g = (grad as any).add((param as any).mulScalar(weightDecay)) as Tensor
        }

        // Update biased first moment: m = beta1 * m + (1 - beta1) * g
        const old_m = state.m
        const m_scaled = (old_m as any).mulScalar(beta1) as Tensor
        const g_scaled = (g as any).mulScalar(1 - beta1) as Tensor
        const new_m = (m_scaled as any).add(g_scaled) as Tensor
        if ('escape' in new_m) (new_m as any).escape()
        if ('free' in old_m) (old_m as any).free()
        state.m = new_m

        // Update biased second moment: v = beta2 * v + (1 - beta2) * g^2
        const old_v = state.v
        const v_scaled = (old_v as any).mulScalar(beta2) as Tensor
        const g_sq = (g as any).mul(g) as Tensor
        const g_sq_scaled = (g_sq as any).mulScalar(1 - beta2) as Tensor
        const new_v = (v_scaled as any).add(g_sq_scaled) as Tensor
        if ('escape' in new_v) (new_v as any).escape()
        if ('free' in old_v) (old_v as any).free()
        state.v = new_v

        // Bias correction
        const bc1 = 1 - Math.pow(beta1, state.step)
        const bc2 = 1 - Math.pow(beta2, state.step)

        // Compute update: lr * m_hat / (sqrt(v_hat) + eps)
        // m_hat = m / bc1, v_hat = v / bc2
        const m_hat = (state.m as any).divScalar(bc1) as Tensor
        const v_hat = (state.v as any).divScalar(bc2) as Tensor
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
