/**
 * RMSprop optimizer
 */

import { type Tensor, validateRMSpropParams } from '@ts-torch/core'
import { Optimizer, type ParameterGroup, type OptimizerOptions } from './optimizer.js'

/**
 * RMSprop optimizer options
 */
export interface RMSpropOptions extends OptimizerOptions {
  /** Learning rate */
  lr: number
  alpha?: number
  eps?: number
  weightDecay?: number
  momentum?: number
  centered?: boolean
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
 * The algorithm:
 *   v_t = alpha * v_{t-1} + (1 - alpha) * g_t^2
 *   If centered:
 *     g_avg_t = alpha * g_avg_{t-1} + (1 - alpha) * g_t
 *     v_hat_t = v_t - g_avg_t^2
 *     theta_t = theta_{t-1} - lr * g_t / (sqrt(v_hat_t) + eps)
 *   Else:
 *     theta_t = theta_{t-1} - lr * g_t / (sqrt(v_t) + eps)
 *
 * If momentum > 0:
 *   buf_t = momentum * buf_{t-1} + g_t / denom
 *   theta_t = theta_{t-1} - lr * buf_t
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
 *
 * // Training loop
 * optimizer.zeroGrad();
 * loss.backward();
 * optimizer.step();
 * ```
 */
export class RMSprop extends Optimizer {
  declare defaults: RMSpropOptions

  // State for each parameter: { square_avg, grad_avg?, momentum_buffer? }
  private rmspropState = new Map<
    Tensor,
    {
      step: number
      square_avg: Tensor
      grad_avg?: Tensor
      momentum_buffer?: Tensor
    }
  >()

  constructor(params: Tensor[] | ParameterGroup[], options: RMSpropOptions) {
    const { lr, alpha = 0.99, eps = 1e-8, weightDecay = 0, momentum = 0, centered = false } = options

    validateRMSpropParams({
      lr: lr,
      alpha,
      eps,
      weightDecay,
      momentum,
    })

    const defaults: RMSpropOptions = {
      lr,
      alpha,
      eps,
      weightDecay,
      momentum,
      centered,
    }

    super(params, defaults)
  }

  step(): void {
    const lr = this.defaults.lr
    const alpha = this.defaults.alpha!
    const eps = this.defaults.eps!
    const weightDecay = this.defaults.weightDecay ?? 0
    const momentum = this.defaults.momentum ?? 0
    const centered = this.defaults.centered ?? false

    for (const group of this.paramGroups) {
      for (const param of group.params) {
        const grad = (param as any).grad as Tensor | null
        if (!grad) continue

        // Get or initialize state
        let state = this.rmspropState.get(param)
        if (!state) {
          // Initialize square_avg to zeros (same shape as param)
          const square_avg = (grad as any).mulScalar(0) as Tensor
          if ('escape' in square_avg) (square_avg as any).escape()

          state = { step: 0, square_avg }

          // Initialize grad_avg if centered
          if (centered) {
            const grad_avg = (grad as any).mulScalar(0) as Tensor
            if ('escape' in grad_avg) (grad_avg as any).escape()
            state.grad_avg = grad_avg
          }

          this.rmspropState.set(param, state)
        }

        state.step++

        // Apply weight decay to gradient if needed
        let g = grad
        let weighted_param: Tensor | null = null
        if (weightDecay !== 0) {
          weighted_param = (param as any).mulScalar(weightDecay) as Tensor
          g = (grad as any).add(weighted_param) as Tensor
        }

        // Update square_avg: v = alpha * v + (1 - alpha) * g^2
        // Use in-place operations if available (following Adam pattern)
        if ('mulScalarInplace' in state.square_avg && 'addScaledInplace' in state.square_avg) {
          ;(state.square_avg as any).mulScalarInplace(alpha)
          const g_sq = (g as any).mul(g) as Tensor
          ;(state.square_avg as any).addScaledInplace(g_sq, 1 - alpha)
          if ('free' in g_sq) (g_sq as any).free()
        } else {
          // Fallback: create new tensor (old behavior)
          const v_scaled = (state.square_avg as any).mulScalar(alpha) as Tensor
          const g_sq = (g as any).mul(g) as Tensor
          const g_sq_scaled = (g_sq as any).mulScalar(1 - alpha) as Tensor
          const new_square_avg = (v_scaled as any).add(g_sq_scaled) as Tensor
          if ('escape' in new_square_avg) (new_square_avg as any).escape()
          if ('free' in state.square_avg) (state.square_avg as any).free()
          state.square_avg = new_square_avg
        }

        let avg: Tensor

        if (centered) {
          // Update grad_avg: g_avg = alpha * g_avg + (1 - alpha) * g
          // Use in-place operations if available
          if ('mulScalarInplace' in state.grad_avg! && 'addScaledInplace' in state.grad_avg!) {
            ;(state.grad_avg as any).mulScalarInplace(alpha)
            ;(state.grad_avg as any).addScaledInplace(g, 1 - alpha)
          } else {
            // Fallback: create new tensor (old behavior)
            const gavg_scaled = (state.grad_avg as any).mulScalar(alpha) as Tensor
            const g_scaled = (g as any).mulScalar(1 - alpha) as Tensor
            const new_grad_avg = (gavg_scaled as any).add(g_scaled) as Tensor
            if ('escape' in new_grad_avg) (new_grad_avg as any).escape()
            if ('free' in state.grad_avg) (state.grad_avg as any).free()
            state.grad_avg = new_grad_avg
          }

          // avg = sqrt(v - g_avg^2 + eps)
          // CRITICAL FIX: epsilon BEFORE sqrt to prevent NaN from sqrt(negative)
          const grad_avg_sq = (state.grad_avg as any).mul(state.grad_avg) as Tensor
          const v_centered = (state.square_avg as any).sub(grad_avg_sq) as Tensor
          const v_centered_eps = (v_centered as any).addScalar(eps) as Tensor
          avg = (v_centered_eps as any).sqrt() as Tensor
          // Note: Don't free intermediate tensors as avg may reference their data
          // Cleanup happens at end of iteration
        } else {
          // avg = sqrt(v + eps)
          // CRITICAL FIX: epsilon BEFORE sqrt to prevent NaN
          const v_eps = (state.square_avg as any).addScalar(eps) as Tensor
          avg = (v_eps as any).sqrt() as Tensor
          // Note: Don't free v_eps as avg may reference its data
          // Cleanup happens at end of iteration
        }

        if (momentum > 0) {
          // Update momentum buffer: buf = momentum * buf + g / avg
          const g_div_avg = (g as any).div(avg) as Tensor

          if (!state.momentum_buffer) {
            // Initialize momentum buffer
            const buf = (g_div_avg as any).clone() as Tensor
            if ('escape' in buf) (buf as any).escape()
            state.momentum_buffer = buf
          } else {
            // Use in-place operations if available
            if ('mulScalarInplace' in state.momentum_buffer && 'addInplace' in state.momentum_buffer) {
              ;(state.momentum_buffer as any).mulScalarInplace(momentum)
              ;(state.momentum_buffer as any).addInplace(g_div_avg)
            } else {
              // Fallback: create new tensor (old behavior)
              const buf_scaled = (state.momentum_buffer as any).mulScalar(momentum) as Tensor
              const new_buf = (buf_scaled as any).add(g_div_avg) as Tensor
              if ('escape' in new_buf) (new_buf as any).escape()
              if ('free' in state.momentum_buffer) (state.momentum_buffer as any).free()
              state.momentum_buffer = new_buf
            }
          }

          if ('free' in g_div_avg) (g_div_avg as any).free()

          // param -= lr * momentum_buffer
          ;(param as any).addScaledInplace(state.momentum_buffer, -lr)
        } else {
          // param -= lr * g / avg
          const update = (g as any).div(avg) as Tensor
          ;(param as any).addScaledInplace(update, -lr)
          if ('free' in update) (update as any).free()
        }

        // Cleanup temporary tensors (weighted_param is safe to free)
        if (weighted_param && 'free' in weighted_param) (weighted_param as any).free()
        // Note: Don't free g, g_div_avg, avg - they may be referenced or needed
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

  /**
   * Clear optimizer state for explicit cleanup
   */
  clearState(): void {
    // Free all stored tensors
    for (const state of this.rmspropState.values()) {
      if ('free' in state.square_avg) (state.square_avg as any).free()
      if (state.grad_avg && 'free' in state.grad_avg) (state.grad_avg as any).free()
      if (state.momentum_buffer && 'free' in state.momentum_buffer)
        (state.momentum_buffer as any).free()
    }
    this.rmspropState.clear()
  }

  override toString(): string {
    return `RMSprop(lr=${this.defaults.lr}, alpha=${this.defaults.alpha}, eps=${this.defaults.eps}, momentum=${this.defaults.momentum}, centered=${this.defaults.centered})`
  }
}
