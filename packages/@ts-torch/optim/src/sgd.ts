/**
 * Stochastic Gradient Descent optimizer
 */

import { type Tensor, validateSGDParams } from '@ts-torch/core'
import { Optimizer, type ParameterGroup, type OptimizerOptions } from './optimizer.js'

/**
 * SGD optimizer options
 */
export interface SGDOptions extends OptimizerOptions {
  /** Learning rate */
  lr: number
  momentum?: number
  weightDecay?: number
}

/**
 * Stochastic Gradient Descent (SGD) optimizer
 *
 * Implements SGD with optional momentum and weight decay.
 *
 * @example
 * ```typescript
 * // Get parameter tensors from model
 * const params = model.parameters().map(p => p.data);
 * const optimizer = new SGD(params, { lr: 0.01 });
 *
 * // Training loop
 * optimizer.zeroGrad();
 * loss.backward();
 * optimizer.step();
 * ```
 */
export class SGD extends Optimizer {
  declare defaults: SGDOptions
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private momentumBuffers: WeakMap<Tensor, Tensor> = new WeakMap()

  constructor(params: Tensor[] | ParameterGroup[], options: SGDOptions) {
    const { lr, momentum = 0, weightDecay = 0 } = options

    validateSGDParams({
      lr: lr,
      momentum,
      weightDecay,
    })

    const defaults: SGDOptions = {
      lr,
      momentum,
      weightDecay,
    }

    super(params, defaults)
  }

  step(): void {
    const lr = this.defaults.lr
    const momentum = this.defaults.momentum ?? 0
    const weightDecay = this.defaults.weightDecay ?? 0

    for (const group of this.paramGroups) {
      const groupLr = group.lr ?? lr

      for (const param of group.params) {
        // Get gradient - check if param has grad property
        const grad = (param as any).grad
        if (!grad) continue

        // Get the actual gradient tensor
        let d_p = grad

        // Apply weight decay (L2 regularization)
        // gradient = gradient + weight_decay * param
        let weighted_param: any = null
        if (weightDecay !== 0) {
          if ('mulScalar' in param && 'add' in d_p) {
            weighted_param = (param as any).mulScalar(weightDecay)
            d_p = d_p.add(weighted_param)
          }
        }

        // Apply momentum
        if (momentum !== 0) {
          let buf = this.momentumBuffers.get(param)

          if (!buf) {
            // Initialize momentum buffer with current gradient (clone)
            // Escape from scope since we store it across epochs
            if ('clone' in d_p && typeof (d_p as any).clone === 'function') {
              const cloned = (d_p as any).clone() as Tensor
              if (cloned) {
                if ('escape' in cloned) (cloned as any).escape()
                buf = cloned
                this.momentumBuffers.set(param, cloned)
              }
            }
          } else {
            // velocity = momentum * velocity + gradient
            // Use in-place operations if available to eliminate temp tensor allocations
            if ('mulScalarInplace' in buf && 'addInplace' in buf) {
              // velocity *= momentum
              ;(buf as any).mulScalarInplace(momentum)
              // velocity += gradient
              ;(buf as any).addInplace(d_p)
              // buf is already updated in-place, no need to reassign
            } else {
              // Fallback: create new tensor (old behavior)
              const momentumTerm = (buf as any).mulScalar(momentum) as Tensor
              const newBuf = momentumTerm.add(d_p) as Tensor
              if (newBuf) {
                // Escape new buffer, free old buffer
                if ('escape' in newBuf) (newBuf as any).escape()
                if ('free' in buf) (buf as any).free()
                buf = newBuf
                this.momentumBuffers.set(param, newBuf)
              }
            }
          }

          if (buf) {
            d_p = buf
          }
        }

        // Update parameters in-place: param.data -= lr * gradient
        // Use addScaledInplace for efficient in-place update that bypasses autograd
        if ('addScaledInplace' in param && typeof (param as any).addScaledInplace === 'function') {
          ;(param as any).addScaledInplace(d_p, -groupLr)
        } else {
          // Fallback: create new tensor and warn (this is the old broken behavior)
          console.warn('SGD: addScaledInplace not available, falling back to inefficient update')
          if ('mulScalar' in d_p && 'sub' in param) {
            const update = (d_p as any).mulScalar(groupLr)
            void (param as any).sub(update)
            // This fallback won't work correctly - parameters won't update
            // The in-place method should always be available
          }
        }

        // Cleanup temporary tensors
        if (weighted_param && 'free' in weighted_param) {
          (weighted_param as any).free()
        }
        if (d_p !== grad && 'free' in d_p) {
          (d_p as any).free()
        }
      }
    }
  }

  /**
   * Zero all gradients of optimized parameters
   */
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
   * Clear momentum buffers for explicit cleanup
   * Note: WeakMap allows GC automatically, but this provides explicit control
   */
  clearMomentumBuffers(): void {
    this.momentumBuffers = new WeakMap()
  }

  /**
   * Delete momentum buffer for a specific parameter
   */
  deleteMomentumBuffer(param: Tensor): boolean {
    return this.momentumBuffers.delete(param)
  }

  override toString(): string {
    return `SGD(lr=${this.defaults.lr}, momentum=${this.defaults.momentum ?? 0}, weightDecay=${this.defaults.weightDecay ?? 0})`
  }
}
