/**
 * OptimizerTensor - Safe in-place operations for optimizer updates (Phase 4)
 *
 * This module provides a wrapper around tensors that exposes in-place operations
 * specifically designed for use within optimizer.step() calls. These operations
 * bypass autograd using .data() which is ONLY safe in the optimizer context.
 *
 * WARNING: Using these operations outside of optimizer.step() can cause
 * silent gradient corruption. The operations are intentionally fenced behind
 * this API to prevent misuse.
 */

import type { Tensor, Shape, DType, DeviceType } from '@ts-torch/core'
import { getLib, withErrorFast, type Pointer } from '@ts-torch/core'

/**
 * Wrapper for safe in-place tensor operations in optimizer context.
 *
 * This class wraps a parameter tensor and provides in-place update methods
 * that bypass autograd. ONLY use within optimizer.step() implementations.
 *
 * @template S - Tensor shape
 * @template D - Data type
 * @template Dev - Device type
 *
 * @example
 * ```ts
 * class SGD extends Optimizer {
 *   step() {
 *     for (const param of this.params) {
 *       const grad = param.grad;
 *       if (grad) {
 *         const optParam = new OptimizerTensor(param);
 *         // Safe in-place update: param.data -= lr * grad
 *         optParam.addInplace(grad, -this.lr);
 *       }
 *     }
 *   }
 * }
 * ```
 */
export class OptimizerTensor<
  S extends Shape = Shape,
  D extends DType<string> = DType<'float32'>,
  Dev extends DeviceType = DeviceType,
> {
  /**
   * The underlying tensor handle
   * @internal
   */
  private readonly _handle: Pointer

  /**
   * Reference to the original tensor for shape/dtype access
   */
  readonly tensor: Tensor<S, D, Dev>

  /**
   * Create an OptimizerTensor wrapper around a parameter tensor.
   *
   * @param tensor - The parameter tensor to wrap
   */
  constructor(tensor: Tensor<S, D, Dev>) {
    this.tensor = tensor
    this._handle = (tensor as any)._handle
  }

  /**
   * Get the native handle
   * @internal
   */
  get handle(): Pointer {
    return this._handle
  }

  /**
   * In-place addition: param.data += alpha * other
   *
   * Uses .data() to bypass autograd. ONLY safe in optimizer.step() context.
   *
   * @param other - Tensor to add
   * @param alpha - Scaling factor (default: 1.0)
   * @returns this for chaining
   *
   * @example
   * ```ts
   * // SGD update: param -= lr * grad
   * optParam.addInplace(grad, -learningRate);
   *
   * // Momentum update: velocity = momentum * velocity + grad
   * optVelocity.mulScalarInplace(momentum);
   * optVelocity.addInplace(grad, 1.0);
   * ```
   */
  addInplace(other: Tensor<S, D, Dev>, alpha: number = 1.0): this {
    const lib = getLib()
    withErrorFast((err) =>
      lib.ts_tensor_optim_add_(this._handle, (other as any)._handle, alpha, err),
    )
    return this
  }

  /**
   * In-place scalar multiplication: param.data *= scalar
   *
   * Uses .data() to bypass autograd. ONLY safe in optimizer.step() context.
   *
   * @param scalar - Scalar to multiply with
   * @returns this for chaining
   *
   * @example
   * ```ts
   * // Weight decay: param *= (1 - lr * weight_decay)
   * optParam.mulScalarInplace(1 - lr * weightDecay);
   * ```
   */
  mulScalarInplace(scalar: number): this {
    const lib = getLib()
    withErrorFast((err) => lib.ts_tensor_mul_scalar_(this._handle, scalar, err))
    return this
  }

  /**
   * Zero the gradient of this parameter
   *
   * @returns this for chaining
   */
  zeroGrad(): this {
    const lib = getLib()
    withErrorFast((err) => lib.ts_tensor_zero_grad_(this._handle, err))
    return this
  }

  /**
   * Get the gradient tensor (if it exists)
   */
  get grad(): Tensor<S, D, Dev> | null {
    return this.tensor.grad as Tensor<S, D, Dev> | null
  }

  /**
   * Get tensor shape
   */
  get shape(): S {
    return this.tensor.shape
  }

  /**
   * Get tensor dtype
   */
  get dtype(): D {
    return this.tensor.dtype
  }
}

/**
 * Helper to wrap multiple parameter tensors for batch optimizer updates.
 *
 * @param params - Array of parameter tensors
 * @returns Array of OptimizerTensor wrappers
 *
 * @example
 * ```ts
 * const optParams = wrapForOptim(model.parameters());
 * for (const optParam of optParams) {
 *   if (optParam.grad) {
 *     optParam.addInplace(optParam.grad, -lr);
 *   }
 * }
 * ```
 */
export function wrapForOptim<S extends Shape, D extends DType<string>, Dev extends DeviceType>(
  params: Tensor<S, D, Dev>[],
): OptimizerTensor<S, D, Dev>[] {
  return params.map((p) => new OptimizerTensor(p))
}
