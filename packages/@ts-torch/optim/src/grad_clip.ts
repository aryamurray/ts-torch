/**
 * Gradient clipping utilities for training stability
 *
 * These utilities help prevent exploding gradients, especially important
 * for training RNNs and transformers.
 */

import type { Tensor, Shape, DType } from '@ts-torch/core'

/**
 * Clips gradient norm of an iterable of parameters
 *
 * The norm is computed over all gradients together, as if they were
 * concatenated into a single vector.
 *
 * @param parameters - Iterable of tensors with gradients
 * @param maxNorm - Maximum allowed norm value
 * @param normType - Type of p-norm (default: 2.0 for L2 norm)
 * @param errorIfNonfinite - Throw error if norm is inf or nan (default: false)
 * @returns Total norm of the gradients (before clipping)
 *
 * @example
 * ```ts
 * const optimizer = new Adam(model.parameters());
 *
 * // Training loop
 * optimizer.zeroGrad();
 * const loss = model.forward(x).loss();
 * loss.backward();
 *
 * // Clip gradients to max norm of 1.0
 * const totalNorm = clipGradNorm(model.parameters(), 1.0);
 * console.log(`Gradient norm: ${totalNorm}`);
 *
 * optimizer.step();
 * ```
 */
export function clipGradNorm<D extends DType<string>>(
  parameters: Iterable<Tensor<Shape, D>>,
  maxNorm: number,
  normType: number = 2.0,
  errorIfNonfinite: boolean = false,
): number {
  const paramsList = Array.from(parameters)

  if (paramsList.length === 0) {
    return 0.0
  }

  // Filter to parameters with gradients
  const grads: Tensor<Shape, D>[] = []
  for (const param of paramsList) {
    const grad = (param as any).grad
    if (grad !== null && grad !== undefined) {
      grads.push(grad)
    }
  }

  if (grads.length === 0) {
    return 0.0
  }

  let totalNorm: number

  if (normType === Infinity) {
    // Infinity norm: max of absolute values
    // Use sum of squares and compare to running max (approximation without abs/max ops)
    let maxVal = 0
    for (const grad of grads) {
      // Get all values and compute max abs manually
      const arr = (grad as any).toArray() as number[] | Float32Array
      for (const v of arr) {
        const absV = Math.abs(v)
        if (absV > maxVal) {
          maxVal = absV
        }
      }
    }
    totalNorm = maxVal
  } else if (normType === 0) {
    // L0 norm: count of non-zero elements
    let count = 0
    for (const grad of grads) {
      const arr = (grad as any).toArray() as number[] | Float32Array
      for (const v of arr) {
        if (v !== 0) {
          count++
        }
      }
    }
    totalNorm = count
  } else if (normType === 2) {
    // L2 norm (special case, most common): sqrt(sum(x^2))
    let sumSq = 0
    for (const grad of grads) {
      const sq = (grad as any).mul(grad)
      const gradSum = sq.sum()
      sumSq += (gradSum as any).item() as number
    }
    totalNorm = Math.sqrt(sumSq)
  } else {
    // Lp norm: (sum(|x|^p))^(1/p)
    // Use toArray to compute manually since abs/pow not available
    let sumPowNorm = 0
    for (const grad of grads) {
      const arr = (grad as any).toArray() as number[] | Float32Array
      for (const v of arr) {
        sumPowNorm += Math.pow(Math.abs(v), normType)
      }
    }
    totalNorm = Math.pow(sumPowNorm, 1.0 / normType)
  }

  // Check for non-finite values
  if (errorIfNonfinite && (!Number.isFinite(totalNorm) || Number.isNaN(totalNorm))) {
    throw new Error(
      `The total norm of order ${normType} for gradients is non-finite (${totalNorm}). ` +
        'Consider using error_if_nonfinite=false to skip this check.',
    )
  }

  // Compute clipping coefficient
  const clipCoef = maxNorm / (totalNorm + 1e-6)

  // Only clip if norm exceeds max
  if (clipCoef < 1) {
    for (const param of paramsList) {
      const grad = (param as any).grad
      if (grad !== null && grad !== undefined) {
        // Scale gradient in-place
        if (typeof (grad as any).mulScalarInplace === 'function') {
          ;(grad as any).mulScalarInplace(clipCoef)
        } else {
          ;(param as any)._gradCache = grad.mulScalar(clipCoef)
        }
      }
    }
  }

  return totalNorm
}

/**
 * Alias for clipGradNorm with underscore suffix (PyTorch convention)
 */
export const clipGradNorm_ = clipGradNorm

/**
 * Clips gradient of an iterable of parameters at specified value
 *
 * Gradients are modified in-place.
 *
 * @param parameters - Iterable of tensors with gradients
 * @param clipValue - Maximum allowed absolute value for gradients
 *
 * @example
 * ```ts
 * optimizer.zeroGrad();
 * loss.backward();
 *
 * // Clip gradients to [-0.5, 0.5]
 * clipGradValue(model.parameters(), 0.5);
 *
 * optimizer.step();
 * ```
 */
export function clipGradValue<D extends DType<string>>(
  parameters: Iterable<Tensor<Shape, D>>,
  clipValue: number,
): void {
  if (clipValue <= 0) {
    throw new Error(`clip_value must be positive, got ${clipValue}`)
  }

  for (const param of parameters) {
    const grad = (param as any).grad
    if (grad !== null && grad !== undefined) {
      // Clamp gradient to [-clipValue, clipValue]
      const clampedGrad = grad.clamp(-clipValue, clipValue)
      // Update gradient
      ;(param as any)._gradCache = clampedGrad
    }
  }
}

/**
 * Alias for clipGradValue with underscore suffix (PyTorch convention)
 */
export const clipGradValue_ = clipGradValue

/**
 * Get the total gradient norm of parameters
 *
 * Useful for monitoring gradient health during training.
 *
 * @param parameters - Iterable of tensors with gradients
 * @param normType - Type of p-norm (default: 2.0 for L2 norm)
 * @returns Total norm of the gradients
 *
 * @example
 * ```ts
 * loss.backward();
 * const gradNorm = getGradNorm(model.parameters());
 * console.log(`Gradient L2 norm: ${gradNorm}`);
 * ```
 */
export function getGradNorm<D extends DType<string>>(
  parameters: Iterable<Tensor<Shape, D>>,
  normType: number = 2.0,
): number {
  const grads: Tensor<Shape, D>[] = []

  for (const param of parameters) {
    const grad = (param as any).grad
    if (grad !== null && grad !== undefined) {
      grads.push(grad)
    }
  }

  if (grads.length === 0) {
    return 0.0
  }

  if (normType === Infinity) {
    // Infinity norm: max of absolute values
    let maxVal = 0
    for (const grad of grads) {
      const arr = (grad as any).toArray() as number[] | Float32Array
      for (const v of arr) {
        const absV = Math.abs(v)
        if (absV > maxVal) {
          maxVal = absV
        }
      }
    }
    return maxVal
  } else if (normType === 2) {
    // L2 norm (most common): sqrt(sum(x^2))
    let sumSq = 0
    for (const grad of grads) {
      const sq = (grad as any).mul(grad)
      const gradSum = sq.sum()
      sumSq += (gradSum as any).item() as number
    }
    return Math.sqrt(sumSq)
  } else {
    // Lp norm: (sum(|x|^p))^(1/p)
    let sumPowNorm = 0
    for (const grad of grads) {
      const arr = (grad as any).toArray() as number[] | Float32Array
      for (const v of arr) {
        sumPowNorm += Math.pow(Math.abs(v), normType)
      }
    }
    return Math.pow(sumPowNorm, 1.0 / normType)
  }
}

/**
 * Check if any gradient is NaN or Inf
 *
 * Useful for debugging gradient issues.
 *
 * @param parameters - Iterable of tensors with gradients
 * @returns Object with hasNaN, hasInf, and names of problematic parameters
 *
 * @example
 * ```ts
 * loss.backward();
 * const { hasNaN, hasInf } = checkGradHealth(model.namedParameters());
 * if (hasNaN || hasInf) {
 *   console.warn('Gradient health check failed!');
 * }
 * ```
 */
export function checkGradHealth<D extends DType<string>>(
  parameters: Iterable<[string, Tensor<Shape, D>]>,
): { hasNaN: boolean; hasInf: boolean; nanParams: string[]; infParams: string[] } {
  let hasNaN = false
  let hasInf = false
  const nanParams: string[] = []
  const infParams: string[] = []

  for (const [name, param] of parameters) {
    const grad = (param as any).grad
    if (grad !== null && grad !== undefined) {
      // Check for NaN and Inf by examining all values
      const arr = (grad as any).toArray() as number[] | Float32Array
      let foundNaN = false
      let foundInf = false

      for (const v of arr) {
        if (Number.isNaN(v)) {
          foundNaN = true
        } else if (!Number.isFinite(v)) {
          foundInf = true
        }
        // Early exit if both found
        if (foundNaN && foundInf) break
      }

      if (foundNaN) {
        hasNaN = true
        nanParams.push(name)
      }
      if (foundInf) {
        hasInf = true
        infParams.push(name)
      }
    }
  }

  return { hasNaN, hasInf, nanParams, infParams }
}
