/**
 * Loss Configuration (Data-First)
 *
 * Loss functions as a discriminated union on `kind`.
 * Factory functions produce pure data objects (serializable, loggable).
 * `resolveLoss()` maps configs to compute functions at training time.
 */

import type { Tensor } from '@ts-torch/core'
import { crossEntropyLoss, mseLoss, nllLoss } from '@ts-torch/optim'

/**
 * Loss function signature
 */
export type LossFn = (predictions: Tensor, targets: Tensor) => Tensor

/**
 * Loss configuration - discriminated union on `kind`
 */
export type LossConfig =
  | { kind: 'crossEntropy'; labelSmoothing?: number }
  | { kind: 'mse' }
  | { kind: 'nll' }
  | { kind: 'custom'; name: string; fn: LossFn }

/**
 * Resolve a LossConfig to its compute function.
 * Throws on unrecognized kind — fail fast at construction, not during training.
 */
export function resolveLoss(config: LossConfig): LossFn {
  switch (config.kind) {
    case 'crossEntropy':
      return crossEntropyLoss as unknown as LossFn
    case 'mse':
      return mseLoss as unknown as LossFn
    case 'nll':
      return nllLoss as unknown as LossFn
    case 'custom':
      return config.fn
    default:
      throw new Error(`Unknown loss kind: ${(config as any).kind}`)
  }
}

/**
 * Loss factory namespace
 *
 * @example
 * ```ts
 * import { loss } from '@ts-torch/train'
 *
 * loss.crossEntropy()
 * loss.crossEntropy({ labelSmoothing: 0.1 })
 * loss.mse()
 * loss.nll()
 * loss.custom('myLoss', (pred, target) => ...)
 * ```
 */
export const loss = {
  crossEntropy(opts?: { labelSmoothing?: number }): LossConfig {
    return { kind: 'crossEntropy', ...opts }
  },
  mse(): LossConfig {
    return { kind: 'mse' }
  },
  nll(): LossConfig {
    return { kind: 'nll' }
  },
  custom(name: string, fn: LossFn): LossConfig {
    return { kind: 'custom', name, fn }
  },
}
