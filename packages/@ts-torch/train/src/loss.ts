/**
 * Loss Configuration (Data-First)
 *
 * Loss functions as a discriminated union on `kind`.
 * Factory functions produce pure data objects (serializable, loggable).
 * `resolveLoss()` maps configs to compute functions at training time.
 */

import type { Tensor } from '@ts-torch/core'
import { crossEntropyLoss, mseLoss, nllLoss, bceLoss, bceWithLogitsLoss, l1Loss, smoothL1Loss, klDivLoss } from '@ts-torch/optim'

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
  | { kind: 'bce'; reduction?: 'none' | 'mean' | 'sum' }
  | { kind: 'bceWithLogits'; reduction?: 'none' | 'mean' | 'sum' }
  | { kind: 'l1'; reduction?: 'none' | 'mean' | 'sum' }
  | { kind: 'smoothL1'; reduction?: 'none' | 'mean' | 'sum'; beta?: number }
  | { kind: 'klDiv'; reduction?: 'none' | 'mean' | 'sum' | 'batchmean'; logTarget?: boolean }
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
    case 'bce':
      return ((pred, target) => bceLoss(pred, target, config.reduction)) as LossFn
    case 'bceWithLogits':
      return ((pred, target) => bceWithLogitsLoss(pred, target, config.reduction)) as LossFn
    case 'l1':
      return ((pred, target) => l1Loss(pred, target, config.reduction)) as LossFn
    case 'smoothL1':
      return ((pred, target) => smoothL1Loss(pred, target, config.reduction, config.beta)) as LossFn
    case 'klDiv':
      return ((pred, target) => klDivLoss(pred, target, config.reduction as any, config.logTarget)) as LossFn
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
  bce(opts?: { reduction?: 'none' | 'mean' | 'sum' }): LossConfig {
    return { kind: 'bce', ...opts }
  },
  bceWithLogits(opts?: { reduction?: 'none' | 'mean' | 'sum' }): LossConfig {
    return { kind: 'bceWithLogits', ...opts }
  },
  l1(opts?: { reduction?: 'none' | 'mean' | 'sum' }): LossConfig {
    return { kind: 'l1', ...opts }
  },
  smoothL1(opts?: { reduction?: 'none' | 'mean' | 'sum'; beta?: number }): LossConfig {
    return { kind: 'smoothL1', ...opts }
  },
  klDiv(opts?: { reduction?: 'none' | 'mean' | 'sum' | 'batchmean'; logTarget?: boolean }): LossConfig {
    return { kind: 'klDiv', ...opts }
  },
  custom(name: string, fn: LossFn): LossConfig {
    return { kind: 'custom', name, fn }
  },
}
