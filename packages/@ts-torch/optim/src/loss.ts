/**
 * Loss functions for training neural networks
 */

import type { Tensor, Shape, DType } from '@ts-torch/core'

/**
 * Cross entropy loss for multi-class classification
 *
 * @param logits - Raw prediction scores [Batch, Classes]
 * @param targets - Ground truth class indices [Batch]
 * @returns Scalar loss tensor
 */
export function crossEntropyLoss<B extends number, C extends number, D extends DType<string>>(
  logits: Tensor<readonly [B, C], D>,
  targets: Tensor<readonly [B], D>,
): Tensor<readonly [], D> {
  return (logits as any).crossEntropyLoss(targets)
}

/**
 * Mean squared error loss for regression
 *
 * @param input - Predicted values
 * @param target - Ground truth values
 * @returns Scalar loss tensor
 */
export function mseLoss<S extends Shape, D extends DType<string>>(
  input: Tensor<S, D>,
  target: Tensor<S, D>,
): Tensor<readonly [], D> {
  return (input as any).mseLoss(target)
}

/**
 * Negative log likelihood loss
 *
 * @param logProbs - Log probabilities [Batch, Classes]
 * @param targets - Ground truth class indices [Batch]
 * @returns Scalar loss tensor
 */
export function nllLoss<B extends number, C extends number, D extends DType<string>>(
  logProbs: Tensor<readonly [B, C], D>,
  targets: Tensor<readonly [B], D>,
): Tensor<readonly [], D> {
  return (logProbs as any).nllLoss(targets)
}
