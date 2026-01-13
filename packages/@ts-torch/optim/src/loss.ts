/**
 * Loss functions for training neural networks
 *
 * These functions delegate to native Tensor methods for optimal performance.
 */

import type { Tensor, Shape, DType } from '@ts-torch/core'

/**
 * Cross entropy loss for multi-class classification
 *
 * Combines log_softmax and negative log likelihood loss in a single operation.
 * More numerically stable than computing them separately.
 *
 * @param logits - Raw prediction scores [Batch, Classes]
 * @param targets - Ground truth class indices [Batch] (int64)
 * @returns Scalar loss tensor
 *
 * @example
 * ```ts
 * const logits = torch.randn([32, 10])  // 32 samples, 10 classes
 * const targets = torch.tensor([...], [32], torch.int64)
 * const loss = crossEntropyLoss(logits, targets)
 * loss.backward()
 * ```
 */
export function crossEntropyLoss<B extends number, C extends number, D extends DType<string>>(
  logits: Tensor<readonly [B, C], D>,
  targets: Tensor<readonly [B], DType<'int64'>>,
): Tensor<readonly [], D> {
  return (logits as any).crossEntropyLoss(targets)
}

/**
 * Mean squared error loss for regression
 *
 * Computes the mean of squared differences between predictions and targets.
 *
 * @param input - Predicted values
 * @param target - Ground truth values (same shape as input)
 * @returns Scalar loss tensor
 *
 * @example
 * ```ts
 * const pred = model.forward(x)
 * const loss = mseLoss(pred, target)
 * loss.backward()
 * ```
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
 * Expects log-probabilities as input (output of log_softmax).
 * Use crossEntropyLoss if you have raw logits.
 *
 * @param logProbs - Log probabilities [Batch, Classes] (output of log_softmax)
 * @param targets - Ground truth class indices [Batch] (int64)
 * @returns Scalar loss tensor
 *
 * @example
 * ```ts
 * const logProbs = logits.logSoftmax(1)
 * const loss = nllLoss(logProbs, targets)
 * ```
 */
export function nllLoss<B extends number, C extends number, D extends DType<string>>(
  logProbs: Tensor<readonly [B, C], D>,
  targets: Tensor<readonly [B], DType<'int64'>>,
): Tensor<readonly [], D> {
  return (logProbs as any).nllLoss(targets)
}
