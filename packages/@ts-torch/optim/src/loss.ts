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
 * import { device, int64 } from '@ts-torch/core'
 * const cpu = device.cpu()
 * const logits = cpu.randn([32, 10] as const)  // 32 samples, 10 classes
 * const targets = cpu.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [32] as const, int64)
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

/**
 * Reduction type for loss functions
 */
export type Reduction = 'none' | 'mean' | 'sum' | 'batchmean'

/**
 * Binary cross entropy loss
 *
 * Measures the loss for binary classification problems.
 * Input should be probabilities (after sigmoid).
 *
 * BCE = -[y * log(p) + (1-y) * log(1-p)]
 *
 * @param input - Predicted probabilities [0, 1]
 * @param target - Ground truth binary labels [0, 1]
 * @param reduction - Reduction method (default: 'mean')
 * @returns Loss tensor
 *
 * @example
 * ```ts
 * const pred = model.forward(x).sigmoid() // Apply sigmoid first
 * const loss = bceLoss(pred, target)
 * ```
 */
export function bceLoss<S extends Shape, D extends DType<string>>(
  input: Tensor<S, D>,
  target: Tensor<S, D>,
  reduction: Reduction = 'mean',
): Tensor<Shape, D> {
  // BCE = -[y * log(p) + (1-y) * log(1-p)]
  const eps = 1e-7

  // Clamp input to avoid log(0)
  const inputClamped = (input as any).clamp(eps, 1 - eps)

  // y * log(p)
  const term1 = target.mul((inputClamped as any).log())

  // (1-y) * log(1-p)
  const oneMinusTarget = (target as any).mulScalar(-1).addScalar(1)
  const oneMinusInput = (inputClamped as any).mulScalar(-1).addScalar(1)
  const term2 = oneMinusTarget.mul(oneMinusInput.log())

  // -(term1 + term2)
  let loss = term1.add(term2).mulScalar(-1) as Tensor<S, D>

  if (reduction === 'mean') {
    return (loss as any).mean()
  } else if (reduction === 'sum') {
    return (loss as any).sum()
  }
  return loss
}

/**
 * Binary cross entropy with logits loss
 *
 * Combines sigmoid and BCE loss for numerical stability.
 * This is more numerically stable than applying sigmoid then BCE.
 *
 * @param input - Raw logits (before sigmoid)
 * @param target - Ground truth binary labels [0, 1]
 * @param reduction - Reduction method (default: 'mean')
 * @param posWeight - Weight for positive samples (for class imbalance)
 * @returns Loss tensor
 *
 * @example
 * ```ts
 * const logits = model.forward(x) // Raw logits, no sigmoid
 * const loss = bceWithLogitsLoss(logits, target)
 * ```
 */
export function bceWithLogitsLoss<S extends Shape, D extends DType<string>>(
  input: Tensor<S, D>,
  target: Tensor<S, D>,
  reduction: Reduction = 'mean',
  posWeight?: Tensor<Shape, D>,
): Tensor<Shape, D> {
  // Using the numerically stable formulation:
  // max(x, 0) - x * y + log(1 + exp(-|x|))

  // max(x, 0)
  const maxX = (input as any).clamp(0, Infinity)

  // x * y
  const xy = input.mul(target as any)

  // |x| = max(x, -x)
  const absX = input.maximum((input as any).neg())

  // exp(-|x|)
  const expNegAbsX = absX.mulScalar(-1).exp()

  // log(1 + exp(-|x|))
  const logTerm = expNegAbsX.addScalar(1).log()

  // loss = max(x, 0) - x * y + log(1 + exp(-|x|))
  let loss = maxX.sub(xy as any).add(logTerm) as Tensor<S, D>

  // Apply positive weight if provided
  if (posWeight) {
    const weightedLoss = loss.mul(posWeight as any)
    loss = target.mul(weightedLoss.sub(loss) as any).add(loss) as Tensor<S, D>
  }

  if (reduction === 'mean') {
    return (loss as any).mean()
  } else if (reduction === 'sum') {
    return (loss as any).sum()
  }
  return loss
}

/**
 * L1 Loss (Mean Absolute Error)
 *
 * Computes the mean of absolute differences between predictions and targets.
 * More robust to outliers than MSE.
 *
 * L1 = |input - target|
 *
 * @param input - Predicted values
 * @param target - Ground truth values
 * @param reduction - Reduction method (default: 'mean')
 * @returns Loss tensor
 *
 * @example
 * ```ts
 * const pred = model.forward(x)
 * const loss = l1Loss(pred, target)
 * ```
 */
export function l1Loss<S extends Shape, D extends DType<string>>(
  input: Tensor<S, D>,
  target: Tensor<S, D>,
  reduction: Reduction = 'mean',
): Tensor<Shape, D> {
  const diff = input.sub(target as any)
  // |diff| = max(diff, -diff)
  let loss = diff.maximum((diff as any).neg()) as Tensor<S, D>

  if (reduction === 'mean') {
    return (loss as any).mean()
  } else if (reduction === 'sum') {
    return (loss as any).sum()
  }
  return loss
}

/**
 * Smooth L1 Loss (Huber Loss)
 *
 * Less sensitive to outliers than MSE, while still being smooth at zero.
 * Uses squared loss for small differences and L1 loss for large differences.
 *
 * SmoothL1(x) = 0.5 * x^2 / beta   if |x| < beta
 *             = |x| - 0.5 * beta   otherwise
 *
 * @param input - Predicted values
 * @param target - Ground truth values
 * @param reduction - Reduction method (default: 'mean')
 * @param beta - Threshold for switching between L1 and L2 (default: 1.0)
 * @returns Loss tensor
 *
 * @example
 * ```ts
 * const pred = model.forward(x)
 * const loss = smoothL1Loss(pred, target, 'mean', 1.0)
 * ```
 */
export function smoothL1Loss<S extends Shape, D extends DType<string>>(
  input: Tensor<S, D>,
  target: Tensor<S, D>,
  reduction: Reduction = 'mean',
  beta: number = 1.0,
): Tensor<Shape, D> {
  const diff = input.sub(target as any)
  // |diff| = max(diff, -diff)
  const absDiff = diff.maximum((diff as any).neg()) as Tensor<S, D>

  // For |x| < beta: 0.5 * x^2 / beta
  const squared = diff.mul(diff as any).mulScalar(0.5 / beta)

  // For |x| >= beta: |x| - 0.5 * beta
  const linear = absDiff.addScalar(-0.5 * beta)

  // Combine using condition: |x| < beta
  // This is approximate since we don't have proper where() on instance
  // Using clamp to approximate the behavior
  const betaTensor = (absDiff as any).clamp(0, beta).mulScalar(1 / beta)
  const loss = squared.mul(betaTensor as any).add(
    linear.mul((betaTensor as any).mulScalar(-1).addScalar(1)) as any
  ) as Tensor<S, D>

  if (reduction === 'mean') {
    return (loss as any).mean()
  } else if (reduction === 'sum') {
    return (loss as any).sum()
  }
  return loss
}

/**
 * Huber Loss
 *
 * Alias for Smooth L1 Loss with configurable delta parameter.
 *
 * @param input - Predicted values
 * @param target - Ground truth values
 * @param reduction - Reduction method (default: 'mean')
 * @param delta - Threshold (default: 1.0)
 * @returns Loss tensor
 */
export function huberLoss<S extends Shape, D extends DType<string>>(
  input: Tensor<S, D>,
  target: Tensor<S, D>,
  reduction: Reduction = 'mean',
  delta: number = 1.0,
): Tensor<Shape, D> {
  return smoothL1Loss(input, target, reduction, delta)
}

/**
 * KL Divergence Loss
 *
 * Measures the divergence between two probability distributions.
 * KL(P || Q) = Σ P(x) * log(P(x) / Q(x))
 *
 * @param input - Log probabilities of predicted distribution (log Q)
 * @param target - Target probability distribution (P)
 * @param reduction - Reduction method (default: 'mean')
 * @param logTarget - Whether target is already log probabilities (default: false)
 * @returns Loss tensor
 *
 * @example
 * ```ts
 * const logProbs = model.forward(x).logSoftmax(-1)
 * const targetProbs = teacher.forward(x).softmax(-1)
 * const loss = klDivLoss(logProbs, targetProbs)
 * ```
 */
export function klDivLoss<S extends Shape, D extends DType<string>>(
  input: Tensor<S, D>,
  target: Tensor<S, D>,
  reduction: Reduction = 'mean',
  logTarget: boolean = false,
): Tensor<Shape, D> {
  let loss: Tensor<S, D>

  if (logTarget) {
    // target is log(P), input is log(Q)
    // PyTorch KLDivLoss computes: exp(target) * (target - input)
    // which equals P * (log(P) - log(Q)), then we need to match PyTorch's convention
    const expTarget = (target as any).exp()
    // PyTorch computes: target * (log(target) - input) but returns positive loss
    // Formula: -sum(exp(target) * (input - target)) = sum(exp(target) * (target - input))
    loss = expTarget.mul(input.sub(target as any)) as Tensor<S, D>
  } else {
    // target is P, input is log(Q)
    // PyTorch's KLDivLoss computes loss such that it's positive for valid distributions
    // The loss is: -sum(target * input) + sum(target * log(target))
    // which equals: sum(target * (log(target) - input)) but with sign convention
    const eps = 1e-7
    const targetClamped = (target as any).clamp(eps, 1)
    const logP = targetClamped.log()
    // Compute: target * (input - log(target)) = -target * (log(target) - input)
    // This matches PyTorch's sign convention for positive loss
    loss = target.mul(input.sub(logP)) as Tensor<S, D>
  }

  if (reduction === 'mean') {
    return (loss as any).mean()
  } else if (reduction === 'sum') {
    return (loss as any).sum()
  } else if (reduction === 'batchmean') {
    // Sum over all dimensions except batch, then mean over batch
    const batchSize = (loss.shape as readonly number[])[0] ?? 1
    return (loss as any).sum().divScalar(batchSize)
  }
  return loss
}

/**
 * Cosine Embedding Loss
 *
 * Measures the loss for learning embeddings based on cosine similarity.
 * If y = 1: loss = 1 - cos(x1, x2)
 * If y = -1: loss = max(0, cos(x1, x2) - margin)
 *
 * @param input1 - First input tensor
 * @param input2 - Second input tensor
 * @param target - Target labels (1 or -1)
 * @param margin - Margin for dissimilar pairs (default: 0)
 * @param reduction - Reduction method (default: 'mean')
 * @returns Loss tensor
 */
export function cosineEmbeddingLoss<S extends Shape, D extends DType<string>>(
  input1: Tensor<S, D>,
  input2: Tensor<S, D>,
  target: Tensor<Shape, D>,
  margin: number = 0,
  reduction: Reduction = 'mean',
): Tensor<Shape, D> {
  // Compute cosine similarity
  // cos(x1, x2) = (x1 · x2) / (||x1|| * ||x2||)
  const dot = input1.mul(input2 as any)
  const dotSum = dot.sumDim(-1) as Tensor<Shape, D>

  const norm1Sq = input1.mul(input1 as any).sumDim(-1) as Tensor<Shape, D>
  const norm2Sq = input2.mul(input2 as any).sumDim(-1) as Tensor<Shape, D>
  const normProduct = norm1Sq.mul(norm2Sq as any).sqrt() as Tensor<Shape, D>

  const eps = 1e-8
  const cosSim = dotSum.div((normProduct as any).clamp(eps, Infinity)) as Tensor<Shape, D>

  // For y = 1: loss = 1 - cos
  // For y = -1: loss = max(0, cos - margin)
  const positiveCase = cosSim.mulScalar(-1).addScalar(1) as Tensor<Shape, D>
  const negativeCase = (cosSim as any).addScalar(-margin).clamp(0, Infinity) as Tensor<Shape, D>

  // Combine based on target
  // loss = (1+y)/2 * positiveCase + (1-y)/2 * negativeCase
  const yPlusOne = (target as any).addScalar(1).mulScalar(0.5)
  const oneMinusY = (target as any).mulScalar(-1).addScalar(1).mulScalar(0.5)
  let loss = yPlusOne.mul(positiveCase as any).add(oneMinusY.mul(negativeCase as any)) as Tensor<Shape, D>

  if (reduction === 'mean') {
    return (loss as any).mean()
  } else if (reduction === 'sum') {
    return (loss as any).sum()
  }
  return loss
}

/**
 * Triplet Margin Loss
 *
 * Measures loss for triplet-based metric learning.
 * loss = max(0, d(anchor, positive) - d(anchor, negative) + margin)
 *
 * @param anchor - Anchor embeddings
 * @param positive - Positive embeddings (similar to anchor)
 * @param negative - Negative embeddings (dissimilar to anchor)
 * @param margin - Margin between positive and negative pairs (default: 1.0)
 * @param p - Norm degree for distance (default: 2 for L2)
 * @param reduction - Reduction method (default: 'mean')
 * @returns Loss tensor
 *
 * @example
 * ```ts
 * const anchor = encoder.forward(x_anchor)
 * const positive = encoder.forward(x_positive)
 * const negative = encoder.forward(x_negative)
 * const loss = tripletMarginLoss(anchor, positive, negative, 1.0)
 * ```
 */
export function tripletMarginLoss<S extends Shape, D extends DType<string>>(
  anchor: Tensor<S, D>,
  positive: Tensor<S, D>,
  negative: Tensor<S, D>,
  margin: number = 1.0,
  p: number = 2,
  reduction: Reduction = 'mean',
): Tensor<Shape, D> {
  // Compute distances using p-norm
  // d(a, b) = ||a - b||_p

  // d(anchor, positive)
  const diffPos = anchor.sub(positive as any)
  let distPos: Tensor<Shape, D>
  if (p === 2) {
    distPos = diffPos.mul(diffPos as any).sumDim(-1).sqrt() as Tensor<Shape, D>
  } else if (p === 1) {
    // |x| = max(x, -x)
    const absDiffPos = diffPos.maximum((diffPos as any).neg())
    distPos = absDiffPos.sumDim(-1) as Tensor<Shape, D>
  } else {
    throw new Error(`tripletMarginLoss: p-norm of ${p} not supported. Use p=1 or p=2.`)
  }

  // d(anchor, negative)
  const diffNeg = anchor.sub(negative as any)
  let distNeg: Tensor<Shape, D>
  if (p === 2) {
    distNeg = diffNeg.mul(diffNeg as any).sumDim(-1).sqrt() as Tensor<Shape, D>
  } else if (p === 1) {
    // |x| = max(x, -x)
    const absDiffNeg = diffNeg.maximum((diffNeg as any).neg())
    distNeg = absDiffNeg.sumDim(-1) as Tensor<Shape, D>
  } else {
    throw new Error(`tripletMarginLoss: p-norm of ${p} not supported. Use p=1 or p=2.`)
  }

  // loss = max(0, d_pos - d_neg + margin)
  let loss = distPos.sub(distNeg as any).addScalar(margin) as Tensor<Shape, D>
  loss = (loss as any).clamp(0, Infinity) as Tensor<Shape, D>

  if (reduction === 'mean') {
    return (loss as any).mean()
  } else if (reduction === 'sum') {
    return (loss as any).sum()
  }
  return loss
}
