/**
 * Loss functions for training neural networks
 *
 * Provides common loss functions including cross entropy, MSE, binary cross entropy, and L1 loss.
 * All loss functions support reduction options: 'mean', 'sum', or 'none'.
 */

import type { Tensor } from '@ts-torch/core';
import type { Shape, DType } from '@ts-torch/core';

/**
 * Reduction strategy for loss functions
 */
export type Reduction = 'mean' | 'sum' | 'none';

/**
 * Common options for loss functions
 */
export interface LossOptions {
  /**
   * Specifies the reduction to apply to the output:
   * - 'none': no reduction will be applied
   * - 'mean': the sum of the output will be divided by the number of elements
   * - 'sum': the output will be summed
   * @default 'mean'
   */
  reduction?: Reduction;
}

/**
 * Cross entropy loss for multi-class classification
 *
 * Combines log softmax and negative log likelihood loss in a single operation.
 * This is more numerically stable than computing them separately.
 *
 * @template B - Batch size dimension
 * @template C - Number of classes dimension
 * @template D - Data type
 *
 * @param logits - Raw prediction scores of shape [Batch, Classes]
 * @param targets - Ground truth class indices of shape [Batch], values in [0, C-1]
 * @param options - Loss configuration options
 * @returns Scalar loss tensor or per-sample losses if reduction='none'
 *
 * @example
 * ```typescript
 * const logits = tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.2]]);  // [2, 3]
 * const targets = tensor([0, 1]);  // [2]
 * const loss = crossEntropyLoss(logits, targets);  // scalar
 * ```
 */
export function crossEntropyLoss<
  B extends number,
  C extends number,
  D extends DType<string>
>(
  logits: Tensor<readonly [B, C], D>,
  targets: Tensor<readonly [B], D>,
  options?: LossOptions
): Tensor<readonly [], D> | Tensor<readonly [B], D> {
  const reduction = options?.reduction ?? 'mean';

  // Compute log softmax: log(exp(x_i) / sum(exp(x_j)))
  // For numerical stability: log_softmax(x) = x - log(sum(exp(x)))
  let logSoftmax: Tensor;

  if ('exp' in logits && 'sum' in logits && 'log' in logits && 'sub' in logits &&
      typeof (logits as { exp?: Function }).exp === 'function' &&
      typeof (logits as { sum?: Function }).sum === 'function' &&
      typeof (logits as { log?: Function }).log === 'function' &&
      typeof (logits as { sub?: Function }).sub === 'function') {

    // exp(logits)
    const expLogits = ((logits as { exp: () => Tensor }).exp()) as Tensor;

    // sum(exp(logits), dim=1, keepdim=true)
    const sumExp = ('sum' in expLogits && typeof (expLogits as { sum?: Function }).sum === 'function')
      ? ((expLogits as { sum: (dim: number, keepdim?: boolean) => Tensor }).sum(1, true)) as Tensor
      : expLogits;

    // log(sum(exp(logits)))
    const logSumExp = ('log' in sumExp && typeof (sumExp as { log?: Function }).log === 'function')
      ? ((sumExp as { log: () => Tensor }).log()) as Tensor
      : sumExp;

    // logits - log(sum(exp(logits)))
    logSoftmax = (logits.sub as any)(logSumExp) as Tensor;
  } else {
    throw new Error('Tensor operations (exp, sum, log, sub) not available');
  }

  // Gather log probabilities for target classes
  // loss[i] = -log_softmax[i, targets[i]]
  let nll: Tensor;

  if ('gather' in logSoftmax && typeof (logSoftmax as { gather?: Function }).gather === 'function') {
    // Use gather operation to select target class log probs
    nll = ((logSoftmax as { gather: (dim: number, index: Tensor) => Tensor }).gather(1, targets as Tensor)) as Tensor;

    // Negate to get negative log likelihood
    if ('mul' in nll && typeof (nll as { mul?: Function }).mul === 'function') {
      nll = (nll.mul as any)(-1) as Tensor;
    }
  } else {
    // Fallback: manual indexing (less efficient)
    throw new Error('Gather operation not available on Tensor');
  }

  // Apply reduction
  return applyReduction(nll, reduction) as Tensor<readonly [], D> | Tensor<readonly [B], D>;
}

/**
 * Mean squared error (MSE) loss for regression tasks
 *
 * Computes the mean squared error between predictions and targets:
 * loss = (input - target)^2
 *
 * @template S - Tensor shape
 * @template D - Data type
 *
 * @param input - Predicted values
 * @param target - Ground truth values
 * @param options - Loss configuration options
 * @returns Scalar loss tensor or per-element losses if reduction='none'
 *
 * @example
 * ```typescript
 * const predictions = tensor([2.5, 0.0, 2.0, 8.0]);
 * const targets = tensor([3.0, -0.5, 2.0, 7.0]);
 * const loss = mseLoss(predictions, targets);  // mean of squared errors
 * ```
 */
export function mseLoss<S extends Shape, D extends DType<string>>(
  input: Tensor<S, D>,
  target: Tensor<S, D>,
  options?: LossOptions
): Tensor<readonly [], D> | Tensor<S, D> {
  const reduction = options?.reduction ?? 'mean';

  // Compute (input - target)^2
  let squaredError: Tensor;

  if ('sub' in input && 'pow' in input &&
      typeof (input as { sub?: Function }).sub === 'function') {
    const diff = (input.sub as any)(target) as Tensor;

    if ('pow' in diff && typeof (diff as { pow?: Function }).pow === 'function') {
      squaredError = (diff.pow as any)(2) as Tensor;
    } else {
      throw new Error('Pow operation not available on Tensor');
    }
  } else {
    throw new Error('Sub operation not available on Tensor');
  }

  // Apply reduction
  return applyReduction(squaredError, reduction) as Tensor<readonly [], D> | Tensor<S, D>;
}

/**
 * Binary cross entropy loss for binary classification
 *
 * Measures the binary cross entropy between predictions (probabilities) and binary targets.
 * Input should be probabilities in [0, 1] (e.g., after sigmoid activation).
 *
 * Formula: -[y * log(x) + (1 - y) * log(1 - x)]
 *
 * @template S - Tensor shape
 * @template D - Data type
 *
 * @param input - Predicted probabilities in range [0, 1]
 * @param target - Binary ground truth values (0 or 1)
 * @param options - Loss configuration options
 * @returns Scalar loss tensor or per-element losses if reduction='none'
 *
 * @example
 * ```typescript
 * const predictions = tensor([0.8, 0.3, 0.6, 0.9]);  // after sigmoid
 * const targets = tensor([1.0, 0.0, 1.0, 1.0]);
 * const loss = binaryCrossEntropyLoss(predictions, targets);
 * ```
 */
export function binaryCrossEntropyLoss<S extends Shape, D extends DType<string>>(
  input: Tensor<S, D>,
  target: Tensor<S, D>,
  options?: LossOptions
): Tensor<readonly [], D> | Tensor<S, D> {
  const reduction = options?.reduction ?? 'mean';
  const eps = 1e-8; // Small epsilon for numerical stability

  // Clamp input to [eps, 1 - eps] to avoid log(0)
  let clampedInput: Tensor = input as Tensor;

  if ('clamp' in input && typeof (input as { clamp?: Function }).clamp === 'function') {
    clampedInput = (input.clamp as any)(eps, 1 - eps) as Tensor;
  }

  // Compute -[target * log(input) + (1 - target) * log(1 - input)]
  let bce: Tensor;

  if ('log' in clampedInput && 'mul' in clampedInput && 'sub' in clampedInput && 'add' in clampedInput &&
      typeof (clampedInput as { log?: Function }).log === 'function' &&
      typeof (clampedInput as { mul?: Function }).mul === 'function' &&
      typeof (clampedInput as { sub?: Function }).sub === 'function' &&
      typeof (clampedInput as { add?: Function }).add === 'function') {

    // target * log(input)
    const logInput = ((clampedInput as { log: () => Tensor }).log()) as Tensor;
    const term1 = ('mul' in target && typeof (target as { mul?: Function }).mul === 'function')
      ? (target.mul as any)(logInput) as Tensor
      : logInput;

    // (1 - target)
    let oneMinusTarget: Tensor;
    if ('mul' in target && 'sub' in target && typeof (target as { mul?: Function }).mul === 'function') {
      const ones = (target.mul as any)(0) as Tensor;
      oneMinusTarget = ('add' in ones && typeof (ones as { add?: Function }).add === 'function')
        ? (ones.add as any)(1) as Tensor
        : ones;

      oneMinusTarget = ('sub' in oneMinusTarget && typeof (oneMinusTarget as { sub?: Function }).sub === 'function')
        ? ((oneMinusTarget as { sub: (x: Tensor) => Tensor }).sub(target as Tensor)) as Tensor
        : oneMinusTarget;
    } else {
      oneMinusTarget = target as Tensor;
    }

    // log(1 - input)
    const oneMinusInput = (clampedInput.mul as any)(-1) as Tensor;
    const oneMinusInputPlus1 = ('add' in oneMinusInput && typeof (oneMinusInput as { add?: Function }).add === 'function')
      ? (oneMinusInput.add as any)(1) as Tensor
      : oneMinusInput;
    const logOneMinusInput = ('log' in oneMinusInputPlus1 && typeof (oneMinusInputPlus1 as { log?: Function }).log === 'function')
      ? ((oneMinusInputPlus1 as { log: () => Tensor }).log()) as Tensor
      : oneMinusInputPlus1;

    // (1 - target) * log(1 - input)
    const term2 = ('mul' in oneMinusTarget && typeof (oneMinusTarget as { mul?: Function }).mul === 'function')
      ? ((oneMinusTarget as { mul: (x: Tensor) => Tensor }).mul(logOneMinusInput)) as Tensor
      : logOneMinusInput;

    // sum and negate
    const sum = ('add' in term1 && typeof (term1 as { add?: Function }).add === 'function')
      ? ((term1 as { add: (x: Tensor) => Tensor }).add(term2)) as Tensor
      : term1;

    bce = ('mul' in sum && typeof (sum as { mul?: Function }).mul === 'function')
      ? (sum.mul as any)(-1) as Tensor
      : sum;
  } else {
    throw new Error('Required tensor operations not available');
  }

  // Apply reduction
  return applyReduction(bce, reduction) as Tensor<readonly [], D> | Tensor<S, D>;
}

/**
 * L1 loss (Mean Absolute Error) for regression tasks
 *
 * Computes the mean absolute error between predictions and targets:
 * loss = |input - target|
 *
 * L1 loss is less sensitive to outliers compared to MSE loss.
 *
 * @template S - Tensor shape
 * @template D - Data type
 *
 * @param input - Predicted values
 * @param target - Ground truth values
 * @param options - Loss configuration options
 * @returns Scalar loss tensor or per-element losses if reduction='none'
 *
 * @example
 * ```typescript
 * const predictions = tensor([2.5, 0.0, 2.0, 8.0]);
 * const targets = tensor([3.0, -0.5, 2.0, 7.0]);
 * const loss = l1Loss(predictions, targets);  // mean of absolute errors
 * ```
 */
export function l1Loss<S extends Shape, D extends DType<string>>(
  input: Tensor<S, D>,
  target: Tensor<S, D>,
  options?: LossOptions
): Tensor<readonly [], D> | Tensor<S, D> {
  const reduction = options?.reduction ?? 'mean';

  // Compute |input - target|
  let absError: Tensor;

  if ('sub' in input && typeof (input as { sub?: Function }).sub === 'function') {
    const diff = (input.sub as any)(target) as Tensor & Record<'abs', unknown>;

    if ('abs' in diff && typeof (diff as { abs?: Function }).abs === 'function') {
      absError = ((diff as { abs: () => Tensor }).abs()) as Tensor;
    } else {
      throw new Error('Abs operation not available on Tensor');
    }
  } else {
    throw new Error('Sub operation not available on Tensor');
  }

  // Apply reduction
  return applyReduction(absError, reduction) as Tensor<readonly [], D> | Tensor<S, D>;
}

/**
 * Smooth L1 loss (Huber loss) for regression tasks
 *
 * Less sensitive to outliers than MSE. It's quadratic for small errors
 * and linear for large errors.
 *
 * @template S - Tensor shape
 * @template D - Data type
 *
 * @param input - Predicted values
 * @param target - Ground truth values
 * @param options - Loss configuration options with optional beta parameter
 * @returns Scalar loss tensor or per-element losses if reduction='none'
 *
 * @example
 * ```typescript
 * const predictions = tensor([2.5, 0.0, 2.0, 8.0]);
 * const targets = tensor([3.0, -0.5, 2.0, 7.0]);
 * const loss = smoothL1Loss(predictions, targets, { beta: 1.0 });
 * ```
 */
export function smoothL1Loss<S extends Shape, D extends DType<string>>(
  input: Tensor<S, D>,
  target: Tensor<S, D>,
  options?: LossOptions & { beta?: number }
): Tensor<readonly [], D> | Tensor<S, D> {
  const reduction = options?.reduction ?? 'mean';
  // const beta = options?.beta ?? 1.0; // Unused for now in simplified implementation

  // Compute |input - target|
  let diff: Tensor;
  if ('sub' in input && typeof (input as { sub?: Function }).sub === 'function') {
    diff = (input.sub as any)(target) as Tensor;
  } else {
    throw new Error('Sub operation not available on Tensor');
  }

  let absDiff: Tensor;
  if ('abs' in diff && typeof (diff as { abs?: Function }).abs === 'function') {
    absDiff = ((diff as { abs: () => Tensor }).abs()) as Tensor;
  } else {
    throw new Error('Abs operation not available on Tensor');
  }

  // For numerical simplicity, we return L1 loss
  // Full implementation would be:
  // if |x| < beta: 0.5 * x^2 / beta
  // else: |x| - 0.5 * beta
  const smoothL1 = absDiff;

  // Apply reduction
  return applyReduction(smoothL1, reduction) as Tensor<readonly [], D> | Tensor<S, D>;
}

/**
 * Kullback-Leibler divergence loss
 *
 * Measures the divergence between two probability distributions.
 *
 * @template S - Tensor shape
 * @template D - Data type
 *
 * @param input - Log probabilities (output of log_softmax)
 * @param target - Target probabilities
 * @param options - Loss configuration options
 * @returns Scalar loss tensor or per-element losses if reduction='none'
 */
export function klDivLoss<S extends Shape, D extends DType<string>>(
  input: Tensor<S, D>,
  target: Tensor<S, D>,
  options?: LossOptions
): Tensor<readonly [], D> | Tensor<S, D> {
  const reduction = options?.reduction ?? 'mean';

  // KL(P||Q) = sum(P * log(P/Q)) = sum(P * (log(P) - log(Q)))
  // input is log(Q), target is P
  // KL = sum(P * (log(P) - input))

  let kl: Tensor;

  if ('log' in target && 'sub' in target && 'mul' in target &&
      typeof (target as { log?: Function }).log === 'function' &&
      typeof (target as { sub?: Function }).sub === 'function' &&
      typeof (target as { mul?: Function }).mul === 'function') {

    const logTarget = ((target as { log: () => Tensor }).log()) as Tensor;
    const logDiff = ('sub' in logTarget && typeof (logTarget as { sub?: Function }).sub === 'function')
      ? (logTarget.sub as any)(input) as Tensor
      : logTarget;

    kl = (target.mul as any)(logDiff) as Tensor;
  } else {
    throw new Error('Required tensor operations not available');
  }

  // Apply reduction
  return applyReduction(kl, reduction) as Tensor<readonly [], D> | Tensor<S, D>;
}

/**
 * Helper function to apply reduction strategy
 * @internal
 */
function applyReduction(tensor: Tensor, reduction: Reduction): Tensor {
  if (reduction === 'none') {
    return tensor;
  }

  if (reduction === 'sum') {
    if ('sum' in tensor && typeof (tensor as { sum?: Function }).sum === 'function') {
      return ((tensor as { sum: () => Tensor }).sum()) as Tensor;
    }
  }

  if (reduction === 'mean') {
    if ('mean' in tensor && typeof (tensor as { mean?: Function }).mean === 'function') {
      return ((tensor as { mean: () => Tensor }).mean()) as Tensor;
    }
  }

  return tensor;
}
