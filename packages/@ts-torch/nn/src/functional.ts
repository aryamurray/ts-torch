/**
 * Functional neural network operations (stateless)
 *
 * These functions apply operations directly to tensors without maintaining state.
 * Use these when you don't need learnable parameters (e.g., in custom forward passes).
 *
 * Contrast with nn.Module versions:
 * - Functional: stateless, no parameters, lower level
 * - Module: stateful, may have parameters, higher level abstraction
 */

import type { Tensor, float32 } from './module.js'
import type { Shape, DType } from '@ts-torch/core'

/**
 * Apply ReLU activation: max(0, x)
 *
 * @template S - Tensor shape (preserved)
 * @template D - Data type
 * @param x - Input tensor
 * @returns Output tensor with same shape
 *
 * @example
 * ```ts
 * const x: Tensor<readonly [32, 128]> = ...;
 * const y = relu(x); // Type: Tensor<readonly [32, 128]>
 * ```
 */
export function relu<S extends Shape, D extends DType<string> = float32>(x: Tensor<S, D>): Tensor<S, D> {
  return x.relu()
}

/**
 * Apply Sigmoid activation: 1 / (1 + e^(-x))
 *
 * @template S - Tensor shape (preserved)
 * @template D - Data type
 * @param x - Input tensor
 * @returns Output tensor with same shape, values in (0, 1)
 *
 * @example
 * ```ts
 * const logits: Tensor<readonly [32, 10]> = ...;
 * const probs = sigmoid(logits); // Type: Tensor<readonly [32, 10]>
 * ```
 */
export function sigmoid<S extends Shape, D extends DType<string> = float32>(x: Tensor<S, D>): Tensor<S, D> {
  return x.sigmoid()
}

/**
 * Apply Tanh activation: (e^x - e^(-x)) / (e^x + e^(-x))
 *
 * @template S - Tensor shape (preserved)
 * @template D - Data type
 * @param x - Input tensor
 * @returns Output tensor with same shape, values in (-1, 1)
 *
 * @example
 * ```ts
 * const x: Tensor<readonly [32, 64]> = ...;
 * const y = tanh(x); // Type: Tensor<readonly [32, 64]>
 * ```
 */
export function tanh<S extends Shape, D extends DType<string> = float32>(x: Tensor<S, D>): Tensor<S, D> {
  return x.tanh()
}

/**
 * Apply Softmax activation along specified dimension
 *
 * @template S - Tensor shape (preserved)
 * @template D - Data type
 * @param x - Input tensor
 * @param dim - Dimension along which to apply softmax
 * @returns Output tensor with same shape, values sum to 1 along dim
 *
 * @example
 * ```ts
 * const logits: Tensor<readonly [32, 10]> = ...;
 * const probs = softmax(logits, -1); // Type: Tensor<readonly [32, 10]>
 * // Each row sums to 1
 * ```
 */
export function softmax<S extends Shape, D extends DType<string> = float32>(
  x: Tensor<S, D>,
  _dim: number = -1,
): Tensor<S, D> {
  return x.softmax(_dim)
}

/**
 * Apply Leaky ReLU activation: max(negative_slope * x, x)
 *
 * @template S - Tensor shape (preserved)
 * @template D - Data type
 * @param x - Input tensor
 * @param negativeSlope - Slope for negative values (default: 0.01)
 * @returns Output tensor with same shape
 *
 * @example
 * ```ts
 * const x: Tensor<readonly [32, 128]> = ...;
 * const y = leakyRelu(x, 0.01); // Type: Tensor<readonly [32, 128]>
 * ```
 */
export function leakyRelu<S extends Shape, D extends DType<string> = float32>(
  x: Tensor<S, D>,
  _negativeSlope: number = 0.01,
): Tensor<S, D> {
  const negative = x.mulScalar(_negativeSlope)
  return x.maximum(negative)
}

/**
 * Apply GELU activation
 *
 * @template S - Tensor shape (preserved)
 * @template D - Data type
 * @param x - Input tensor
 * @returns Output tensor with same shape
 *
 * @example
 * ```ts
 * const x: Tensor<readonly [32, 768]> = ...;
 * const y = gelu(x); // Type: Tensor<readonly [32, 768]>
 * ```
 */
export function gelu<S extends Shape, D extends DType<string> = float32>(x: Tensor<S, D>): Tensor<S, D> {
  const x3 = x.mul(x).mul(x)
  const inner = x.add(x3.mulScalar(0.044715)).mulScalar(Math.sqrt(2 / Math.PI))
  const tanhValue = inner.tanh()
  return x.mul(tanhValue.addScalar(1)).mulScalar(0.5)
}

/**
 * Apply dropout during training
 *
 * During training, randomly zeros elements with probability p.
 * During evaluation, returns input unchanged.
 *
 * @template S - Tensor shape (preserved)
 * @template D - Data type
 * @param x - Input tensor
 * @param p - Dropout probability (0 <= p < 1)
 * @param training - Whether in training mode
 * @returns Output tensor with same shape
 *
 * @example
 * ```ts
 * const x: Tensor<readonly [32, 128]> = ...;
 * const y = dropout(x, 0.5, true); // Type: Tensor<readonly [32, 128]>
 * ```
 */
export function dropout<S extends Shape, D extends DType<string> = float32>(
  x: Tensor<S, D>,
  p: number = 0.5,
  training: boolean = true,
): Tensor<S, D> {
  if (!training || p === 0) {
    return x
  }

  if (p < 0 || p >= 1) {
    throw new Error(`Dropout probability must be in [0, 1), got ${p}`)
  }

  return x.dropout(p, training)
}

/**
 * Apply linear transformation: xW^T + b
 *
 * @template BatchShape - Batch dimensions
 * @template InFeatures - Input feature dimension
 * @template OutFeatures - Output feature dimension
 * @template D - Data type
 * @param input - Input tensor [...BatchShape, InFeatures]
 * @param weight - Weight matrix [OutFeatures, InFeatures]
 * @param bias - Optional bias vector [OutFeatures]
 * @returns Output tensor [...BatchShape, OutFeatures]
 *
 * @example
 * ```ts
 * const x: Tensor<readonly [32, 784]> = ...;
 * const w: Tensor<readonly [128, 784]> = ...;
 * const b: Tensor<readonly [128]> = ...;
 * const y = linear(x, w, b); // Type: Tensor<readonly [32, 128]>
 * ```
 */
export function linear<
  BatchShape extends readonly number[],
  InFeatures extends number,
  OutFeatures extends number,
  D extends DType<string> = float32,
>(
  input: Tensor<readonly [...BatchShape, InFeatures], D>,
  _weight: Tensor<readonly [OutFeatures, InFeatures], D>,
  _bias?: Tensor<readonly [OutFeatures], D>,
): Tensor<readonly [...BatchShape, OutFeatures], D> {
  let output = input.matmul(_weight.transpose(0, 1)) as Tensor<readonly [...BatchShape, OutFeatures], D>
  if (_bias) {
    output = output.add(_bias) as Tensor<readonly [...BatchShape, OutFeatures], D>
  }
  return output
}

/**
 * Apply log softmax for numerical stability
 *
 * Equivalent to log(softmax(x)) but more numerically stable.
 * Useful for computing log probabilities.
 *
 * @template S - Tensor shape (preserved)
 * @template D - Data type
 * @param x - Input tensor
 * @param dim - Dimension along which to apply log_softmax
 * @returns Output tensor with same shape
 *
 * @example
 * ```ts
 * const logits: Tensor<readonly [32, 10]> = ...;
 * const logProbs = logSoftmax(logits, -1); // Type: Tensor<readonly [32, 10]>
 * ```
 */
export function logSoftmax<S extends Shape, D extends DType<string> = float32>(
  x: Tensor<S, D>,
  _dim: number = -1,
): Tensor<S, D> {
  return x.logSoftmax(_dim)
}

/**
 * Normalize a tensor along specified dimensions
 *
 * @template S - Tensor shape (preserved)
 * @template D - Data type
 * @param x - Input tensor
 * @param p - Order of norm (default: 2 for L2 norm)
 * @param dim - Dimension to normalize along
 * @param eps - Small value to avoid division by zero (default: 1e-12)
 * @returns Normalized tensor with same shape
 *
 * @example
 * ```ts
 * const x: Tensor<readonly [32, 128]> = ...;
 * const normalized = normalize(x, 2, -1); // L2 normalize each row
 * ```
 */
export function normalize<S extends Shape, D extends DType<string> = float32>(
  x: Tensor<S, D>,
  _p: number = 2,
  _dim: number = -1,
  _eps: number = 1e-12,
): Tensor<S, D> {
  if (_p !== 2) {
    throw new Error(`normalize currently supports p=2, got p=${_p}`)
  }
  const norm = x.mul(x).sumDim(_dim, true).sqrt().clampMin(_eps)
  return x.div(norm) as Tensor<S, D>
}

/**
 * Clamp (clip) tensor values to a specified range
 *
 * @template S - Tensor shape (preserved)
 * @template D - Data type
 * @param x - Input tensor
 * @param min - Minimum value (or null for no minimum)
 * @param max - Maximum value (or null for no maximum)
 * @returns Clamped tensor with same shape
 *
 * @example
 * ```ts
 * const x: Tensor<readonly [32, 128]> = ...;
 * const clamped = clamp(x, 0, 1); // Clamp to [0, 1]
 * ```
 */
export function clamp<S extends Shape, D extends DType<string> = float32>(
  x: Tensor<S, D>,
  _min: number | null = null,
  _max: number | null = null,
): Tensor<S, D> {
  if (_min === null && _max === null) {
    return x
  }
  if (_min === null) {
    return x.clampMax(_max as number)
  }
  if (_max === null) {
    return x.clampMin(_min)
  }
  return x.clamp(_min, _max)
}
