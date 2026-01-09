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

import type { Tensor, float32 } from "./module.js";
import type { Shape, DType } from "@ts-torch/core";

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
export function relu<S extends Shape, D extends DType<string> = float32>(
  x: Tensor<S, D>,
): Tensor<S, D> {
  // TODO: Implement when Tensor ops are ready
  // return x.relu();
  return x as any; // Placeholder
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
export function sigmoid<S extends Shape, D extends DType<string> = float32>(
  x: Tensor<S, D>,
): Tensor<S, D> {
  // TODO: Implement when Tensor ops are ready
  // return x.sigmoid();
  return x as any; // Placeholder
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
export function tanh<S extends Shape, D extends DType<string> = float32>(
  x: Tensor<S, D>,
): Tensor<S, D> {
  // TODO: Implement when Tensor ops are ready
  // return x.tanh();
  return x as any; // Placeholder
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
  // TODO: Implement when Tensor ops are ready
  // return x.softmax(dim);
  return x as any; // Placeholder
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
  // TODO: Implement when Tensor ops are ready
  // return x.leakyRelu(negativeSlope);
  return x as any; // Placeholder
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
export function gelu<S extends Shape, D extends DType<string> = float32>(
  x: Tensor<S, D>,
): Tensor<S, D> {
  // TODO: Implement when Tensor ops are ready
  // return x.gelu();
  return x as any; // Placeholder
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
    return x;
  }

  if (p < 0 || p >= 1) {
    throw new Error(`Dropout probability must be in [0, 1), got ${p}`);
  }

  // TODO: Implement actual dropout when Tensor ops are ready
  // During training:
  // const mask = (random(x.shape) > p).cast(x.dtype);
  // return x.mul(mask).div(1 - p); // Scale to maintain expected value

  return x as any; // Placeholder
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
  // TODO: Implement when Tensor ops are ready
  // let output = input.matmul(weight.transpose(-1, -2));
  // if (bias) {
  //   output = output.add(bias);
  // }
  // return output;

  return input as any; // Placeholder
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
  // TODO: Implement when Tensor ops are ready
  // return x.logSoftmax(dim);
  //
  // Numerically stable implementation:
  // const maxVals = x.max(dim=dim, keepdim=true);
  // const shifted = x.sub(maxVals);
  // const logSumExp = shifted.exp().sum(dim=dim, keepdim=true).log();
  // return shifted.sub(logSumExp);

  return x as any; // Placeholder
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
  // TODO: Implement when Tensor ops are ready
  // const norm = x.norm(p, dim=dim, keepdim=true);
  // return x.div(norm.clamp(min=eps));

  return x as any; // Placeholder
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
  // TODO: Implement when Tensor ops are ready
  // return x.clamp(min, max);

  return x as any; // Placeholder
}
