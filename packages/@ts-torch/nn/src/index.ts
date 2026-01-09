/**
 * @ts-torch/nn - Neural network modules with advanced type safety
 *
 * This package provides type-safe neural network layers with compile-time shape checking
 * and composable .pipe() chaining for building complex models.
 *
 * @example
 * ```ts
 * import { Linear, ReLU, Sequential } from '@ts-torch/nn';
 *
 * // Type-safe module chaining with .pipe()
 * const model = new Linear(784, 128)
 *   .pipe(new ReLU())
 *   .pipe(new Linear(128, 64))
 *   .pipe(new ReLU())
 *   .pipe(new Linear(64, 10));
 *
 * // Or use Sequential
 * const seq = new Sequential(
 *   new Linear(784, 128),
 *   new ReLU(),
 *   new Linear(128, 10)
 * );
 *
 * // Type-safe forward pass
 * const input: Tensor<readonly [32, 784]> = ...;
 * const output = model.forward(input); // Type: Tensor<readonly [32, 10]>
 * ```
 *
 * @module @ts-torch/nn
 */

// Core module abstractions
export { Module, Parameter, PipedModule, type Tensor, type float32 } from './module.js'

// Neural network modules
export {
  // Linear layers
  Linear,
  type LinearOptions,
} from './modules/linear.js'

export {
  // Activation functions
  ReLU,
  Sigmoid,
  Tanh,
  Softmax,
  LeakyReLU,
  GELU,
} from './modules/activation.js'

export {
  // Container modules
  Sequential,
  SequentialBuilder,
  sequential,
} from './modules/container.js'

// Re-export other modules (conv, pooling, etc.)
export * from './modules/conv.js'
export * from './modules/pooling.js'
export * from './modules/normalization.js'
export * from './modules/dropout.js'

// Functional API (stateless operations)
export * as F from './functional.js'

/**
 * Functional namespace (alternative import style)
 *
 * @example
 * ```ts
 * import { functional } from '@ts-torch/nn';
 *
 * const y = functional.relu(x);
 * const z = functional.softmax(y, -1);
 * ```
 */
export * as functional from './functional.js'
