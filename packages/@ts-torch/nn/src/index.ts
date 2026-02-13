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
  type InitStrategy as LinearInitStrategy,
  type InitFn,
  type ConstantInit,
} from './modules/linear.js'

export {
  // Activation functions
  ReLU,
  Sigmoid,
  Tanh,
  Softmax,
  LeakyReLU,
  GELU,
  ELU,
  SELU,
  SiLU,
  Swish,
  Mish,
  Hardswish,
  Hardsigmoid,
  Hardtanh,
  ReLU6,
  PReLU,
  Softplus,
  Softsign,
  LogSoftmax,
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
export * from './modules/embedding.js'
export * from './modules/attention.js'
export * from './modules/rnn.js'
export * from './modules/transformer.js'

// Weight initialization
export * from './init.js'

// Functional API (stateless operations)
export * as F from './functional.js'

// Checkpoint / Serialization
export {
  saveCheckpoint,
  loadCheckpoint,
  encodeCheckpoint,
  decodeCheckpoint,
  float32Tensor,
  paramsToTensors,
} from './checkpoint.js'
export type { TensorData, CheckpointData, StateDict } from './checkpoint.js'

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

// ==================== Fluent Builders ====================

/**
 * Fluent model builders - separates configuration from memory allocation
 *
 * @example
 * ```ts
 * import { nn } from '@ts-torch/nn';
 * import { device } from '@ts-torch/core';
 *
 * // Define model (no memory allocated - just JS objects)
 * const config = nn.sequence(
 *   nn.input(784),
 *   nn.fc(128).relu().dropout(0.2),
 *   nn.fc(64).gelu(),
 *   nn.fc(10)
 * );
 *
 * // Initialize model on device (memory allocated here)
 * const model = config.init(device.cuda(0));
 * ```
 */
export { nn, input, sequence, fc } from './builders.js'
export type { InputDef, BlockDef, SequenceDef, ActivationType, InitStrategy } from './builders.js'
