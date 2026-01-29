/**
 * @ts-torch/optim - Optimizers for neural network training
 *
 * This package provides various optimization algorithms for training neural networks,
 * including SGD, Adam, RMSprop, and more. It also includes loss functions and
 * learning rate schedulers.
 */

// Base optimizer
export * from './optimizer.js'

// Optimizer tensor wrapper (safe in-place operations)
export { OptimizerTensor, wrapForOptim } from './optimizer-tensor.js'

// Optimizers
export * from './sgd.js'
export * from './adam.js'
export * from './rmsprop.js'
export * from './adamw.js'

// Loss functions
export * from './loss.js'

// Learning rate schedulers
export * from './lr_scheduler.js'

// Gradient clipping utilities
export * from './grad_clip.js'
