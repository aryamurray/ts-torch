/**
 * @ts-torch/datasets - Dataset loaders and utilities
 *
 * This package provides dataset loading utilities, data transformations,
 * and common dataset implementations for machine learning with ts-torch.
 *
 * @example
 * ```ts
 * import { Data, MNIST } from '@ts-torch/datasets'
 *
 * const mnist = new MNIST('./data/mnist', true)
 * await mnist.load()
 *
 * // Batches yield { input, target } directly
 * const pipeline = Data.pipeline(mnist)
 *   .shuffle()
 *   .batch(64)
 *
 * for await (const batch of pipeline) {
 *   // batch.input, batch.target
 * }
 * ```
 */

export * from './dataset.js'
export * from './dataloader.js'
export * from './transforms.js'
export * from './pipeline.js'
export * from './vision/index.js'
export * from './text/index.js'
export * from './worker-loader/index.js'
