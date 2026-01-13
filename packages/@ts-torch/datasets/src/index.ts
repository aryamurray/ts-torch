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
 * const pipeline = Data.pipeline(mnist)
 *   .shuffle()
 *   .batch(64)
 *   .to('cuda')
 *   .map(b => ({ data: b.images, label: b.labelsTensor }))
 *
 * for await (const batch of pipeline) {
 *   // train...
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
