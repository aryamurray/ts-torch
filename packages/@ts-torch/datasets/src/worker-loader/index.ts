/**
 * Worker-based DataLoader module
 *
 * Provides multi-threaded data loading using worker threads
 * to prevent the main thread from blocking on data loading.
 */

// Main exports
export { WorkerDataLoader, createWorkerDataLoader } from './worker-data-loader.js'
export type { DatasetFactory, WorkerDataLoaderConfig } from './worker-data-loader.js'

// Fluent API
export { WorkerPipeline, workerPipeline } from './worker-pipeline.js'

// Transform utilities
export { Transforms, createTransform, createTransforms, composeTransforms } from './transform-registry.js'
export type { TransformFn } from './transform-registry.js'

// Types
export type {
  WorkerDataLoaderOptions,
  TransformConfig,
  TransformType,
  TensorPair,
  BatchMetadata,
  DType,
} from './types.js'
