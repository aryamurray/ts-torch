/**
 * Worker Pipeline - fluent API for worker-based data loading
 *
 * Provides a similar API to DataPipeline but uses worker threads
 * for parallel data loading.
 */

import type { TransformConfig, TensorPair } from './types.js'
import { WorkerDataLoader, type DatasetFactory, type WorkerDataLoaderConfig } from './worker-data-loader.js'

/**
 * Configuration builder for worker-based data loading
 */
export class WorkerPipeline {
  private config: Partial<WorkerDataLoaderConfig> = {}
  private datasetLength: number
  private factory: DatasetFactory

  constructor(datasetLength: number, factory: DatasetFactory) {
    this.datasetLength = datasetLength
    this.factory = factory
  }

  /**
   * Set the number of worker threads
   *
   * @param n - Number of workers (default: os.cpus().length)
   */
  workers(n: number): WorkerPipeline {
    this.config.numWorkers = n
    return this
  }

  /**
   * Set batch size
   *
   * @param size - Number of samples per batch
   */
  batch(size: number): WorkerPipeline {
    this.config.batchSize = size
    return this
  }

  /**
   * Enable shuffling
   */
  shuffle(): WorkerPipeline {
    this.config.shuffle = true
    return this
  }

  /**
   * Set number of batches to prefetch
   *
   * @param count - Number of batches to keep ready (default: 2)
   */
  prefetch(count: number): WorkerPipeline {
    this.config.prefetchCount = count
    return this
  }

  /**
   * Drop the last incomplete batch
   */
  dropLast(): WorkerPipeline {
    this.config.dropLast = true
    return this
  }

  /**
   * Add transforms to apply in workers
   *
   * @param transforms - Array of transform configurations
   */
  transform(...transforms: TransformConfig[]): WorkerPipeline {
    this.config.transforms = [...(this.config.transforms ?? []), ...transforms]
    return this
  }

  /**
   * Set maximum bytes per batch slot
   *
   * @param bytes - Maximum bytes per slot (default: 16MB)
   */
  maxBatchBytes(bytes: number): WorkerPipeline {
    this.config.maxBatchBytes = bytes
    return this
  }

  /**
   * Set worker task timeout
   *
   * @param ms - Timeout in milliseconds (default: 30000)
   */
  timeout(ms: number): WorkerPipeline {
    this.config.workerTimeout = ms
    return this
  }

  /**
   * Build the WorkerDataLoader
   *
   * @returns WorkerDataLoader ready for iteration
   */
  build(): WorkerDataLoader {
    if (!this.config.batchSize) {
      throw new Error('batchSize is required. Call .batch(size) before .build()')
    }

    return new WorkerDataLoader(this.datasetLength, {
      ...this.config,
      batchSize: this.config.batchSize,
      datasetFactory: this.factory,
    })
  }

  /**
   * Iterate over batches
   *
   * Convenience method that builds the loader and returns its async iterator.
   */
  async *[Symbol.asyncIterator](): AsyncIterator<TensorPair> {
    const loader = this.build()
    try {
      yield* loader
    } finally {
      await loader.shutdown()
    }
  }
}

/**
 * Create a worker-based data pipeline
 *
 * @param datasetLength - Total number of samples in the dataset
 * @param factory - Factory configuration for creating the dataset in workers
 * @returns WorkerPipeline for fluent configuration
 *
 * @example
 * ```typescript
 * import { workerPipeline, Transforms } from '@ts-torch/datasets'
 *
 * const loader = workerPipeline(60000, {
 *   module: './mnist-dataset.js',
 *   exportName: 'createMNIST',
 *   args: ['./data/mnist']
 * })
 *   .workers(4)
 *   .batch(64)
 *   .shuffle()
 *   .prefetch(2)
 *   .transform(
 *     Transforms.normalize([0.1307], [0.3081])
 *   )
 *   .build()
 *
 * for await (const batch of loader) {
 *   // batch.data and batch.label are tensors
 * }
 *
 * await loader.shutdown()
 * ```
 */
export function workerPipeline(
  datasetLength: number,
  factory: DatasetFactory,
): WorkerPipeline {
  return new WorkerPipeline(datasetLength, factory)
}
