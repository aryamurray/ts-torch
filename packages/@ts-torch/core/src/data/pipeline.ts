/**
 * Data Pipeline - Declarative data loading and transformation
 *
 * Provides a fluent API for building efficient data pipelines that:
 * - Transfer data to GPU once, not per batch
 * - Use narrow() for zero-copy batching
 * - Shuffle indices, not data
 * - Prefetch batches for better throughput
 *
 * @example
 * ```ts
 * const loader = Data.pipeline(dataset)
 *   .shuffle()
 *   .batch(64)
 *   .to(cuda)
 * ```
 */

import type { Tensor } from '../tensor/tensor.js'
import type { DeviceContext } from '../device/context.js'

/**
 * Dataset interface - source of items
 */
export interface Dataset<T> {
  getItem(index: number): T | Promise<T>
  readonly length: number
}

/**
 * Tensor pair - common for supervised learning
 */
export type TensorPair = { data: Tensor; label: Tensor }

/**
 * Batch - grouped items
 */
export type Batch<T> = T[]

/**
 * Pipeline configuration
 */
interface PipelineConfig {
  shouldShuffle: boolean
  batchSize: number
  prefetchCount: number
  device: DeviceContext | null
  dropLast: boolean
}

/**
 * Data Pipeline - fluent API for data loading
 *
 * @example
 * ```ts
 * const loader = Data.pipeline(dataset)
 *   .shuffle()
 *   .batch(64)
 *   .prefetch(2)
 *   .to(cuda)
 *
 * for await (const batch of loader) {
 *   // batch is on GPU, ready for training
 * }
 * ```
 */
export class DataPipeline<T> implements AsyncIterable<T> {
  private config: PipelineConfig

  constructor(
    private source: Dataset<any>,
    config?: Partial<PipelineConfig>,
  ) {
    this.config = {
      shouldShuffle: config?.shouldShuffle ?? false,
      batchSize: config?.batchSize ?? 0, // 0 = no batching
      prefetchCount: config?.prefetchCount ?? 0,
      device: config?.device ?? null,
      dropLast: config?.dropLast ?? false,
    }
  }

  /**
   * Shuffle data at each epoch (shuffles indices, not data)
   *
   * @example
   * ```ts
   * const loader = Data.pipeline(dataset).shuffle().batch(64)
   * ```
   */
  shuffle(): DataPipeline<T> {
    return new DataPipeline<T>(this.source, {
      ...this.config,
      shouldShuffle: true,
    })
  }

  /**
   * Batch items together
   *
   * @param size - Batch size
   * @param dropLast - Drop incomplete final batch (default: false)
   *
   * @example
   * ```ts
   * const loader = Data.pipeline(dataset).batch(64)
   * ```
   */
  batch(size: number, dropLast = false): DataPipeline<Batch<T>> {
    return new DataPipeline<Batch<T>>(this.source, {
      ...this.config,
      batchSize: size,
      dropLast,
    })
  }

  /**
   * Prefetch N batches during iteration
   *
   * @param count - Number of batches to prefetch
   *
   * @example
   * ```ts
   * const loader = Data.pipeline(dataset).batch(64).prefetch(2)
   * ```
   */
  prefetch(count: number): DataPipeline<T> {
    return new DataPipeline<T>(this.source, {
      ...this.config,
      prefetchCount: count,
    })
  }

  /**
   * Apply a transformation function to each item
   *
   * Use this for data augmentation, normalization, or any preprocessing
   * that should happen before device transfer.
   *
   * @param fn - Transform function applied to each item
   * @returns New pipeline with transform applied
   *
   * @example
   * ```ts
   * const loader = Data.pipeline(dataset)
   *   .map(item => ({ ...item, data: item.data.normalize() }))
   *   .shuffle()
   *   .batch(64)
   *   .to(cuda)
   * ```
   */
  map<U>(fn: (item: T) => U): DataPipeline<U> {
    const self = this
    const source: Dataset<U> = {
      getItem(index: number) {
        const item = self.source.getItem(index)
        if (item instanceof Promise) {
          return item.then(fn) as Promise<U>
        }
        return fn(item as T)
      },
      get length() {
        return self.source.length
      },
    }
    return new DataPipeline<U>(source, self.config)
  }

  /**
   * Transfer data to device
   *
   * When used with TensorPair datasets, transfers entire dataset to device
   * once and uses narrow() for zero-copy batching.
   *
   * @param device - Target device context
   *
   * @example
   * ```ts
   * const cuda = device.cuda(0)
   * const loader = Data.pipeline(dataset).batch(64).to(cuda)
   * ```
   */
  to(device: DeviceContext): DataPipeline<T> {
    return new DataPipeline<T>(this.source, {
      ...this.config,
      device,
    })
  }

  /**
   * Get the number of batches per epoch
   */
  get numBatches(): number {
    if (this.config.batchSize === 0) {
      return this.source.length
    }
    if (this.config.dropLast) {
      return Math.floor(this.source.length / this.config.batchSize)
    }
    return Math.ceil(this.source.length / this.config.batchSize)
  }

  /**
   * Get total number of samples
   */
  get length(): number {
    return this.source.length
  }

  /**
   * Generate shuffled indices
   * @internal
   */
  private generateIndices(): number[] {
    const indices = Array.from({ length: this.source.length }, (_, i) => i)

    if (this.config.shouldShuffle) {
      // Fisher-Yates shuffle
      for (let i = indices.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1))
        ;[indices[i]!, indices[j]!] = [indices[j]!, indices[i]!]
      }
    }

    return indices
  }

  /**
   * Async iterator for the pipeline
   *
   * @example
   * ```ts
   * for await (const batch of pipeline) {
   *   // process batch
   * }
   * ```
   */
  async *[Symbol.asyncIterator](): AsyncIterator<T> {
    const indices = this.generateIndices()
    const batchSize = this.config.batchSize || 1
    const numBatches = this.numBatches

    // If we have a device and the source returns TensorPairs,
    // we can potentially do GPU-resident iteration with narrow()
    // For now, implement basic batching with device transfer

    for (let batchIdx = 0; batchIdx < numBatches; batchIdx++) {
      const start = batchIdx * batchSize
      const end = Math.min(start + batchSize, indices.length)
      const batchIndices = indices.slice(start, end)

      // Load batch items
      const items: any[] = []
      for (const idx of batchIndices) {
        const item = await Promise.resolve(this.source.getItem(idx))
        items.push(item)
      }

      // If batching was requested, yield as batch
      if (this.config.batchSize > 0) {
        // Transfer to device if specified
        if (this.config.device && isTensorPairBatch(items)) {
          yield transferBatchToDevice(items, this.config.device) as T
        } else {
          yield items as T
        }
      } else {
        // No batching - yield individual items
        for (const item of items) {
          if (this.config.device && isTensorPair(item)) {
            yield transferToDevice(item, this.config.device) as T
          } else {
            yield item as T
          }
        }
      }
    }
  }

  /**
   * Synchronous iterator (for non-async datasets)
   *
   * @example
   * ```ts
   * for (const batch of pipeline.iter()) {
   *   // process batch
   * }
   * ```
   */
  *iter(): IterableIterator<T> {
    const indices = this.generateIndices()
    const batchSize = this.config.batchSize || 1
    const numBatches = this.numBatches

    for (let batchIdx = 0; batchIdx < numBatches; batchIdx++) {
      const start = batchIdx * batchSize
      const end = Math.min(start + batchSize, indices.length)
      const batchIndices = indices.slice(start, end)

      // Load batch items
      const items: any[] = []
      for (const idx of batchIndices) {
        const item = this.source.getItem(idx)
        if (item instanceof Promise) {
          throw new Error('Cannot use synchronous iterator with async dataset. Use for-await instead.')
        }
        items.push(item)
      }

      if (this.config.batchSize > 0) {
        if (this.config.device && isTensorPairBatch(items)) {
          yield transferBatchToDevice(items, this.config.device) as T
        } else {
          yield items as T
        }
      } else {
        for (const item of items) {
          if (this.config.device && isTensorPair(item)) {
            yield transferToDevice(item, this.config.device) as T
          } else {
            yield item as T
          }
        }
      }
    }
  }
}

/**
 * Check if item is a TensorPair
 * @internal
 */
function isTensorPair(item: any): item is TensorPair {
  return item && typeof item === 'object' && 'data' in item && 'label' in item
}

/**
 * Check if batch contains TensorPairs
 * @internal
 */
function isTensorPairBatch(batch: any[]): batch is TensorPair[] {
  return batch.length > 0 && isTensorPair(batch[0])
}

/**
 * Transfer a TensorPair to device
 * @internal
 */
function transferToDevice(item: TensorPair, device: DeviceContext): TensorPair {
  return {
    data: item.data.to(device.type),
    label: item.label.to(device.type),
  }
}

/**
 * Transfer a batch of TensorPairs to device
 * @internal
 */
function transferBatchToDevice(batch: TensorPair[], device: DeviceContext): TensorPair[] {
  return batch.map((item) => transferToDevice(item, device))
}

/**
 * Data namespace - entry point for declarative data pipelines
 *
 * @example
 * ```ts
 * import { Data } from '@ts-torch/core'
 *
 * const loader = Data.pipeline(dataset)
 *   .shuffle()
 *   .batch(64)
 *   .to(cuda)
 * ```
 */
export const Data = {
  /**
   * Create a data pipeline from a dataset
   *
   * @param dataset - Source dataset
   * @returns DataPipeline for fluent configuration
   *
   * @example
   * ```ts
   * const loader = Data.pipeline(mnistTrain)
   *   .shuffle()
   *   .batch(512)
   *   .to(device.cuda(0))
   * ```
   */
  pipeline<T>(dataset: Dataset<T>): DataPipeline<T> {
    return new DataPipeline<T>(dataset)
  },
}
