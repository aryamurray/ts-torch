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
import { cat } from '../tensor/factory.js'
import type { DeviceContext } from '../device/context.js'
import type { DeviceType } from '../types/tensor.js'

/**
 * Dataset interface - source of items
 */
export interface Dataset<T> {
  getItem(index: number): T | Promise<T>
  readonly length: number
}

/**
 * Batchable dataset interface - supports efficient batch loading
 *
 * Datasets implementing this interface can load multiple items at once,
 * avoiding the "death by a thousand allocations" problem where individual
 * items are loaded and then concatenated.
 *
 * @example
 * ```ts
 * class MyDataset implements BatchableDataset<TensorPair> {
 *   getItem(index: number) { ... }
 *   getBatch(indices: number[]) {
 *     // Load all items efficiently in one operation
 *     return { data: batchedData, label: batchedLabels }
 *   }
 *   get length() { return this._length }
 * }
 * ```
 */
export interface BatchableDataset<T> extends Dataset<T> {
  /**
   * Load multiple items at once, returning a single batched result.
   * For TensorPair datasets, this should return tensors with shape [batchSize, ...itemShape]
   */
  getBatch(indices: number[]): T | Promise<T>
}

/**
 * Tensor pair - common for supervised learning
 * @template Dev - Device type where tensors are stored
 */
export type TensorPair<Dev extends DeviceType = DeviceType> = { data: Tensor<any, any, Dev>; label: Tensor<any, any, Dev> }

/**
 * Utility type to update device type in TensorPair
 * @internal
 */
export type WithDevice<T, Dev extends DeviceType> = T extends TensorPair<any> ? TensorPair<Dev> : T

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
 * Provides efficient data loading with batching, shuffling, and device transfer.
 * When used with TensorPair datasets, the `.to()` method updates both the data
 * and the type to reflect the target device.
 *
 * @template T - Item type yielded by the pipeline. For supervised learning,
 *               typically TensorPair<Dev> where Dev is the device type.
 *
 * @remarks
 * The `.to(device)` method uses the WithDevice utility type to correctly
 * update TensorPair's device type parameter.
 *
 * @example
 * ```ts
 * const cuda = device.cuda(0)
 *
 * // loader type: DataPipeline<TensorPair<'cuda'>>
 * const loader = Data.pipeline(dataset)  // DataPipeline<TensorPair<'cpu'>>
 *   .shuffle()
 *   .batch(64)
 *   .to(cuda)  // Type updated to TensorPair<'cuda'>
 *
 * for await (const batch of loader) {
 *   // batch.data and batch.label are on CUDA
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
   * @template TargetDev - Target device type
   * @param device - Target device context
   * @returns Pipeline yielding items on the target device
   *
   * @example
   * ```ts
   * const cuda = device.cuda(0)
   * // loader type is DataPipeline<TensorPair<'cuda'>>
   * const loader = Data.pipeline(dataset).batch(64).to(cuda)
   * ```
   */
  to<TargetDev extends DeviceType>(device: DeviceContext<TargetDev>): DataPipeline<WithDevice<T, TargetDev>> {
    return new DataPipeline<WithDevice<T, TargetDev>>(this.source, {
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

    // Check if source supports efficient batch loading
    const hasBatchSupport = isBatchableDataset(this.source)

    for (let batchIdx = 0; batchIdx < numBatches; batchIdx++) {
      const start = batchIdx * batchSize
      const end = Math.min(start + batchSize, indices.length)
      const batchIndices = indices.slice(start, end)

      // If batching was requested
      if (this.config.batchSize > 0) {
        // FAST PATH: Use getBatch if available (avoids N tensor allocations)
        if (hasBatchSupport) {
          const batchableSource = this.source as BatchableDataset<any>
          const batch = await Promise.resolve(batchableSource.getBatch(batchIndices))

          // Transfer to device if specified
          if (this.config.device && isTensorPair(batch)) {
            yield transferToDevice(batch as TensorPair, this.config.device.type) as T
          } else {
            yield batch as T
          }
        } else {
          // SLOW PATH: Load items one-by-one and concatenate
          const items: any[] = []
          for (const idx of batchIndices) {
            const item = await Promise.resolve(this.source.getItem(idx))
            items.push(item)
          }

          // Transfer to device if specified - concatenate on CPU first for efficiency
          if (this.config.device && isTensorPairBatch(items)) {
            // Yields single pre-concatenated TensorPair (not array)
            yield concatenateAndTransferBatch(items, this.config.device.type) as T
          } else {
            yield items as T
          }
        }
      } else {
        // No batching - yield individual items
        for (const idx of batchIndices) {
          const item = await Promise.resolve(this.source.getItem(idx))
          if (this.config.device && isTensorPair(item)) {
            yield transferToDevice(item, this.config.device.type) as T
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

    // Check if source supports efficient batch loading
    const hasBatchSupport = isBatchableDataset(this.source)

    for (let batchIdx = 0; batchIdx < numBatches; batchIdx++) {
      const start = batchIdx * batchSize
      const end = Math.min(start + batchSize, indices.length)
      const batchIndices = indices.slice(start, end)

      if (this.config.batchSize > 0) {
        // FAST PATH: Use getBatch if available
        if (hasBatchSupport) {
          const batchableSource = this.source as BatchableDataset<any>
          const batch = batchableSource.getBatch(batchIndices)
          if (batch instanceof Promise) {
            throw new Error('Cannot use synchronous iterator with async dataset. Use for-await instead.')
          }

          if (this.config.device && isTensorPair(batch)) {
            yield transferToDevice(batch as TensorPair, this.config.device.type) as T
          } else {
            yield batch as T
          }
        } else {
          // SLOW PATH: Load items one-by-one
          const items: any[] = []
          for (const idx of batchIndices) {
            const item = this.source.getItem(idx)
            if (item instanceof Promise) {
              throw new Error('Cannot use synchronous iterator with async dataset. Use for-await instead.')
            }
            items.push(item)
          }

          if (this.config.device && isTensorPairBatch(items)) {
            yield concatenateAndTransferBatch(items, this.config.device.type) as T
          } else {
            yield items as T
          }
        }
      } else {
        // No batching - yield individual items
        for (const idx of batchIndices) {
          const item = this.source.getItem(idx)
          if (item instanceof Promise) {
            throw new Error('Cannot use synchronous iterator with async dataset. Use for-await instead.')
          }
          if (this.config.device && isTensorPair(item)) {
            yield transferToDevice(item, this.config.device.type) as T
          } else {
            yield item as T
          }
        }
      }
    }
  }
}

/**
 * Check if dataset supports batch loading
 * @internal
 */
function isBatchableDataset(dataset: any): dataset is BatchableDataset<any> {
  return dataset && typeof dataset.getBatch === 'function'
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
 * Transfer a TensorPair to device and FREE the source tensors
 *
 * Uses move semantics to prevent memory leaks - the original CPU
 * tensors are freed after transfer to the target device.
 *
 * @internal
 */
function transferToDevice<Dev extends DeviceType>(item: TensorPair<DeviceType>, targetDevice: Dev): TensorPair<Dev> {
  return {
    data: (item.data as any).move(targetDevice),
    label: (item.label as any).move(targetDevice),
  }
}

/**
 * Concatenate and transfer a batch of TensorPairs to device
 *
 * OPTIMIZATION: Instead of transferring each item individually and then
 * concatenating on GPU (expensive), we concatenate on CPU first and
 * transfer a single batched tensor to GPU.
 *
 * @internal
 */
function concatenateAndTransferBatch<Dev extends DeviceType>(
  batch: TensorPair<DeviceType>[],
  targetDevice: Dev,
): TensorPair<Dev> {
  // Extract all data and label tensors (still on CPU)
  const dataTensors = batch.map((item) => item.data) as Tensor<any, any, any>[]
  const labelTensors = batch.map((item) => item.label) as Tensor<any, any, any>[]

  // Concatenate on CPU (fast, no GPU overhead)
  const batchedData = cat(dataTensors, 0)
  const batchedLabels = cat(labelTensors, 0)

  // Free individual CPU tensors after concatenation
  for (const item of batch) {
    item.data?.free?.()
    item.label?.free?.()
  }

  // Transfer single batched tensors to GPU (one transfer each)
  return {
    data: (batchedData as any).move(targetDevice),
    label: (batchedLabels as any).move(targetDevice),
  }
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
