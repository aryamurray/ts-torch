/**
 * Fluent Data Pipeline
 *
 * Provides a declarative, chainable API for data loading.
 * Configuration is lazy - no work happens until iteration.
 *
 * Pipelines prepare data on CPU. Device transfer (CPU → GPU) happens
 * in the Trainer just-in-time before the forward pass.
 *
 * @example
 * ```ts
 * import { Data } from '@ts-torch/datasets'
 *
 * const pipeline = Data.pipeline(mnist)
 *   .shuffle()
 *   .batch(64)
 *   .map(b => ({ data: b.images, label: b.labelsTensor }))
 *
 * // Trainer handles CPU → GPU transfer internally
 * await trainer.fit(pipeline, { ... })
 * ```
 */

/**
 * A data source that can produce batches.
 * MNIST and similar datasets implement this pattern.
 */
export interface BatchableDataset<TBatch> {
  batches(batchSize: number, shuffle?: boolean): Generator<TBatch> | AsyncGenerator<TBatch>
  length: number
}

/**
 * Pipeline configuration - all lazy until iteration
 */
interface PipelineConfig<TBatch> {
  source: BatchableDataset<TBatch>
  batchSize: number
  shuffle: boolean
  transforms: Array<(batch: any) => any>
}

/**
 * Fluent data pipeline - lazy configuration for data loading
 */
export class Pipeline<TOut> implements AsyncIterable<TOut> {
  private constructor(private readonly config: PipelineConfig<any>) {}

  /**
   * Create a pipeline from a batchable dataset
   */
  static from<TBatch>(source: BatchableDataset<TBatch>): Pipeline<TBatch> {
    return new Pipeline<TBatch>({
      source,
      batchSize: 32,
      shuffle: false,
      transforms: [],
    })
  }

  /**
   * Enable shuffling
   */
  shuffle(): Pipeline<TOut> {
    return new Pipeline({
      ...this.config,
      shuffle: true,
    })
  }

  /**
   * Set batch size
   */
  batch(size: number): Pipeline<TOut> {
    return new Pipeline({
      ...this.config,
      batchSize: size,
    })
  }

  /**
   * Transform each batch
   *
   * @example
   * ```ts
   * pipeline.map(batch => ({
   *   data: batch.images,
   *   label: batch.labelsTensor
   * }))
   * ```
   */
  map<TNext>(fn: (batch: TOut) => TNext): Pipeline<TNext> {
    return new Pipeline({
      ...this.config,
      transforms: [...this.config.transforms, fn],
    })
  }

  /**
   * Iterate over the pipeline (async)
   */
  async *[Symbol.asyncIterator](): AsyncIterator<TOut> {
    const { source, batchSize, shuffle, transforms } = this.config
    const batches = source.batches(batchSize, shuffle)

    // Handle both sync and async generators
    for await (const batch of toAsyncIterable(batches)) {
      let result: any = batch

      // Apply transforms
      for (const transform of transforms) {
        result = transform(result)
      }

      yield result as TOut
    }
  }
}

/**
 * Convert a sync or async iterable to an async iterable
 */
async function* toAsyncIterable<T>(
  iterable: Generator<T> | AsyncGenerator<T> | Iterable<T> | AsyncIterable<T>,
): AsyncGenerator<T> {
  if (Symbol.asyncIterator in iterable) {
    yield* iterable as AsyncIterable<T>
  } else {
    for (const item of iterable as Iterable<T>) {
      yield item
    }
  }
}

/**
 * Data namespace - entry point for fluent data loading
 *
 * @example
 * ```ts
 * import { Data } from '@ts-torch/datasets'
 *
 * const trainLoader = Data.pipeline(mnist)
 *   .shuffle()
 *   .batch(64)
 *   .map(b => ({ data: b.images, label: b.labelsTensor }))
 *
 * // Trainer handles device transfer
 * await trainer.fit(trainLoader, { ... })
 * ```
 */
export const Data = {
  /**
   * Create a fluent pipeline from a batchable dataset
   */
  pipeline<TBatch>(source: BatchableDataset<TBatch>): Pipeline<TBatch> {
    return Pipeline.from(source)
  },
}
