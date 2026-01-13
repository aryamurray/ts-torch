/**
 * Worker-based DataLoader for parallel data loading
 *
 * Uses worker threads to load and preprocess data in parallel,
 * preventing the main thread from blocking on data loading.
 */

import { cpus } from 'node:os'
import { BatchAssembler } from './batch-assembler.js'
import { RingBuffer } from './ring-buffer.js'
import type {
  TensorPair,
  TransformConfig,
  WorkerConfig,
  WorkerDataLoaderOptions,
} from './types.js'
import { WorkerPool } from './worker-pool.js'

/** Default bytes per batch slot (16MB) */
const DEFAULT_MAX_BATCH_BYTES = 16 * 1024 * 1024

/** Default prefetch count */
const DEFAULT_PREFETCH_COUNT = 2

/** Default worker timeout (30 seconds) */
const DEFAULT_WORKER_TIMEOUT = 30000

/**
 * Configuration for creating the dataset in workers
 */
export interface DatasetFactory {
  /** Module path to import (must be an absolute path or package name) */
  module: string

  /** Export name of the factory function */
  exportName: string

  /** Arguments to pass to the factory */
  args: unknown[]
}

/**
 * Options for WorkerDataLoader with dataset factory
 */
export interface WorkerDataLoaderConfig extends Omit<WorkerDataLoaderOptions, 'transforms'> {
  /** Factory for creating the dataset in workers */
  datasetFactory: DatasetFactory

  /** Transform configurations (serializable) */
  transforms?: TransformConfig[]
}

/**
 * Multi-threaded data loader using worker threads
 *
 * @example
 * ```typescript
 * const loader = new WorkerDataLoader({
 *   datasetFactory: {
 *     module: './my-dataset.js',
 *     exportName: 'createDataset',
 *     args: ['/path/to/data']
 *   },
 *   batchSize: 64,
 *   numWorkers: 4,
 *   prefetchCount: 2,
 *   shuffle: true,
 *   transforms: [
 *     { type: 'resize', params: { size: [224, 224] } },
 *     { type: 'normalize', params: { mean: [0.485], std: [0.229] } }
 *   ]
 * })
 *
 * for await (const batch of loader) {
 *   // batch.data and batch.label are tensors
 * }
 *
 * await loader.shutdown()
 * ```
 */
export class WorkerDataLoader implements AsyncIterable<TensorPair> {
  private pool: WorkerPool | null = null
  private ringBuffer: RingBuffer
  private assembler: BatchAssembler
  private config: Required<
    Pick<WorkerDataLoaderConfig, 'numWorkers' | 'prefetchCount' | 'batchSize'>
  > &
    WorkerDataLoaderConfig

  private datasetLength: number
  private currentEpoch = 0
  private isInitialized = false
  private initPromise: Promise<void> | null = null

  constructor(
    /** Length of the dataset (must be known upfront) */
    datasetLength: number,
    options: WorkerDataLoaderConfig,
  ) {
    this.datasetLength = datasetLength

    // Apply defaults
    this.config = {
      numWorkers: options.numWorkers ?? cpus().length,
      prefetchCount: options.prefetchCount ?? DEFAULT_PREFETCH_COUNT,
      batchSize: options.batchSize,
      shuffle: options.shuffle ?? false,
      dropLast: options.dropLast ?? false,
      maxBatchBytes: options.maxBatchBytes ?? DEFAULT_MAX_BATCH_BYTES,
      workerTimeout: options.workerTimeout ?? DEFAULT_WORKER_TIMEOUT,
      datasetFactory: options.datasetFactory,
      transforms: options.transforms ?? [],
    }

    // Validate
    if (this.config.batchSize < 1) {
      throw new Error('batchSize must be at least 1')
    }
    if (this.config.numWorkers < 1) {
      throw new Error('numWorkers must be at least 1')
    }
    if (this.config.prefetchCount < 1) {
      throw new Error('prefetchCount must be at least 1')
    }

    // Create ring buffer
    this.ringBuffer = new RingBuffer(this.config.prefetchCount, this.config.maxBatchBytes!)

    // Create batch assembler
    this.assembler = new BatchAssembler()
  }

  /**
   * Initialize the worker pool
   */
  private async initialize(): Promise<void> {
    if (this.isInitialized) return
    if (this.initPromise) return this.initPromise

    this.initPromise = (async () => {
      // Get path to worker script
      const workerPath = new URL('./worker.js', import.meta.url)

      // Create worker config
      const workerConfig: WorkerConfig = {
        datasetModule: this.config.datasetFactory.module,
        datasetExport: this.config.datasetFactory.exportName,
        datasetArgs: this.config.datasetFactory.args,
        transforms: this.config.transforms ?? [],
      }

      // Create worker pool
      this.pool = new WorkerPool(this.config.numWorkers, workerConfig, workerPath)

      // Wait for workers to be ready
      await this.pool.waitForReady()

      this.isInitialized = true
    })()

    return this.initPromise
  }

  /**
   * Get the number of batches per epoch
   */
  get numBatches(): number {
    if (this.config.dropLast) {
      return Math.floor(this.datasetLength / this.config.batchSize)
    }
    return Math.ceil(this.datasetLength / this.config.batchSize)
  }

  /**
   * Generate indices for an epoch
   */
  private generateIndices(): number[] {
    const indices = Array.from({ length: this.datasetLength }, (_, i) => i)

    if (this.config.shuffle) {
      // Fisher-Yates shuffle
      for (let i = indices.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1))
        const temp = indices[i]!
        indices[i] = indices[j]!
        indices[j] = temp
      }
    }

    return indices
  }

  /**
   * Async iterator implementation
   */
  async *[Symbol.asyncIterator](): AsyncIterator<TensorPair> {
    // Ensure initialized
    await this.initialize()

    if (!this.pool) {
      throw new Error('Worker pool not initialized')
    }

    // Start new epoch
    this.currentEpoch++
    const indices = this.generateIndices()
    const numBatches = this.numBatches

    // Reset ring buffer
    this.ringBuffer.reset()

    // Track batch scheduling
    let nextBatchToSchedule = 0
    let nextBatchToYield = 0
    const pendingBatches = new Map<number, Promise<void>>()

    // Schedule initial prefetch batches
    const scheduleNext = async (): Promise<void> => {
      while (
        nextBatchToSchedule < numBatches &&
        nextBatchToSchedule - nextBatchToYield < this.config.prefetchCount
      ) {
        const batchIdx = nextBatchToSchedule++
        const start = batchIdx * this.config.batchSize
        const end = Math.min(start + this.config.batchSize, indices.length)
        const batchIndices = indices.slice(start, end)

        // Acquire a slot
        const slot = await this.ringBuffer.acquireWriteSlot()

        // Submit task to worker
        const taskPromise = this.pool!.submitTask(
          {
            indices: batchIndices,
            slotIndex: slot.index,
            byteOffset: slot.byteOffset,
            maxBytes: slot.maxBytes,
          },
          this.ringBuffer.getBuffer(),
        ).then((metadata) => {
          // Mark slot as ready when worker completes
          this.ringBuffer.markReady(slot.index, metadata)
        })

        pendingBatches.set(batchIdx, taskPromise)
      }
    }

    // Schedule initial batches
    await scheduleNext()

    // Yield batches in order
    for (let batchIdx = 0; batchIdx < numBatches; batchIdx++) {
      // Wait for this batch's slot to be ready
      const slot = await this.ringBuffer.waitForReady(batchIdx)

      // Assemble tensors from buffer
      const batch = this.assembler.assemble(slot)

      // Release the slot
      this.ringBuffer.release(slot.index)
      nextBatchToYield++

      // Schedule more batches
      await scheduleNext()

      // Yield the batch
      yield batch
    }

    // Wait for any remaining pending batches
    await Promise.all(pendingBatches.values())
  }

  /**
   * Shutdown the data loader and terminate workers
   */
  async shutdown(): Promise<void> {
    if (this.pool) {
      await this.pool.shutdown()
      this.pool = null
    }
    this.isInitialized = false
    this.initPromise = null
  }

  /**
   * Get number of workers
   */
  get workerCount(): number {
    return this.config.numWorkers
  }
}

/**
 * Create a WorkerDataLoader from an existing dataset
 *
 * Note: This requires the dataset to be createable in workers via a factory function.
 * The dataset itself cannot be passed directly since worker threads cannot share objects.
 */
export function createWorkerDataLoader(
  datasetLength: number,
  options: WorkerDataLoaderConfig,
): WorkerDataLoader {
  return new WorkerDataLoader(datasetLength, options)
}
