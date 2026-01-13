/**
 * DataLoader for batching and iterating over datasets
 */

import { validateDataLoaderParams } from '@ts-torch/core'
import type { Dataset } from './dataset.js'

/**
 * DataLoader options
 */
export interface DataLoaderOptions {
  batchSize?: number
  shuffle?: boolean
  drop_last?: boolean
  numWorkers?: number
}

/**
 * DataLoader for iterating over a dataset in batches
 *
 * Provides batching, shuffling, and parallel loading capabilities.
 */
export class DataLoader<T = unknown> implements AsyncIterable<T[]> {
  private batchSize: number
  private shuffle: boolean
  private dropLast: boolean

  constructor(
    private dataset: Dataset<T>,
    options: DataLoaderOptions = {},
  ) {
    this.batchSize = options.batchSize ?? 1
    this.shuffle = options.shuffle ?? false
    this.dropLast = options.drop_last ?? false

    validateDataLoaderParams({
      batchSize: this.batchSize,
      shuffle: this.shuffle,
      dropLast: this.dropLast,
    })
  }

  /**
   * Get the number of batches
   */
  get numBatches(): number {
    if (this.dropLast) {
      return Math.floor(this.dataset.length / this.batchSize)
    }
    return Math.ceil(this.dataset.length / this.batchSize)
  }

  /**
   * Generate indices for iteration
   */
  private getIndices(): number[] {
    const indices = Array.from({ length: this.dataset.length }, (_, i) => i)

    if (this.shuffle) {
      // Fisher-Yates shuffle
      for (let i = indices.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1))
        const temp = indices[i]
        const swapVal = indices[j]
        if (temp !== undefined && swapVal !== undefined) {
          indices[i] = swapVal
          indices[j] = temp
        }
      }
    }

    return indices
  }

  /**
   * Async iterator for batches
   */
  async *[Symbol.asyncIterator](): AsyncIterator<T[]> {
    const indices = this.getIndices()
    const numBatches = this.numBatches

    for (let i = 0; i < numBatches; i++) {
      const start = i * this.batchSize
      const end = Math.min(start + this.batchSize, indices.length)
      const batchIndices = indices.slice(start, end)

      // Load batch samples
      const batch: T[] = []
      for (const idx of batchIndices) {
        try {
          const sample = await Promise.resolve(this.dataset.getItem(idx))
          batch.push(sample)
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : String(error)
          throw new Error(
            `Failed to load sample at index ${idx}: ${errorMessage}`,
          )
        }
      }

      yield batch
    }
  }

  /**
   * Synchronous iterator (for sync datasets)
   */
  *iter(): IterableIterator<T[]> {
    const indices = this.getIndices()
    const numBatches = this.numBatches

    for (let i = 0; i < numBatches; i++) {
      const start = i * this.batchSize
      const end = Math.min(start + this.batchSize, indices.length)
      const batchIndices = indices.slice(start, end)

      // Load batch samples
      const batch: T[] = []
      for (const idx of batchIndices) {
        const sample = this.dataset.getItem(idx)
        if (sample instanceof Promise) {
          throw new Error('Cannot use synchronous iterator with async dataset')
        }
        batch.push(sample)
      }

      yield batch
    }
  }
}
