/**
 * Base Dataset and utility types
 */

import type { Tensor } from '@ts-torch/core'

/**
 * Base interface for all datasets
 *
 * Datasets represent a collection of data samples that can be accessed by index.
 */
export interface Dataset<T = Tensor> {
  /**
   * Get a data sample by index
   */
  getItem(index: number): T | Promise<T>

  /**
   * Get the total number of samples in the dataset
   */
  get length(): number
}

/**
 * Abstract base class for datasets
 */
export abstract class BaseDataset<T = Tensor> implements Dataset<T> {
  abstract getItem(index: number): T | Promise<T>
  abstract get length(): number

  /**
   * Create an iterable interface for the dataset
   */
  *[Symbol.iterator](): Iterator<T> {
    for (let i = 0; i < this.length; i++) {
      const item = this.getItem(i)
      if (item instanceof Promise) {
        throw new Error('Cannot iterate over async dataset synchronously. Use DataLoader instead.')
      }
      yield item
    }
  }

  /**
   * Get a subset of the dataset
   */
  subset(indices: number[]): SubsetDataset<T> {
    return new SubsetDataset(this, indices)
  }

  /**
   * Split dataset into train/test sets
   */
  split(ratio: number): [SubsetDataset<T>, SubsetDataset<T>] {
    const trainSize = Math.floor(this.length * ratio)
    const indices = Array.from({ length: this.length }, (_, i) => i)

    // Shuffle indices
    for (let i = indices.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1))
      ;[indices[i]!, indices[j]!] = [indices[j]!, indices[i]!]
    }

    const trainIndices = indices.slice(0, trainSize)
    const testIndices = indices.slice(trainSize)

    return [new SubsetDataset(this, trainIndices), new SubsetDataset(this, testIndices)]
  }
}

/**
 * Subset of a dataset
 */
export class SubsetDataset<T = Tensor> extends BaseDataset<T> {
  constructor(
    private dataset: Dataset<T>,
    private indices: number[],
  ) {
    super()
  }

  getItem(index: number): T | Promise<T> {
    const actualIndex = this.indices[index]
    if (actualIndex === undefined) {
      throw new Error(`Index ${index} out of bounds for dataset of length ${this.length}`)
    }
    return this.dataset.getItem(actualIndex)
  }

  get length(): number {
    return this.indices.length
  }
}

/**
 * In-memory dataset from arrays
 */
export class TensorDataset extends BaseDataset<Tensor[]> {
  constructor(private tensors: Tensor[]) {
    super()

    // Validate all tensors have the same first dimension
    if (tensors.length === 0) {
      throw new Error('TensorDataset requires at least one tensor')
    }

    const firstDim = tensors[0]?.shape[0]
    for (const tensor of tensors) {
      if (tensor.shape[0] !== firstDim) {
        throw new Error('All tensors must have the same size in the first dimension')
      }
    }
  }

  getItem(index: number): Tensor[] {
    if (index < 0 || index >= this.length) {
      throw new Error(`Index ${index} out of bounds for dataset of length ${this.length}`)
    }

    // TODO: Implement tensor indexing
    // return this.tensors.map(tensor => tensor[index]);
    return this.tensors
  }

  get length(): number {
    return this.tensors[0]?.shape[0] ?? 0
  }
}
