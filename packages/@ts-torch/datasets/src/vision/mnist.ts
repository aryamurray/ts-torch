/**
 * MNIST dataset
 */

import type { Tensor } from '@ts-torch/core';
import { BaseDataset } from '../dataset.js';
import type { Transform } from '../transforms.js';

/**
 * MNIST handwritten digits dataset
 *
 * The MNIST database consists of 70,000 28x28 grayscale images of handwritten digits.
 * - Training set: 60,000 images
 * - Test set: 10,000 images
 */
export class MNIST extends BaseDataset<[Tensor, number]> {
  private data: Tensor | null = null;

  constructor(
    _root: string,
    private train: boolean = true,
    _transform?: Transform<Tensor, Tensor>,
    _download: boolean = false
  ) {
    super();
  }

  async init(): Promise<void> {
    // TODO: Download and load MNIST data
    // if (this.download) {
    //   await this.downloadData();
    // }
    // this.loadData();
  }

  getItem(index: number): [Tensor, number] {
    if (!this.data) {
      throw new Error('MNIST dataset not initialized. Call init() first.');
    }

    if (index < 0 || index >= this.length) {
      throw new Error(`Index ${index} out of bounds`);
    }

    // TODO: Get image and apply transform
    // let image = this.data[index];
    // if (this.transform) {
    //   image = await this.transform(image);
    // }
    // return [image, this.targets[index]];

    throw new Error('MNIST.getItem not yet implemented');
  }

  get length(): number {
    return this.train ? 60000 : 10000;
  }

  get classes(): string[] {
    return ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];
  }
}
