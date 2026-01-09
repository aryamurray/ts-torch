/**
 * CIFAR-10 and CIFAR-100 datasets
 */

import type { Tensor } from "@ts-torch/core";
import { BaseDataset } from "../dataset.js";
import type { Transform } from "../transforms.js";

/**
 * CIFAR-10 dataset
 *
 * The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes.
 * - Training set: 50,000 images
 * - Test set: 10,000 images
 */
export class CIFAR10 extends BaseDataset<[Tensor, number]> {
  private data: Tensor | null = null;

  constructor(
    _root: string,
    private train: boolean = true,
    _transform?: Transform<Tensor, Tensor>,
    _download: boolean = false,
  ) {
    super();
  }

  async init(): Promise<void> {
    // TODO: Download and load CIFAR-10 data
  }

  getItem(index: number): [Tensor, number] {
    if (!this.data) {
      throw new Error("CIFAR10 dataset not initialized. Call init() first.");
    }

    if (index < 0 || index >= this.length) {
      throw new Error(`Index ${index} out of bounds`);
    }

    throw new Error("CIFAR10.getItem not yet implemented");
  }

  get length(): number {
    return this.train ? 50000 : 10000;
  }

  get classes(): string[] {
    return [
      "airplane",
      "automobile",
      "bird",
      "cat",
      "deer",
      "dog",
      "frog",
      "horse",
      "ship",
      "truck",
    ];
  }
}

/**
 * CIFAR-100 dataset
 *
 * The CIFAR-100 dataset is similar to CIFAR-10 but has 100 classes.
 * Each class contains 600 images (500 training, 100 test).
 */
export class CIFAR100 extends BaseDataset<[Tensor, number]> {
  private fineLabels: boolean;

  constructor(
    _root: string,
    private train: boolean = true,
    _transform?: Transform<Tensor, Tensor>,
    _download: boolean = false,
    fineLabels: boolean = true,
  ) {
    super();
    this.fineLabels = fineLabels;
  }

  async init(): Promise<void> {
    // TODO: Download and load CIFAR-100 data
  }

  getItem(_index: number): [Tensor, number] {
    throw new Error("CIFAR100.getItem not yet implemented");
  }

  get length(): number {
    return this.train ? 50000 : 10000;
  }

  get numClasses(): number {
    return this.fineLabels ? 100 : 20;
  }
}
