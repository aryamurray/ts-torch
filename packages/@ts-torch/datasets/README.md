# @ts-torch/datasets

Dataset loaders and utilities for ts-torch.

## Overview

This package provides dataset loading utilities, data transformations, and common dataset implementations for machine learning with ts-torch. It includes popular vision and text datasets, along with flexible data loading and transformation pipelines.

## Features

- **Base Dataset Classes**: Abstract interfaces for custom datasets
- **DataLoader**: Efficient batching, shuffling, and parallel loading
- **Transforms**: Common data transformations (normalization, augmentation, etc.)
- **Vision Datasets**: MNIST, CIFAR-10/100, ImageFolder
- **Text Datasets**: Text classification utilities
- **Data Splitting**: Train/test split utilities

## Installation

```bash
bun add @ts-torch/datasets
```

## Usage

### Using Built-in Datasets

```typescript
import { MNIST, DataLoader } from "@ts-torch/datasets";

// Load MNIST dataset
const dataset = new MNIST("./data", true, undefined, true);
await dataset.init();

// Create data loader
const loader = new DataLoader(dataset, {
  batchSize: 32,
  shuffle: true,
});

// Iterate over batches
for await (const batch of loader) {
  console.log("Batch size:", batch.length);
}
```

### Custom Datasets

```typescript
import { BaseDataset } from "@ts-torch/datasets";
import type { Tensor } from "@ts-torch/core";

class MyDataset extends BaseDataset<[Tensor, number]> {
  getItem(index: number): [Tensor, number] {
    // Load and return your data
    const data = loadData(index);
    const label = loadLabel(index);
    return [data, label];
  }

  get length(): number {
    return 1000; // Total number of samples
  }
}
```

### Data Transformations

```typescript
import { Compose, Normalize, RandomHorizontalFlip, Resize } from "@ts-torch/datasets";

const transform = new Compose([
  new Resize([224, 224]),
  new RandomHorizontalFlip(0.5),
  new Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
]);

// Apply to dataset
const dataset = new ImageFolder("./data/train", transform);
```

### DataLoader

```typescript
import { DataLoader } from "@ts-torch/datasets";

const loader = new DataLoader(dataset, {
  batchSize: 64,
  shuffle: true,
  drop_last: false,
  numWorkers: 4,
});

console.log("Number of batches:", loader.numBatches);

// Async iteration
for await (const batch of loader) {
  // Process batch
}
```

### Train/Test Split

```typescript
const [trainSet, testSet] = dataset.split(0.8); // 80% train, 20% test

const trainLoader = new DataLoader(trainSet, { batchSize: 32, shuffle: true });
const testLoader = new DataLoader(testSet, { batchSize: 32, shuffle: false });
```
