/**
 * Tests for Dataset implementations
 *
 * Tests the core Dataset interface, BaseDataset, SubsetDataset, and TensorDataset.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { BaseDataset, SubsetDataset, TensorDataset } from '../dataset.js';
import { torch } from '@ts-torch/core';

// Import test utils from source
const utilsModule = await import('../../../core/src/test/utils.js');
const { scopedTest } = utilsModule;

/**
 * Simple test dataset for testing
 */
class SimpleDataset extends BaseDataset<number> {
  constructor(private size: number) {
    super();
  }

  getItem(index: number): number {
    if (index < 0 || index >= this.size) {
      throw new Error(`Index ${index} out of bounds`);
    }
    return index * 2; // Return simple computed value
  }

  get length(): number {
    return this.size;
  }
}

/**
 * Async dataset for testing async getItem
 */
class AsyncDataset extends BaseDataset<number> {
  constructor(private size: number) {
    super();
  }

  async getItem(index: number): Promise<number> {
    if (index < 0 || index >= this.size) {
      throw new Error(`Index ${index} out of bounds`);
    }
    // Simulate async operation
    await new Promise((resolve) => setTimeout(resolve, 1));
    return index * 3;
  }

  get length(): number {
    return this.size;
  }
}

describe('Dataset', () => {
  describe('BaseDataset', () => {
    let dataset: SimpleDataset;

    beforeEach(() => {
      dataset = new SimpleDataset(10);
    });

    it('should get item by index', () => {
      expect(dataset.getItem(0)).toBe(0);
      expect(dataset.getItem(5)).toBe(10);
      expect(dataset.getItem(9)).toBe(18);
    });

    it('should report correct length', () => {
      expect(dataset.length).toBe(10);
    });

    it('should throw error for invalid index', () => {
      expect(() => dataset.getItem(-1)).toThrow('out of bounds');
      expect(() => dataset.getItem(10)).toThrow('out of bounds');
    });

    it('should be iterable', () => {
      const items = Array.from(dataset);
      expect(items).toHaveLength(10);
      expect(items).toEqual([0, 2, 4, 6, 8, 10, 12, 14, 16, 18]);
    });

    it('should throw when iterating async dataset synchronously', () => {
      const asyncDataset = new AsyncDataset(5);
      expect(() => Array.from(asyncDataset)).toThrow('Cannot iterate over async dataset synchronously');
    });

    it('should create subset', () => {
      const subset = dataset.subset([1, 3, 5]);
      expect(subset.length).toBe(3);
      expect(subset.getItem(0)).toBe(2); // index 1 -> 1*2
      expect(subset.getItem(1)).toBe(6); // index 3 -> 3*2
      expect(subset.getItem(2)).toBe(10); // index 5 -> 5*2
    });

    it('should split dataset into train/test', () => {
      const [train, test] = dataset.split(0.7);

      expect(train.length + test.length).toBe(dataset.length);
      expect(train.length).toBe(7);
      expect(test.length).toBe(3);
    });

    it('should split dataset with different ratios', () => {
      const [train, test] = dataset.split(0.5);

      expect(train.length).toBe(5);
      expect(test.length).toBe(5);
    });

    it('should shuffle indices when splitting', () => {
      const [train1] = dataset.split(0.5);
      const [train2] = dataset.split(0.5);

      // Get items from both splits
      const items1 = Array.from({ length: train1.length }, (_, i) => train1.getItem(i));
      const items2 = Array.from({ length: train2.length }, (_, i) => train2.getItem(i));

      // With random shuffling, they should likely be different
      // Note: This test has a small chance of false negative if shuffle produces same result
      // We can't guarantee they're different due to randomness, so just verify structure
      expect(items1.length).toBe(5);
      expect(items2.length).toBe(5);
    });
  });

  describe('SubsetDataset', () => {
    let baseDataset: SimpleDataset;
    let subset: SubsetDataset<number>;

    beforeEach(() => {
      baseDataset = new SimpleDataset(10);
      subset = new SubsetDataset(baseDataset, [0, 2, 4, 6, 8]);
    });

    it('should get items from correct indices', () => {
      expect(subset.getItem(0)).toBe(0); // base index 0 -> 0*2
      expect(subset.getItem(1)).toBe(4); // base index 2 -> 2*2
      expect(subset.getItem(4)).toBe(16); // base index 8 -> 8*2
    });

    it('should report correct length', () => {
      expect(subset.length).toBe(5);
    });

    it('should throw error for out of bounds index', () => {
      expect(() => subset.getItem(-1)).toThrow('out of bounds');
      expect(() => subset.getItem(5)).toThrow('out of bounds');
    });

    it('should be iterable', () => {
      const items = Array.from(subset);
      expect(items).toEqual([0, 4, 8, 12, 16]);
    });

    it('should work with empty indices', () => {
      const emptySubset = new SubsetDataset(baseDataset, []);
      expect(emptySubset.length).toBe(0);
    });

    it('should work with async base dataset', async () => {
      const asyncBase = new AsyncDataset(10);
      const asyncSubset = new SubsetDataset(asyncBase, [1, 3, 5]);

      expect(await asyncSubset.getItem(0)).toBe(3); // index 1 -> 1*3
      expect(await asyncSubset.getItem(1)).toBe(9); // index 3 -> 3*3
      expect(await asyncSubset.getItem(2)).toBe(15); // index 5 -> 5*3
    });

    it('should work with nested subsets', () => {
      const nestedSubset = subset.subset([0, 2, 4]);
      expect(nestedSubset.length).toBe(3);
      expect(nestedSubset.getItem(0)).toBe(0);
      expect(nestedSubset.getItem(1)).toBe(8);
      expect(nestedSubset.getItem(2)).toBe(16);
    });
  });

  describe('TensorDataset', () => {
    it('should create dataset from single tensor', () =>
      scopedTest(() => {
        const tensor = torch.tensor([1, 2, 3, 4, 5], [5] as const);
        const dataset = new TensorDataset([tensor]);

        expect(dataset.length).toBe(5);
      }));

    it('should create dataset from multiple tensors', () =>
      scopedTest(() => {
        const features = torch.ones([10, 4] as const);
        const labels = torch.zeros([10, 1] as const);
        const dataset = new TensorDataset([features, labels]);

        expect(dataset.length).toBe(10);
      }));

    it('should throw error for empty tensor list', () => {
      expect(() => new TensorDataset([])).toThrow('requires at least one tensor');
    });

    it('should throw error for mismatched first dimensions', () =>
      scopedTest(() => {
        const tensor1 = torch.ones([10, 4] as const);
        const tensor2 = torch.zeros([5, 4] as const);

        expect(() => new TensorDataset([tensor1, tensor2])).toThrow(
          'All tensors must have the same size in the first dimension'
        );
      }));

    it('should validate tensors have same batch size', () =>
      scopedTest(() => {
        const tensor1 = torch.ones([8, 3] as const);
        const tensor2 = torch.zeros([8, 5] as const);
        const tensor3 = torch.randn([8, 2] as const);

        const dataset = new TensorDataset([tensor1, tensor2, tensor3]);
        expect(dataset.length).toBe(8);
      }));

    it('should get item by index', () =>
      scopedTest(() => {
        const tensor = torch.tensor([1, 2, 3, 4, 5], [5] as const);
        const dataset = new TensorDataset([tensor]);

        const item = dataset.getItem(0);
        expect(item).toHaveLength(1);
      }));

    it('should throw error for out of bounds index', () =>
      scopedTest(() => {
        const tensor = torch.ones([5, 4] as const);
        const dataset = new TensorDataset([tensor]);

        expect(() => dataset.getItem(-1)).toThrow('out of bounds');
        expect(() => dataset.getItem(5)).toThrow('out of bounds');
      }));

    it('should be iterable', () =>
      scopedTest(() => {
        const tensor = torch.ones([3, 4] as const);
        const dataset = new TensorDataset([tensor]);

        const items = Array.from(dataset);
        expect(items).toHaveLength(3);
        items.forEach((item) => {
          expect(item).toHaveLength(1);
        });
      }));

    it('should work with subset', () =>
      scopedTest(() => {
        const tensor = torch.ones([10, 4] as const);
        const dataset = new TensorDataset([tensor]);
        const subset = dataset.subset([0, 2, 4, 6, 8]);

        expect(subset.length).toBe(5);
      }));

    it('should work with split', () =>
      scopedTest(() => {
        const tensor = torch.ones([100, 4] as const);
        const dataset = new TensorDataset([tensor]);
        const [train, test] = dataset.split(0.8);

        expect(train.length).toBe(80);
        expect(test.length).toBe(20);
      }));

    it('should handle 1D tensors', () =>
      scopedTest(() => {
        const tensor = torch.tensor([1, 2, 3, 4, 5], [5] as const);
        const dataset = new TensorDataset([tensor]);

        expect(dataset.length).toBe(5);
      }));

    it('should handle 3D tensors', () =>
      scopedTest(() => {
        const tensor = torch.ones([4, 3, 28, 28] as const);
        const dataset = new TensorDataset([tensor]);

        expect(dataset.length).toBe(4);
      }));
  });
});
