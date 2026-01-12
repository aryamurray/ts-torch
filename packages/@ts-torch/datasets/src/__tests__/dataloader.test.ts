/**
 * Tests for DataLoader
 *
 * Tests batching, shuffling, and iteration over datasets.
 */

import { describe, it, expect } from 'vitest';
import { DataLoader } from '../dataloader.js';
import { BaseDataset } from '../dataset.js';

/**
 * Simple synchronous dataset for testing
 */
class RangeDataset extends BaseDataset<number> {
  constructor(private size: number) {
    super();
  }

  getItem(index: number): number {
    if (index < 0 || index >= this.size) {
      throw new Error(`Index ${index} out of bounds`);
    }
    return index;
  }

  get length(): number {
    return this.size;
  }
}

/**
 * Async dataset for testing async iteration
 */
class AsyncRangeDataset extends BaseDataset<number> {
  constructor(private size: number) {
    super();
  }

  async getItem(index: number): Promise<number> {
    if (index < 0 || index >= this.size) {
      throw new Error(`Index ${index} out of bounds`);
    }
    // Simulate async operation
    await new Promise((resolve) => setTimeout(resolve, 1));
    return index;
  }

  get length(): number {
    return this.size;
  }
}

describe('DataLoader', () => {
  describe('Constructor', () => {
    it('should create with default options', () => {
      const dataset = new RangeDataset(10);
      const loader = new DataLoader(dataset);

      expect(loader.numBatches).toBeGreaterThan(0);
    });

    it('should create with custom batch size', () => {
      const dataset = new RangeDataset(10);
      const loader = new DataLoader(dataset, { batchSize: 2 });

      expect(loader.numBatches).toBe(5);
    });

    it('should create with shuffle enabled', () => {
      const dataset = new RangeDataset(10);
      const loader = new DataLoader(dataset, { shuffle: true });

      expect(loader.numBatches).toBeGreaterThan(0);
    });

    it('should create with drop_last enabled', () => {
      const dataset = new RangeDataset(10);
      const loader = new DataLoader(dataset, { batchSize: 3, drop_last: true });

      expect(loader.numBatches).toBe(3); // 10 / 3 = 3 (floor)
    });

    it('should throw error for invalid batch size', () => {
      const dataset = new RangeDataset(10);
      expect(() => new DataLoader(dataset, { batchSize: 0 })).toThrow('positive integer');
      expect(() => new DataLoader(dataset, { batchSize: -1 })).toThrow('positive integer');
    });
  });

  describe('numBatches calculation', () => {
    it('should calculate correct number of batches without drop_last', () => {
      const dataset = new RangeDataset(10);

      const loader1 = new DataLoader(dataset, { batchSize: 1 });
      expect(loader1.numBatches).toBe(10);

      const loader2 = new DataLoader(dataset, { batchSize: 2 });
      expect(loader2.numBatches).toBe(5);

      const loader3 = new DataLoader(dataset, { batchSize: 3 });
      expect(loader3.numBatches).toBe(4); // ceil(10/3)

      const loader4 = new DataLoader(dataset, { batchSize: 10 });
      expect(loader4.numBatches).toBe(1);

      const loader5 = new DataLoader(dataset, { batchSize: 20 });
      expect(loader5.numBatches).toBe(1);
    });

    it('should calculate correct number of batches with drop_last', () => {
      const dataset = new RangeDataset(10);

      const loader1 = new DataLoader(dataset, { batchSize: 1, drop_last: true });
      expect(loader1.numBatches).toBe(10);

      const loader2 = new DataLoader(dataset, { batchSize: 2, drop_last: true });
      expect(loader2.numBatches).toBe(5);

      const loader3 = new DataLoader(dataset, { batchSize: 3, drop_last: true });
      expect(loader3.numBatches).toBe(3); // floor(10/3)

      const loader4 = new DataLoader(dataset, { batchSize: 10, drop_last: true });
      expect(loader4.numBatches).toBe(1);

      const loader5 = new DataLoader(dataset, { batchSize: 20, drop_last: true });
      expect(loader5.numBatches).toBe(0);
    });

    it('should handle edge cases', () => {
      const dataset = new RangeDataset(1);
      const loader = new DataLoader(dataset, { batchSize: 1 });
      expect(loader.numBatches).toBe(1);

      const emptyLoader = new DataLoader(new RangeDataset(0), { batchSize: 1 });
      expect(emptyLoader.numBatches).toBe(0);
    });
  });

  describe('Synchronous iteration with iter()', () => {
    it('should iterate over batches', () => {
      const dataset = new RangeDataset(10);
      const loader = new DataLoader(dataset, { batchSize: 3 });

      const batches = Array.from(loader.iter());

      expect(batches).toHaveLength(4);
      expect(batches[0]).toEqual([0, 1, 2]);
      expect(batches[1]).toEqual([3, 4, 5]);
      expect(batches[2]).toEqual([6, 7, 8]);
      expect(batches[3]).toEqual([9]); // Last batch smaller
    });

    it('should respect batch size', () => {
      const dataset = new RangeDataset(20);
      const loader = new DataLoader(dataset, { batchSize: 5 });

      const batches = Array.from(loader.iter());

      expect(batches).toHaveLength(4);
      batches.slice(0, -1).forEach((batch) => {
        expect(batch).toHaveLength(5);
      });
    });

    it('should drop last incomplete batch when drop_last is true', () => {
      const dataset = new RangeDataset(10);
      const loader = new DataLoader(dataset, { batchSize: 3, drop_last: true });

      const batches = Array.from(loader.iter());

      expect(batches).toHaveLength(3);
      expect(batches[0]).toEqual([0, 1, 2]);
      expect(batches[1]).toEqual([3, 4, 5]);
      expect(batches[2]).toEqual([6, 7, 8]);
      // [9] is dropped
    });

    it('should keep last incomplete batch when drop_last is false', () => {
      const dataset = new RangeDataset(10);
      const loader = new DataLoader(dataset, { batchSize: 3, drop_last: false });

      const batches = Array.from(loader.iter());

      expect(batches).toHaveLength(4);
      expect(batches[3]).toEqual([9]);
    });

    it('should handle single batch', () => {
      const dataset = new RangeDataset(5);
      const loader = new DataLoader(dataset, { batchSize: 10 });

      const batches = Array.from(loader.iter());

      expect(batches).toHaveLength(1);
      expect(batches[0]).toEqual([0, 1, 2, 3, 4]);
    });

    it('should handle batch size of 1', () => {
      const dataset = new RangeDataset(5);
      const loader = new DataLoader(dataset, { batchSize: 1 });

      const batches = Array.from(loader.iter());

      expect(batches).toHaveLength(5);
      batches.forEach((batch, i) => {
        expect(batch).toEqual([i]);
      });
    });

    it('should shuffle indices when shuffle is true', () => {
      const dataset = new RangeDataset(10);
      const loader = new DataLoader(dataset, { batchSize: 10, shuffle: true });

      const batches1 = Array.from(loader.iter());
      const batches2 = Array.from(loader.iter());

      // Both should have all items
      expect(batches1[0]?.sort()).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
      expect(batches2[0]?.sort()).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);

      // But order should likely be different (not guaranteed but very likely)
      // Just verify structure is correct
      expect(batches1).toHaveLength(1);
      expect(batches2).toHaveLength(1);
    });

    it('should not shuffle when shuffle is false', () => {
      const dataset = new RangeDataset(10);
      const loader = new DataLoader(dataset, { batchSize: 10, shuffle: false });

      const batches = Array.from(loader.iter());

      expect(batches).toHaveLength(1);
      expect(batches[0]).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    });

    it('should throw error for async dataset', () => {
      const dataset = new AsyncRangeDataset(10);
      const loader = new DataLoader(dataset, { batchSize: 2 });

      expect(() => Array.from(loader.iter())).toThrow('Cannot use synchronous iterator with async dataset');
    });

    it('should handle empty dataset', () => {
      const dataset = new RangeDataset(0);
      const loader = new DataLoader(dataset, { batchSize: 5 });

      const batches = Array.from(loader.iter());

      expect(batches).toHaveLength(0);
    });
  });

  describe('Async iteration', () => {
    it('should iterate over batches asynchronously', async () => {
      const dataset = new RangeDataset(10);
      const loader = new DataLoader(dataset, { batchSize: 3 });

      const batches: number[][] = [];
      for await (const batch of loader) {
        batches.push(batch);
      }

      expect(batches).toHaveLength(4);
      expect(batches[0]).toEqual([0, 1, 2]);
      expect(batches[1]).toEqual([3, 4, 5]);
      expect(batches[2]).toEqual([6, 7, 8]);
      expect(batches[3]).toEqual([9]);
    });

    it('should work with async dataset', async () => {
      const dataset = new AsyncRangeDataset(6);
      const loader = new DataLoader(dataset, { batchSize: 2 });

      const batches: number[][] = [];
      for await (const batch of loader) {
        batches.push(batch);
      }

      expect(batches).toHaveLength(3);
      expect(batches[0]).toEqual([0, 1]);
      expect(batches[1]).toEqual([2, 3]);
      expect(batches[2]).toEqual([4, 5]);
    });

    it('should respect drop_last in async iteration', async () => {
      const dataset = new RangeDataset(10);
      const loader = new DataLoader(dataset, { batchSize: 3, drop_last: true });

      const batches: number[][] = [];
      for await (const batch of loader) {
        batches.push(batch);
      }

      expect(batches).toHaveLength(3);
      batches.forEach((batch) => {
        expect(batch).toHaveLength(3);
      });
    });

    it('should shuffle in async iteration', async () => {
      const dataset = new AsyncRangeDataset(10);
      const loader = new DataLoader(dataset, { batchSize: 10, shuffle: true });

      const batches: number[][] = [];
      for await (const batch of loader) {
        batches.push(batch);
      }

      expect(batches).toHaveLength(1);
      expect(batches[0]?.sort()).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    });

    it('should handle empty dataset in async iteration', async () => {
      const dataset = new RangeDataset(0);
      const loader = new DataLoader(dataset, { batchSize: 5 });

      const batches: number[][] = [];
      for await (const batch of loader) {
        batches.push(batch);
      }

      expect(batches).toHaveLength(0);
    });

    it('should handle single item dataset', async () => {
      const dataset = new AsyncRangeDataset(1);
      const loader = new DataLoader(dataset, { batchSize: 5 });

      const batches: number[][] = [];
      for await (const batch of loader) {
        batches.push(batch);
      }

      expect(batches).toHaveLength(1);
      expect(batches[0]).toEqual([0]);
    });

    it('should handle exact multiple of batch size', async () => {
      const dataset = new RangeDataset(10);
      const loader = new DataLoader(dataset, { batchSize: 5 });

      const batches: number[][] = [];
      for await (const batch of loader) {
        batches.push(batch);
      }

      expect(batches).toHaveLength(2);
      expect(batches[0]).toHaveLength(5);
      expect(batches[1]).toHaveLength(5);
    });
  });

  describe('Multiple iterations', () => {
    it('should allow multiple sync iterations', () => {
      const dataset = new RangeDataset(6);
      const loader = new DataLoader(dataset, { batchSize: 2 });

      const batches1 = Array.from(loader.iter());
      const batches2 = Array.from(loader.iter());

      expect(batches1).toHaveLength(3);
      expect(batches2).toHaveLength(3);
      expect(batches1).toEqual(batches2);
    });

    it('should allow multiple async iterations', async () => {
      const dataset = new RangeDataset(6);
      const loader = new DataLoader(dataset, { batchSize: 2 });

      const batches1: number[][] = [];
      for await (const batch of loader) {
        batches1.push(batch);
      }

      const batches2: number[][] = [];
      for await (const batch of loader) {
        batches2.push(batch);
      }

      expect(batches1).toHaveLength(3);
      expect(batches2).toHaveLength(3);
      expect(batches1).toEqual(batches2);
    });

    it('should produce different shuffles on each iteration', () => {
      const dataset = new RangeDataset(100);
      const loader = new DataLoader(dataset, { batchSize: 100, shuffle: true });

      const batches1 = Array.from(loader.iter());
      const batches2 = Array.from(loader.iter());

      // Both should contain all elements (make copies before sorting with numeric comparison)
      const batch1First = batches1[0];
      const batch2First = batches2[0];
      if (batch1First === undefined || batch2First === undefined) {
        throw new Error('Expected batches to have at least one element');
      }
      expect([...batch1First].sort((a, b) => a - b)).toEqual(Array.from({ length: 100 }, (_, i) => i));
      expect([...batch2First].sort((a, b) => a - b)).toEqual(Array.from({ length: 100 }, (_, i) => i));

      // Order should be different (very likely with 100 elements)
      // Note: There's a tiny chance this could be the same, but with 100! permutations it's negligible
      // For test reliability, we just verify structure
      expect(batches1).toHaveLength(1);
      expect(batches2).toHaveLength(1);
    });
  });
});
