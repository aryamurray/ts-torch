/**
 * Tests for tensor pool optimization
 */

import { describe, test, expect, beforeEach } from "bun:test";
import { TensorPool, type PoolableTensor } from "../pool";

// Mock tensor for testing
class MockPoolableTensor implements PoolableTensor {
  constructor(
    public readonly shape: readonly number[],
    public readonly dtype: "float32" | "float64" | "int32" | "int64" | "float16" | "bfloat16" | "bool",
    public readonly handle: unknown = Math.random()
  ) {}
}

describe("TensorPool", () => {
  let pool: TensorPool;

  beforeEach(() => {
    pool = new TensorPool();
  });

  describe("acquire and release", () => {
    test("returns null when pool is empty", () => {
      const tensor = pool.acquire([10, 10], "float32");
      expect(tensor).toBeNull();
    });

    test("returns tensor after release", () => {
      const original = new MockPoolableTensor([10, 10], "float32");
      pool.release(original);

      const acquired = pool.acquire([10, 10], "float32");
      expect(acquired).toBe(original);
    });

    test("matches shape exactly", () => {
      const t1 = new MockPoolableTensor([10, 10], "float32");
      const t2 = new MockPoolableTensor([10, 20], "float32");

      pool.release(t1);
      pool.release(t2);

      const acquired = pool.acquire([10, 20], "float32");
      expect(acquired).toBe(t2);
    });

    test("matches dtype exactly", () => {
      const t1 = new MockPoolableTensor([10, 10], "float32");
      const t2 = new MockPoolableTensor([10, 10], "float64");

      pool.release(t1);
      pool.release(t2);

      const acquired = pool.acquire([10, 10], "float64");
      expect(acquired).toBe(t2);
    });

    test("handles different shapes independently", () => {
      const t1 = new MockPoolableTensor([5, 5], "float32");
      const t2 = new MockPoolableTensor([10, 10], "float32");
      const t3 = new MockPoolableTensor([20, 20], "float32");

      pool.release(t1);
      pool.release(t2);
      pool.release(t3);

      expect(pool.acquire([10, 10], "float32")).toBe(t2);
      expect(pool.acquire([5, 5], "float32")).toBe(t1);
      expect(pool.acquire([20, 20], "float32")).toBe(t3);
    });

    test("LIFO ordering (stack)", () => {
      const tensors = [
        new MockPoolableTensor([10, 10], "float32"),
        new MockPoolableTensor([10, 10], "float32"),
        new MockPoolableTensor([10, 10], "float32"),
      ];

      pool.release(tensors[0]);
      pool.release(tensors[1]);
      pool.release(tensors[2]);

      // Last in, first out
      expect(pool.acquire([10, 10], "float32")).toBe(tensors[2]);
      expect(pool.acquire([10, 10], "float32")).toBe(tensors[1]);
      expect(pool.acquire([10, 10], "float32")).toBe(tensors[0]);
    });
  });

  describe("maxPoolSize", () => {
    test("respects max pool size", () => {
      const smallPool = new TensorPool(2);
      const tensors = [
        new MockPoolableTensor([10, 10], "float32"),
        new MockPoolableTensor([10, 10], "float32"),
        new MockPoolableTensor([10, 10], "float32"),
      ];

      smallPool.release(tensors[0]);
      smallPool.release(tensors[1]);
      smallPool.release(tensors[2]);

      const stats = smallPool.stats();
      expect(stats.size).toBe(2); // Only 2 cached
    });

    test("keeps most recently released when full", () => {
      const smallPool = new TensorPool(2);
      const t1 = new MockPoolableTensor([10, 10], "float32");
      const t2 = new MockPoolableTensor([10, 10], "float32");
      const t3 = new MockPoolableTensor([10, 10], "float32");

      smallPool.release(t1);
      smallPool.release(t2);
      smallPool.release(t3); // This one is discarded

      const acquired1 = smallPool.acquire([10, 10], "float32");
      const acquired2 = smallPool.acquire([10, 10], "float32");

      expect([acquired1, acquired2]).toContain(t1);
      expect([acquired1, acquired2]).toContain(t2);
    });
  });

  describe("stats", () => {
    test("initial stats are empty", () => {
      const stats = pool.stats();
      expect(stats.size).toBe(0);
      expect(stats.hitCount).toBe(0);
      expect(stats.missCount).toBe(0);
      expect(stats.hitRate).toBe(0);
    });

    test("tracks hits and misses", () => {
      const tensor = new MockPoolableTensor([10, 10], "float32");
      pool.release(tensor);

      pool.acquire([10, 10], "float32"); // hit
      pool.acquire([10, 10], "float32"); // miss
      pool.acquire([20, 20], "float32"); // miss

      const stats = pool.stats();
      expect(stats.hitCount).toBe(1);
      expect(stats.missCount).toBe(2);
      expect(stats.hitRate).toBeCloseTo(1 / 3);
    });

    test("calculates hit rate correctly", () => {
      const tensors = [
        new MockPoolableTensor([10, 10], "float32"),
        new MockPoolableTensor([10, 10], "float32"),
      ];

      pool.release(tensors[0]);
      pool.release(tensors[1]);

      pool.acquire([10, 10], "float32"); // hit
      pool.acquire([10, 10], "float32"); // hit
      pool.acquire([10, 10], "float32"); // miss

      const stats = pool.stats();
      expect(stats.hitRate).toBeCloseTo(2 / 3);
    });

    test("tracks pool sizes per key", () => {
      pool.release(new MockPoolableTensor([10, 10], "float32"));
      pool.release(new MockPoolableTensor([10, 10], "float32"));
      pool.release(new MockPoolableTensor([20, 20], "float64"));

      const stats = pool.stats();
      expect(stats.size).toBe(3);
      expect(stats.pools.size).toBe(2); // Two unique keys
    });
  });

  describe("clear", () => {
    test("removes all cached tensors", () => {
      pool.release(new MockPoolableTensor([10, 10], "float32"));
      pool.release(new MockPoolableTensor([20, 20], "float64"));

      expect(pool.stats().size).toBe(2);

      pool.clear();

      expect(pool.stats().size).toBe(0);
      expect(pool.acquire([10, 10], "float32")).toBeNull();
    });

    test("resets statistics", () => {
      pool.release(new MockPoolableTensor([10, 10], "float32"));
      pool.acquire([10, 10], "float32");
      pool.acquire([10, 10], "float32");

      pool.clear();

      const stats = pool.stats();
      expect(stats.hitCount).toBe(0);
      expect(stats.missCount).toBe(0);
    });
  });

  describe("clearKey", () => {
    test("clears specific key only", () => {
      const t1 = new MockPoolableTensor([10, 10], "float32");
      const t2 = new MockPoolableTensor([20, 20], "float32");

      pool.release(t1);
      pool.release(t2);

      pool.clearKey([10, 10], "float32");

      expect(pool.acquire([10, 10], "float32")).toBeNull();
      expect(pool.acquire([20, 20], "float32")).toBe(t2);
    });
  });

  describe("getKeySize", () => {
    test("returns 0 for empty key", () => {
      expect(pool.getKeySize([10, 10], "float32")).toBe(0);
    });

    test("returns correct count for key", () => {
      pool.release(new MockPoolableTensor([10, 10], "float32"));
      pool.release(new MockPoolableTensor([10, 10], "float32"));
      pool.release(new MockPoolableTensor([20, 20], "float32"));

      expect(pool.getKeySize([10, 10], "float32")).toBe(2);
      expect(pool.getKeySize([20, 20], "float32")).toBe(1);
    });
  });

  describe("getKeys", () => {
    test("returns empty array initially", () => {
      expect(pool.getKeys()).toEqual([]);
    });

    test("returns all unique keys", () => {
      pool.release(new MockPoolableTensor([10, 10], "float32"));
      pool.release(new MockPoolableTensor([10, 10], "float32"));
      pool.release(new MockPoolableTensor([20, 20], "float64"));

      const keys = pool.getKeys();
      expect(keys).toHaveLength(2);
      expect(keys).toContain("float32:[10,10]");
      expect(keys).toContain("float64:[20,20]");
    });
  });

  describe("prune", () => {
    test("reduces pool size to target", () => {
      for (let i = 0; i < 10; i++) {
        pool.release(new MockPoolableTensor([10, 10], "float32"));
      }

      expect(pool.stats().size).toBe(10);

      pool.prune(5);

      expect(pool.stats().size).toBe(5);
    });

    test("does nothing if already under target", () => {
      pool.release(new MockPoolableTensor([10, 10], "float32"));
      pool.release(new MockPoolableTensor([10, 10], "float32"));

      pool.prune(5);

      expect(pool.stats().size).toBe(2);
    });

    test("defaults to half size when no target given", () => {
      for (let i = 0; i < 10; i++) {
        pool.release(new MockPoolableTensor([10, 10], "float32"));
      }

      pool.prune();

      expect(pool.stats().size).toBe(5);
    });

    test("removes empty pools after pruning", () => {
      pool.release(new MockPoolableTensor([10, 10], "float32"));
      pool.release(new MockPoolableTensor([20, 20], "float64"));

      pool.prune(0);

      expect(pool.getKeys()).toEqual([]);
    });
  });

  describe("Real-world scenarios", () => {
    test("training loop pattern", () => {
      // Simulate a training loop reusing tensors
      const results: (PoolableTensor | null)[] = [];

      for (let i = 0; i < 100; i++) {
        const tensor =
          pool.acquire([256, 256], "float32") ??
          new MockPoolableTensor([256, 256], "float32");

        results.push(tensor);
        pool.release(tensor);
      }

      const stats = pool.stats();
      expect(stats.hitRate).toBeGreaterThan(0.9); // High hit rate expected
    });

    test("mixed shapes and dtypes", () => {
      const shapes: Array<readonly number[]> = [
        [10, 10],
        [20, 20],
        [10, 10],
        [30, 30],
        [20, 20],
      ];
      const dtypes: Array<"float32" | "float64"> = [
        "float32",
        "float32",
        "float64",
        "float32",
        "float64",
      ];

      // Release phase
      for (let i = 0; i < shapes.length; i++) {
        pool.release(new MockPoolableTensor(shapes[i]!, dtypes[i]!));
      }

      // Acquire phase
      for (let i = 0; i < shapes.length; i++) {
        const tensor = pool.acquire(shapes[i]!, dtypes[i]!);
        expect(tensor).not.toBeNull();
      }

      // All should be exhausted now
      for (let i = 0; i < shapes.length; i++) {
        const tensor = pool.acquire(shapes[i]!, dtypes[i]!);
        expect(tensor).toBeNull();
      }
    });

    test("memory pressure management", () => {
      // Fill pool
      for (let i = 0; i < 1000; i++) {
        pool.release(new MockPoolableTensor([10, 10], "float32"));
      }

      const beforePrune = pool.stats().size;
      expect(beforePrune).toBe(1000);

      // Simulate memory pressure
      pool.prune(100);

      const afterPrune = pool.stats().size;
      expect(afterPrune).toBe(100);
    });
  });
});
