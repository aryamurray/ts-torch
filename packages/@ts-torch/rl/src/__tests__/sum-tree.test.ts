import { describe, it, expect } from 'vitest'
import { SumTree } from '../utils/sum-tree.js'

describe('SumTree', () => {
  describe('constructor', () => {
    it('creates tree with specified capacity', () => {
      const tree = new SumTree(100)
      expect(tree.size).toBe(100)
      expect(tree.total).toBe(0)
    })

    it('throws for non-positive capacity', () => {
      expect(() => new SumTree(0)).toThrow()
      expect(() => new SumTree(-1)).toThrow()
    })
  })

  describe('update()', () => {
    it('updates priority at index', () => {
      const tree = new SumTree(4)

      tree.update(0, 1.0)
      expect(tree.get(0)).toBe(1.0)
      expect(tree.total).toBe(1.0)

      tree.update(1, 2.0)
      expect(tree.get(1)).toBe(2.0)
      expect(tree.total).toBe(3.0)
    })

    it('updates total when priority changes', () => {
      const tree = new SumTree(4)

      tree.update(0, 1.0)
      tree.update(1, 2.0)
      tree.update(2, 3.0)
      tree.update(3, 4.0)

      expect(tree.total).toBe(10.0)

      // Update existing priority
      tree.update(1, 5.0)
      expect(tree.total).toBe(13.0) // 1 + 5 + 3 + 4
    })

    it('throws for out of bounds index', () => {
      const tree = new SumTree(4)
      expect(() => tree.update(-1, 1.0)).toThrow()
      expect(() => tree.update(4, 1.0)).toThrow()
    })

    it('throws for negative priority', () => {
      const tree = new SumTree(4)
      expect(() => tree.update(0, -1.0)).toThrow()
    })

    it('allows zero priority', () => {
      const tree = new SumTree(4)
      tree.update(0, 1.0)
      tree.update(0, 0.0)
      expect(tree.get(0)).toBe(0.0)
      expect(tree.total).toBe(0.0)
    })
  })

  describe('sample()', () => {
    it('samples proportionally to priorities', () => {
      const tree = new SumTree(4)

      // Set up priorities: [1, 2, 3, 4] (total = 10)
      tree.update(0, 1.0)
      tree.update(1, 2.0)
      tree.update(2, 3.0)
      tree.update(3, 4.0)

      // Sample at specific values
      expect(tree.sample(0.5)).toBe(0) // First bucket [0, 1)
      expect(tree.sample(1.5)).toBe(1) // Second bucket [1, 3)
      expect(tree.sample(4.5)).toBe(2) // Third bucket [3, 6)
      expect(tree.sample(8.5)).toBe(3) // Fourth bucket [6, 10)
    })

    it('handles edge values', () => {
      const tree = new SumTree(4)
      tree.update(0, 1.0)
      tree.update(1, 1.0)
      tree.update(2, 1.0)
      tree.update(3, 1.0)

      expect(tree.sample(0)).toBe(0)
      // Value very close to total should return last valid index
      expect(tree.sample(tree.total - 0.001)).toBeLessThan(4)
    })

    it('returns uniform sample when all priorities are zero', () => {
      const tree = new SumTree(4)
      // All priorities default to 0

      const idx = tree.sample(0.5)
      expect(idx).toBeGreaterThanOrEqual(0)
      expect(idx).toBeLessThan(4)
    })

    it('samples higher priority indices more frequently', () => {
      const tree = new SumTree(2)
      tree.update(0, 1.0) // 10% probability
      tree.update(1, 9.0) // 90% probability

      const counts = [0, 0]
      const iterations = 1000

      for (let i = 0; i < iterations; i++) {
        const value = Math.random() * tree.total
        const idx = tree.sample(value)
        counts[idx]!++
      }

      // Second index should be sampled much more frequently
      // Allow for statistical variance
      expect(counts[1]!).toBeGreaterThan(counts[0]! * 2)
    })
  })

  describe('get()', () => {
    it('returns priority at index', () => {
      const tree = new SumTree(4)
      tree.update(2, 5.5)

      expect(tree.get(2)).toBe(5.5)
      expect(tree.get(0)).toBe(0) // Not set
    })

    it('returns 0 for out of bounds index', () => {
      const tree = new SumTree(4)
      expect(tree.get(-1)).toBe(0)
      expect(tree.get(10)).toBe(0)
    })
  })

  describe('total', () => {
    it('returns sum of all priorities', () => {
      const tree = new SumTree(4)
      tree.update(0, 1.0)
      tree.update(1, 2.0)
      tree.update(2, 3.0)
      tree.update(3, 4.0)

      expect(tree.total).toBe(10.0)
    })

    it('returns 0 for empty tree', () => {
      const tree = new SumTree(4)
      expect(tree.total).toBe(0)
    })
  })

  describe('min', () => {
    it('returns minimum non-zero priority', () => {
      const tree = new SumTree(4)
      tree.update(0, 5.0)
      tree.update(1, 1.0)
      tree.update(2, 3.0)

      expect(tree.min).toBe(1.0)
    })

    it('ignores zero priorities', () => {
      const tree = new SumTree(4)
      tree.update(0, 0.0)
      tree.update(1, 2.0)
      tree.update(2, 5.0)

      expect(tree.min).toBe(2.0)
    })

    it('returns Infinity when all priorities are zero', () => {
      const tree = new SumTree(4)
      expect(tree.min).toBe(Infinity)
    })

    it('caches min value', () => {
      const tree = new SumTree(4)
      tree.update(0, 1.0)
      tree.update(1, 2.0)

      // First call computes min
      const min1 = tree.min
      // Second call should use cache
      const min2 = tree.min

      expect(min1).toBe(min2)
    })

    it('invalidates cache on update', () => {
      const tree = new SumTree(4)
      tree.update(0, 2.0)
      tree.update(1, 3.0)

      expect(tree.min).toBe(2.0)

      // Update with smaller priority
      tree.update(2, 1.0)

      expect(tree.min).toBe(1.0)
    })
  })

  describe('max', () => {
    it('returns maximum priority', () => {
      const tree = new SumTree(4)
      tree.update(0, 1.0)
      tree.update(1, 5.0)
      tree.update(2, 3.0)

      expect(tree.max).toBe(5.0)
    })

    it('returns 0 for empty tree', () => {
      const tree = new SumTree(4)
      expect(tree.max).toBe(0)
    })
  })

  describe('clear()', () => {
    it('resets all priorities to zero', () => {
      const tree = new SumTree(4)
      tree.update(0, 1.0)
      tree.update(1, 2.0)
      tree.update(2, 3.0)

      tree.clear()

      expect(tree.total).toBe(0)
      expect(tree.get(0)).toBe(0)
      expect(tree.get(1)).toBe(0)
      expect(tree.get(2)).toBe(0)
    })

    it('invalidates min cache', () => {
      const tree = new SumTree(4)
      tree.update(0, 1.0)

      // Prime the cache
      void tree.min

      tree.clear()

      expect(tree.min).toBe(Infinity)
    })
  })

  describe('edge cases', () => {
    it('handles capacity of 1', () => {
      const tree = new SumTree(1)
      tree.update(0, 5.0)

      expect(tree.total).toBe(5.0)
      expect(tree.sample(2.5)).toBe(0)
    })

    it('handles non-power-of-2 capacity', () => {
      const tree = new SumTree(5)

      for (let i = 0; i < 5; i++) {
        tree.update(i, i + 1)
      }

      expect(tree.total).toBe(15) // 1+2+3+4+5
      expect(tree.size).toBe(5)
    })

    it('handles very small priorities', () => {
      const tree = new SumTree(4)
      tree.update(0, 1e-10)
      tree.update(1, 1e-10)

      expect(Math.abs(tree.total - 2e-10)).toBeLessThan(1e-15)
    })

    it('handles very large priorities', () => {
      const tree = new SumTree(4)
      tree.update(0, 1e10)
      tree.update(1, 1e10)

      expect(Math.abs(tree.total - 2e10)).toBeLessThan(1)
    })
  })
})
