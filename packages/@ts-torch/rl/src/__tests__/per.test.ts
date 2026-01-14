import { describe, it, expect } from 'vitest'
import { ReplayBuffer } from '../replay-buffer.js'

describe('Prioritized Experience Replay', () => {
  const createTransition = (id: number) => ({
    state: new Float32Array([id, id + 0.1, id + 0.2, id + 0.3]),
    action: id % 3,
    reward: id * 0.1,
    nextState: new Float32Array([id + 1, id + 1.1, id + 1.2, id + 1.3]),
    done: id % 10 === 0,
  })

  describe('PER-enabled buffer', () => {
    it('creates prioritized buffer', () => {
      const buffer = new ReplayBuffer(100, 4, 1, {
        prioritized: true,
        alpha: 0.6,
        betaStart: 0.4,
        betaEnd: 1.0,
        betaFrames: 10000,
      })

      expect(buffer.isPrioritized).toBe(true)
    })

    it('uses default PER config values', () => {
      const buffer = new ReplayBuffer(100, 4, 1, { prioritized: true })

      expect(buffer.isPrioritized).toBe(true)
      expect(buffer.currentBeta).toBeGreaterThan(0)
    })
  })

  describe('push() with priorities', () => {
    it('adds transitions with default max priority', () => {
      const buffer = new ReplayBuffer(100, 4, 1, { prioritized: true })

      buffer.push(createTransition(0))
      buffer.push(createTransition(1))

      expect(buffer.size).toBe(2)
    })

    it('adds transitions with custom priority', () => {
      const buffer = new ReplayBuffer(100, 4, 1, { prioritized: true })

      buffer.push(createTransition(0), 1.0)
      buffer.push(createTransition(1), 5.0)

      expect(buffer.size).toBe(2)
    })
  })

  describe('sample() with PER', () => {
    it('returns indices for priority updates', () => {
      const buffer = new ReplayBuffer(100, 4, 1, { prioritized: true })

      for (let i = 0; i < 50; i++) {
        buffer.push(createTransition(i))
      }

      const batch = buffer.sample(10)

      expect(batch.indices).toBeDefined()
      expect(batch.indices!.length).toBe(10)

      // All indices should be valid
      for (const idx of batch.indices!) {
        expect(idx).toBeGreaterThanOrEqual(0)
        expect(idx).toBeLessThan(50)
      }
    })

    it('returns importance sampling weights', () => {
      const buffer = new ReplayBuffer(100, 4, 1, { prioritized: true })

      for (let i = 0; i < 50; i++) {
        buffer.push(createTransition(i))
      }

      const batch = buffer.sample(10)

      expect(batch.weights).toBeDefined()
      expect(batch.weights!.length).toBe(10)

      // All weights should be positive and <= 1 (normalized)
      for (const weight of batch.weights!) {
        expect(weight).toBeGreaterThan(0)
        expect(weight).toBeLessThanOrEqual(1.001) // Allow small float error
      }
    })

    it('samples high-priority transitions more frequently', () => {
      const buffer = new ReplayBuffer(10, 4, 1, { prioritized: true, alpha: 1.0 })

      // Add low priority transitions
      for (let i = 0; i < 5; i++) {
        buffer.push(createTransition(i), 0.1)
      }

      // Add high priority transitions
      for (let i = 5; i < 10; i++) {
        buffer.push(createTransition(i), 10.0)
      }

      // Sample many times and count indices
      const lowPriorityCount = { count: 0 }
      const highPriorityCount = { count: 0 }

      for (let i = 0; i < 100; i++) {
        const batch = buffer.sample(5)
        for (const idx of batch.indices!) {
          if (idx < 5) lowPriorityCount.count++
          else highPriorityCount.count++
        }
      }

      // High priority transitions should be sampled much more
      expect(highPriorityCount.count).toBeGreaterThan(lowPriorityCount.count * 2)
    })
  })

  describe('updatePriorities()', () => {
    it('updates priorities for sampled transitions', () => {
      const buffer = new ReplayBuffer(100, 4, 1, { prioritized: true })

      for (let i = 0; i < 50; i++) {
        buffer.push(createTransition(i), 1.0)
      }

      const batch = buffer.sample(10)

      // Simulate TD-errors
      const tdErrors = new Float32Array(10)
      for (let i = 0; i < 10; i++) {
        tdErrors[i] = Math.random() * 2 // Random TD-errors
      }

      // Should not throw
      buffer.updatePriorities(batch.indices!, tdErrors)
    })

    it('affects subsequent sampling', () => {
      const buffer = new ReplayBuffer(10, 4, 1, { prioritized: true, alpha: 1.0 })

      // Add transitions with equal priority
      for (let i = 0; i < 10; i++) {
        buffer.push(createTransition(i), 1.0)
      }

      // Give first transition very high priority
      const indices = new Int32Array([0])
      const priorities = new Float32Array([100.0])
      buffer.updatePriorities(indices, priorities)

      // Sample and check if index 0 is sampled more
      let index0Count = 0
      for (let i = 0; i < 100; i++) {
        const batch = buffer.sample(1)
        if (batch.indices![0] === 0) index0Count++
      }

      // Index 0 should be sampled frequently (>30% of time at least)
      expect(index0Count).toBeGreaterThan(30)
    })

    it('no-op for non-prioritized buffer', () => {
      const buffer = new ReplayBuffer(100, 4, 1) // Non-prioritized

      for (let i = 0; i < 50; i++) {
        buffer.push(createTransition(i))
      }

      const indices = new Int32Array([0, 1, 2])
      const priorities = new Float32Array([1.0, 2.0, 3.0])

      // Should not throw
      buffer.updatePriorities(indices, priorities)
    })
  })

  describe('beta annealing', () => {
    it('starts at betaStart', () => {
      const buffer = new ReplayBuffer(100, 4, 1, {
        prioritized: true,
        betaStart: 0.4,
        betaEnd: 1.0,
        betaFrames: 10000,
      })

      expect(Math.abs(buffer.currentBeta - 0.4)).toBeLessThan(1e-5)
    })

    it('anneals beta towards betaEnd', () => {
      const buffer = new ReplayBuffer(100, 4, 1, {
        prioritized: true,
        betaStart: 0.4,
        betaEnd: 1.0,
        betaFrames: 100,
      })

      // Fill buffer
      for (let i = 0; i < 50; i++) {
        buffer.push(createTransition(i))
      }

      // Sample many times to increment frame count
      for (let i = 0; i < 50; i++) {
        buffer.sample(5)
      }

      // Beta should have increased from 0.4
      expect(buffer.currentBeta).toBeGreaterThan(0.4)
    })

    it('caps beta at betaEnd', () => {
      const buffer = new ReplayBuffer(100, 4, 1, {
        prioritized: true,
        betaStart: 0.4,
        betaEnd: 1.0,
        betaFrames: 10, // Very short annealing
      })

      // Fill buffer
      for (let i = 0; i < 50; i++) {
        buffer.push(createTransition(i))
      }

      // Sample many times to exceed betaFrames
      for (let i = 0; i < 50; i++) {
        buffer.sample(5)
      }

      expect(Math.abs(buffer.currentBeta - 1.0)).toBeLessThan(1e-5)
    })
  })

  describe('importance sampling weights', () => {
    it('weights are normalized to max 1', () => {
      const buffer = new ReplayBuffer(100, 4, 1, { prioritized: true })

      for (let i = 0; i < 50; i++) {
        buffer.push(createTransition(i), (i + 1) * 0.1)
      }

      const batch = buffer.sample(20)

      // Find max weight
      let maxWeight = 0
      for (const w of batch.weights!) {
        if (w > maxWeight) maxWeight = w
      }

      // Max weight should be close to 1 (normalized)
      // Note: Due to the segment-based sampling strategy, the max weight
      // is typically 1 but can vary slightly based on which transitions are sampled
      expect(maxWeight).toBeGreaterThan(0.5)
      expect(maxWeight).toBeLessThanOrEqual(1.001)
    })

    it('lower beta gives more uniform weights', () => {
      const bufferLowBeta = new ReplayBuffer(50, 4, 1, {
        prioritized: true,
        betaStart: 0.1,
        betaEnd: 0.1,
        betaFrames: 1000000,
      })

      const bufferHighBeta = new ReplayBuffer(50, 4, 1, {
        prioritized: true,
        betaStart: 1.0,
        betaEnd: 1.0,
        betaFrames: 1000000,
      })

      // Fill both with same data
      for (let i = 0; i < 50; i++) {
        bufferLowBeta.push(createTransition(i), (i + 1) * 0.1)
        bufferHighBeta.push(createTransition(i), (i + 1) * 0.1)
      }

      // Compute weight variance for multiple samples
      const computeWeightVariance = (weights: Float32Array) => {
        const mean = weights.reduce((a, b) => a + b, 0) / weights.length
        const variance = weights.reduce((sum, w) => sum + (w - mean) ** 2, 0) / weights.length
        return variance
      }

      const batchLow = bufferLowBeta.sample(20)
      const batchHigh = bufferHighBeta.sample(20)

      const varianceLow = computeWeightVariance(batchLow.weights!)
      const varianceHigh = computeWeightVariance(batchHigh.weights!)

      // Lower beta should give lower variance (more uniform weights)
      expect(varianceLow).toBeLessThan(varianceHigh)
    })
  })

  describe('clear() with PER', () => {
    it('clears SumTree', () => {
      const buffer = new ReplayBuffer(100, 4, 1, { prioritized: true })

      for (let i = 0; i < 50; i++) {
        buffer.push(createTransition(i))
      }

      buffer.clear()

      expect(buffer.size).toBe(0)
      expect(buffer.isPrioritized).toBe(true) // Still PER-enabled
    })
  })

  describe('non-PER buffer comparison', () => {
    it('does not return indices/weights for non-PER buffer', () => {
      const buffer = new ReplayBuffer(100, 4, 1) // No PER config

      for (let i = 0; i < 50; i++) {
        buffer.push(createTransition(i))
      }

      const batch = buffer.sample(10)

      expect(batch.indices).toBeUndefined()
      expect(batch.weights).toBeUndefined()
      expect(buffer.isPrioritized).toBe(false)
    })
  })
})
