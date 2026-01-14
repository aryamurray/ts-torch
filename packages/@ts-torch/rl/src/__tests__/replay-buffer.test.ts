import { describe, it, expect } from 'vitest'
import { ReplayBuffer } from '../replay-buffer.js'

describe('ReplayBuffer', () => {
  const createTransition = (id: number) => ({
    state: new Float32Array([id, id + 0.1, id + 0.2, id + 0.3]),
    action: id % 3,
    reward: id * 0.1,
    nextState: new Float32Array([id + 1, id + 1.1, id + 1.2, id + 1.3]),
    done: id % 10 === 0,
  })

  describe('constructor', () => {
    it('creates buffer with specified capacity', () => {
      const buffer = new ReplayBuffer(100, 4)
      expect(buffer.maxCapacity).toBe(100)
      expect(buffer.size).toBe(0)
    })

    it('supports multi-dimensional rewards', () => {
      const buffer = new ReplayBuffer(100, 4, 3) // 3 objectives
      expect(buffer.maxCapacity).toBe(100)
    })
  })

  describe('push()', () => {
    it('adds transitions to buffer', () => {
      const buffer = new ReplayBuffer(100, 4)

      buffer.push(createTransition(0))
      expect(buffer.size).toBe(1)

      buffer.push(createTransition(1))
      expect(buffer.size).toBe(2)
    })

    it('handles scalar rewards', () => {
      const buffer = new ReplayBuffer(100, 4, 1)
      buffer.push({
        state: new Float32Array([1, 2, 3, 4]),
        action: 0,
        reward: 5.0,
        nextState: new Float32Array([2, 3, 4, 5]),
        done: false,
      })
      expect(buffer.size).toBe(1)
    })

    it('handles vector rewards', () => {
      const buffer = new ReplayBuffer(100, 4, 3)
      buffer.push({
        state: new Float32Array([1, 2, 3, 4]),
        action: 0,
        reward: new Float32Array([1, 2, 3]),
        nextState: new Float32Array([2, 3, 4, 5]),
        done: false,
      })
      expect(buffer.size).toBe(1)
    })

    it('wraps around when full', () => {
      const buffer = new ReplayBuffer(5, 4)

      for (let i = 0; i < 10; i++) {
        buffer.push(createTransition(i))
      }

      expect(buffer.size).toBe(5)
      expect(buffer.isFull).toBe(true)
    })
  })

  describe('sample()', () => {
    it('samples batch of transitions', () => {
      const buffer = new ReplayBuffer(100, 4)

      for (let i = 0; i < 50; i++) {
        buffer.push(createTransition(i))
      }

      const batch = buffer.sample(10)

      expect(batch.batchSize).toBe(10)
      expect(batch.stateSize).toBe(4)
      expect(batch.states.length).toBe(40) // 10 * 4
      expect(batch.actions.length).toBe(10)
      expect(batch.rewards.length).toBe(10)
      expect(batch.nextStates.length).toBe(40)
      expect(batch.dones.length).toBe(10)
    })

    it('throws if buffer too small', () => {
      const buffer = new ReplayBuffer(100, 4)

      buffer.push(createTransition(0))

      expect(() => buffer.sample(10)).toThrow()
    })

    it('samples different indices', () => {
      const buffer = new ReplayBuffer(100, 4)

      for (let i = 0; i < 100; i++) {
        buffer.push(createTransition(i))
      }

      // Sample multiple times and check for variation
      const samples: number[][] = []
      for (let i = 0; i < 10; i++) {
        const batch = buffer.sample(5)
        samples.push(Array.from(batch.states.slice(0, 4)))
      }

      // Not all samples should be identical
      const allSame = samples.every(
        (s) => s[0] === samples[0]![0] && s[1] === samples[0]![1],
      )
      expect(allSame).toBe(false)
    })
  })

  describe('sampleTransitions()', () => {
    it('returns array of Transition objects', () => {
      const buffer = new ReplayBuffer(100, 4)

      for (let i = 0; i < 50; i++) {
        buffer.push(createTransition(i))
      }

      const transitions = buffer.sampleTransitions(5)

      expect(transitions.length).toBe(5)
      expect(transitions[0]!.state).toBeInstanceOf(Float32Array)
      expect(transitions[0]!.state.length).toBe(4)
      expect(typeof transitions[0]!.action).toBe('number')
    })
  })

  describe('get()', () => {
    it('returns transition at index', () => {
      const buffer = new ReplayBuffer(100, 4)

      buffer.push(createTransition(42))

      const transition = buffer.get(0)

      expect(Math.abs(transition.state[0]! - 42)).toBeLessThan(0.001)
      expect(transition.action).toBe(42 % 3)
      expect(Math.abs((transition.reward as number) - 4.2)).toBeLessThan(0.001)
    })

    it('throws for out of bounds index', () => {
      const buffer = new ReplayBuffer(100, 4)
      buffer.push(createTransition(0))

      expect(() => buffer.get(-1)).toThrow()
      expect(() => buffer.get(10)).toThrow()
    })
  })

  describe('clear()', () => {
    it('empties the buffer', () => {
      const buffer = new ReplayBuffer(100, 4)

      for (let i = 0; i < 50; i++) {
        buffer.push(createTransition(i))
      }

      expect(buffer.size).toBe(50)

      buffer.clear()

      expect(buffer.size).toBe(0)
      expect(buffer.isFull).toBe(false)
    })
  })
})
