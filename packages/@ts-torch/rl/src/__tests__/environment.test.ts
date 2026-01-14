import { describe, it, expect } from 'vitest'
import { env, FunctionalEnv } from '../environment.js'

describe('Environment', () => {
  // Simple counter environment for testing
  interface CounterState {
    count: number
  }

  const createCounterEnv = () =>
    env<CounterState>({
      createState: () => ({ count: 0 }),
      step: (state, action) => {
        state.count += action === 1 ? 1 : -1
        return {
          reward: state.count >= 10 ? 10 : -1,
          done: state.count >= 10 || state.count <= -10,
        }
      },
      observe: (state) => new Float32Array([state.count / 10]),
      actionSpace: 2,
    })

  describe('env()', () => {
    it('creates a FunctionalEnv from config', () => {
      const counterEnv = createCounterEnv()
      expect(counterEnv).toBeInstanceOf(FunctionalEnv)
    })

    it('returns correct observation size', () => {
      const counterEnv = createCounterEnv()
      expect(counterEnv.observationSize).toBe(1)
    })

    it('returns correct action space', () => {
      const counterEnv = createCounterEnv()
      expect(counterEnv.actionSpace).toBe(2)
    })
  })

  describe('reset()', () => {
    it('returns initial observation', () => {
      const counterEnv = createCounterEnv()
      const obs = counterEnv.reset()
      expect(obs).toBeInstanceOf(Float32Array)
      expect(obs.length).toBe(1)
      expect(obs[0]).toBe(0) // count starts at 0
    })

    it('resets step count', () => {
      const counterEnv = createCounterEnv()
      counterEnv.step(1)
      counterEnv.step(1)
      expect(counterEnv.stepCount).toBe(2)

      counterEnv.reset()
      expect(counterEnv.stepCount).toBe(0)
    })
  })

  describe('step()', () => {
    it('mutates state in place', () => {
      const counterEnv = createCounterEnv()
      counterEnv.reset()

      const result = counterEnv.step(1) // increment
      expect(Math.abs(result.observation[0]! - 0.1)).toBeLessThan(0.001) // 1/10
    })

    it('returns reward and done flag', () => {
      const counterEnv = createCounterEnv()
      counterEnv.reset()

      const result = counterEnv.step(1)
      expect(typeof result.reward).toBe('number')
      expect(typeof result.done).toBe('boolean')
    })

    it('increments step count', () => {
      const counterEnv = createCounterEnv()
      counterEnv.reset()

      expect(counterEnv.stepCount).toBe(0)
      counterEnv.step(1)
      expect(counterEnv.stepCount).toBe(1)
      counterEnv.step(1)
      expect(counterEnv.stepCount).toBe(2)
    })

    it('reaches terminal state', () => {
      const counterEnv = createCounterEnv()
      counterEnv.reset()

      // Take 10 steps to increment count to 10
      for (let i = 0; i < 9; i++) {
        const result = counterEnv.step(1)
        expect(result.done).toBe(false)
      }

      const finalResult = counterEnv.step(1)
      expect(finalResult.done).toBe(true)
      expect(finalResult.reward).toBe(10)
    })
  })

  describe('observe()', () => {
    it('returns current observation without stepping', () => {
      const counterEnv = createCounterEnv()
      counterEnv.reset()
      counterEnv.step(1)
      counterEnv.step(1)

      const obs = counterEnv.observe()
      expect(Math.abs(obs[0]! - 0.2)).toBeLessThan(0.001) // 2/10
    })
  })

  describe('getState()', () => {
    it('returns internal state object', () => {
      const counterEnv = createCounterEnv()
      counterEnv.reset()
      counterEnv.step(1)

      const state = counterEnv.getState()
      expect(state.count).toBe(1)
    })
  })

  describe('multi-objective rewards', () => {
    it('supports Float32Array rewards', () => {
      const moEnv = env({
        createState: () => ({ x: 0, y: 0 }),
        step: (state, action) => {
          state.x += action
          state.y -= action
          return {
            reward: new Float32Array([state.x, state.y, 1]),
            done: false,
          }
        },
        observe: (state) => new Float32Array([state.x, state.y]),
        actionSpace: 3,
      })

      moEnv.reset()
      const result = moEnv.step(1)

      expect(result.reward).toBeInstanceOf(Float32Array)
      expect((result.reward as Float32Array).length).toBe(3)
    })
  })

  describe('maxSteps truncation', () => {
    it('truncates episode at max steps', () => {
      const shortEnv = env({
        createState: () => ({ t: 0 }),
        step: (state, _action) => {
          state.t++
          return { reward: 1, done: false }
        },
        observe: (state) => new Float32Array([state.t]),
        maxSteps: 5,
      })

      shortEnv.reset()

      for (let i = 0; i < 4; i++) {
        const result = shortEnv.step(0)
        expect(result.done).toBe(false)
      }

      const finalResult = shortEnv.step(0)
      expect(finalResult.done).toBe(true)
      expect(finalResult.truncated).toBe(true)
    })
  })
})
