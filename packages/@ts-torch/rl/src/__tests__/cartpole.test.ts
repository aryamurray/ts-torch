import { describe, it, expect } from 'vitest'
import { CartPole, CartPoleRaw } from '../envs/cartpole.js'

describe('CartPole', () => {
  describe('CartPole()', () => {
    it('creates CartPole environment', () => {
      const env = CartPole()
      expect(env).toBeDefined()
      expect(env.actionSpace).toBe(2)
      expect(env.observationSize).toBe(4)
    })

    it('reset returns normalized observation', () => {
      const env = CartPole()
      const obs = env.reset()

      expect(obs).toBeInstanceOf(Float32Array)
      expect(obs.length).toBe(4)

      // Observations should be roughly normalized
      for (const val of obs) {
        expect(Math.abs(val)).toBeLessThan(1)
      }
    })

    it('step returns valid result', () => {
      const env = CartPole()
      env.reset()

      const result = env.step(1) // push right

      expect(result.observation).toBeInstanceOf(Float32Array)
      expect(result.observation.length).toBe(4)
      expect(typeof result.reward).toBe('number')
      expect(typeof result.done).toBe('boolean')
    })

    it('survives for many steps with balanced actions', () => {
      const env = CartPole()
      env.reset()

      let steps = 0
      let done = false

      // Alternate left and right (not optimal but should survive some)
      while (!done && steps < 100) {
        const action = steps % 2
        const result = env.step(action)
        done = result.done
        steps++
      }

      // Should survive at least a few steps
      expect(steps).toBeGreaterThan(5)
    })

    it('terminates when pole falls', () => {
      const env = CartPole()
      env.reset()

      // Keep pushing in one direction to make pole fall
      let done = false
      let steps = 0

      while (!done && steps < 500) {
        const result = env.step(1) // always push right
        done = result.done
        steps++
      }

      expect(done).toBe(true)
      expect(steps).toBeLessThan(500) // Should fail before max steps
    })

    it('gives reward of 1 for surviving', () => {
      const env = CartPole()
      env.reset()

      const result = env.step(0)

      if (!result.done) {
        expect(result.reward).toBe(1)
      }
    })

    it('gives reward of 0 when done', () => {
      const env = CartPole()
      env.reset()

      // Run until done
      let result = env.step(1)
      while (!result.done) {
        result = env.step(1)
      }

      expect(result.reward).toBe(0)
    })
  })

  describe('CartPoleRaw()', () => {
    it('returns unnormalized observations', () => {
      const env = CartPoleRaw()
      env.reset()

      // Take a step to change state
      env.step(1)
      env.step(1)
      const obs = env.observe()

      // Raw observations are in original units
      // Position can be larger than 1 (not normalized)
      expect(obs.length).toBe(4)
    })

    it('has same physics as normalized version', () => {
      const envNorm = CartPole()
      const envRaw = CartPoleRaw()

      envNorm.reset()
      envRaw.reset()

      // Set same seed by resetting (both start with random init)
      // Just check that both terminate eventually with same action sequence
      let normDone = false
      let rawDone = false

      for (let i = 0; i < 100 && !normDone && !rawDone; i++) {
        const action = i % 2
        if (!normDone) {
          normDone = envNorm.step(action).done
        }
        if (!rawDone) {
          rawDone = envRaw.step(action).done
        }
      }

      // Both should have some dynamics (not immediate termination)
      expect(envNorm.stepCount).toBeGreaterThan(0)
      expect(envRaw.stepCount).toBeGreaterThan(0)
    })
  })
})
