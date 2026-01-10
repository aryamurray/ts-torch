/**
 * Tests for scoped memory management
 */

import { describe, it, expect } from 'vitest'
import {
  run,
  runAsync,
  inScope,
  scopeDepth,
  currentScopeId,
  scopeTensorCount,
  registerTensor,
  escapeTensor,
  type ScopedTensor,
} from '../scope'
import type { Pointer } from '../../ffi/error.js'

// Mock tensor for testing
class MockTensor implements ScopedTensor {
  handle: Pointer = 0 as unknown as Pointer
  escaped = false
  freed = false

  constructor(handle: number) {
    this.handle = handle as unknown as Pointer
  }

  markEscaped(): void {
    this.escaped = true
  }

  free(): void {
    this.freed = true
  }
}

describe('Scope Management', () => {
  describe('inScope', () => {
    it('returns false outside scope', () => {
      expect(inScope()).toBe(false)
    })

    it('returns true inside scope', () => {
      run(() => {
        expect(inScope()).toBe(true)
      })
    })

    it('returns false after scope exits', () => {
      run(() => {
        expect(inScope()).toBe(true)
      })
      expect(inScope()).toBe(false)
    })
  })

  describe('scopeDepth', () => {
    it('returns 0 outside scope', () => {
      expect(scopeDepth()).toBe(0)
    })

    it('returns 1 in single scope', () => {
      run(() => {
        expect(scopeDepth()).toBe(1)
      })
    })

    it('tracks nested scopes correctly', () => {
      run(() => {
        expect(scopeDepth()).toBe(1)
        run(() => {
          expect(scopeDepth()).toBe(2)
          run(() => {
            expect(scopeDepth()).toBe(3)
          })
          expect(scopeDepth()).toBe(2)
        })
        expect(scopeDepth()).toBe(1)
      })
      expect(scopeDepth()).toBe(0)
    })
  })

  describe('currentScopeId', () => {
    it('returns -1 outside scope', () => {
      expect(currentScopeId()).toBe(-1)
    })

    it('returns valid ID inside scope', () => {
      run(() => {
        const id = currentScopeId()
        expect(id).toBeGreaterThanOrEqual(0)
      })
    })

    it('different scopes have different IDs', () => {
      const ids: number[] = []
      run(() => {
        ids.push(currentScopeId())
      })
      run(() => {
        ids.push(currentScopeId())
      })
      expect(ids[0]).not.toBe(ids[1])
    })
  })

  describe('run', () => {
    it('executes function and returns result', () => {
      const result = run(() => {
        return 42
      })
      expect(result).toBe(42)
    })

    it('preserves return type', () => {
      const obj = run(() => {
        return { value: 100, nested: { key: 'test' } }
      })
      expect(obj.value).toBe(100)
      expect(obj.nested.key).toBe('test')
    })

    it('handles exceptions', () => {
      expect(() => {
        run(() => {
          throw new Error('Test error')
        })
      }).toThrow('Test error')
    })

    it('cleans up scope after exception', () => {
      expect(() => {
        run(() => {
          throw new Error('Test error')
        })
      }).toThrow()

      expect(inScope()).toBe(false)
      expect(scopeDepth()).toBe(0)
    })
  })

  describe('runAsync', () => {
    it('executes async function and returns result', async () => {
      const result = await runAsync(async () => {
        await new Promise((resolve) => setTimeout(resolve, 10))
        return 42
      })
      expect(result).toBe(42)
    })

    it('maintains scope during async operations', async () => {
      const checks: boolean[] = []
      await runAsync(async () => {
        checks.push(inScope())
        await new Promise((resolve) => setTimeout(resolve, 10))
        checks.push(inScope())
      })
      expect(checks).toEqual([true, true])
      expect(inScope()).toBe(false)
    })

    it('handles async exceptions', async () => {
      await expect(
        runAsync(async () => {
          await new Promise((resolve) => setTimeout(resolve, 10))
          throw new Error('Async error')
        }),
      ).rejects.toThrow('Async error')

      expect(inScope()).toBe(false)
    })
  })

  describe('registerTensor', () => {
    it('does nothing outside scope', () => {
      const tensor = new MockTensor(1)
      expect(() => registerTensor(tensor)).not.toThrow()
    })

    it('tracks tensor inside scope', () => {
      run(() => {
        const tensor = new MockTensor(1)
        registerTensor(tensor)
        expect(scopeTensorCount()).toBe(1)
      })
    })

    it('tracks multiple tensors', () => {
      run(() => {
        for (let i = 0; i < 5; i++) {
          const tensor = new MockTensor(i)
          registerTensor(tensor)
        }
        expect(scopeTensorCount()).toBe(5)
      })
    })

    it('nested scopes track tensors separately', () => {
      run(() => {
        registerTensor(new MockTensor(1))
        expect(scopeTensorCount()).toBe(1)

        run(() => {
          registerTensor(new MockTensor(2))
          registerTensor(new MockTensor(3))
          expect(scopeTensorCount()).toBe(2)
        })

        expect(scopeTensorCount()).toBe(1)
      })
    })
  })

  describe('escapeTensor', () => {
    it('throws outside scope', () => {
      const tensor = new MockTensor(1)
      expect(() => escapeTensor(tensor)).toThrow('not currently in a scope')
    })

    it('marks tensor as escaped', () => {
      run(() => {
        const tensor = new MockTensor(1)
        registerTensor(tensor)
        expect(tensor.escaped).toBe(false)

        escapeTensor(tensor)
        expect(tensor.escaped).toBe(true)
      })
    })

    it('returns the same tensor for chaining', () => {
      run(() => {
        const tensor = new MockTensor(1)
        registerTensor(tensor)
        const escaped = escapeTensor(tensor)
        expect(escaped).toBe(tensor)
      })
    })
  })

  describe('scopeTensorCount', () => {
    it('returns 0 outside scope', () => {
      expect(scopeTensorCount()).toBe(0)
    })

    it('counts registered tensors', () => {
      run(() => {
        expect(scopeTensorCount()).toBe(0)
        registerTensor(new MockTensor(1))
        expect(scopeTensorCount()).toBe(1)
        registerTensor(new MockTensor(2))
        expect(scopeTensorCount()).toBe(2)
      })
    })

    it('resets after scope exit', () => {
      run(() => {
        registerTensor(new MockTensor(1))
        registerTensor(new MockTensor(2))
        expect(scopeTensorCount()).toBe(2)
      })
      expect(scopeTensorCount()).toBe(0)
    })
  })

  describe('Integration scenarios', () => {
    it('simple scope with escape', () => {
      let escapedTensor: MockTensor | null = null

      run(() => {
        const temp1 = new MockTensor(1)
        const temp2 = new MockTensor(2)
        const keep = new MockTensor(3)

        registerTensor(temp1)
        registerTensor(temp2)
        registerTensor(keep)

        escapedTensor = escapeTensor(keep)

        expect(scopeTensorCount()).toBe(3)
        expect(escapedTensor.escaped).toBe(true)
        expect(temp1.escaped).toBe(false)
        expect(temp2.escaped).toBe(false)
      })

      expect(escapedTensor).not.toBeNull()
      expect(escapedTensor!.escaped).toBe(true)
    })

    it('nested scopes with selective escaping', () => {
      const results: MockTensor[] = []

      run(() => {
        const outer1 = new MockTensor(1)
        registerTensor(outer1)

        run(() => {
          const inner1 = new MockTensor(2)
          const inner2 = new MockTensor(3)
          registerTensor(inner1)
          registerTensor(inner2)

          results.push(escapeTensor(inner1))
        })

        results.push(escapeTensor(outer1))
      })

      expect(results).toHaveLength(2)
      expect(results[0]!.escaped).toBe(true)
      expect(results[1]!.escaped).toBe(true)
    })

    it('async scope with tensor lifecycle', async () => {
      const tensor = await runAsync(async () => {
        const t = new MockTensor(1)
        registerTensor(t)

        await new Promise((resolve) => setTimeout(resolve, 10))

        return escapeTensor(t)
      })

      expect(tensor.escaped).toBe(true)
    })
  })
})
