/**
 * Tests for container modules (Sequential, SequentialBuilder)
 */

import { describe, test, expect } from 'vitest'
import { Sequential, SequentialBuilder, sequential } from '../container.js'
import { Module, type Tensor } from '../../module.js'
import { mockTensorFactories } from '@ts-torch/test-utils'
import type { Shape, DType } from '@ts-torch/core'

// Simple test modules for composition
class Identity<S extends Shape = Shape, D extends DType<string> = any> extends Module<S, S, D> {
  forward(input: Tensor<S, D>): Tensor<S, D> {
    return input
  }

  override toString(): string {
    return 'Identity()'
  }
}

class Doubler<S extends Shape = Shape, D extends DType<string> = any> extends Module<S, S, D> {
  forward(input: Tensor<S, D>): Tensor<S, D> {
    // In real implementation, would multiply by 2
    return input
  }

  override toString(): string {
    return 'Doubler()'
  }
}

describe('Sequential', () => {
  describe('constructor', () => {
    test('creates sequential with single module', () => {
      const module = new Identity()
      const seq = new Sequential(module)

      expect(seq).toBeDefined()
      expect(seq.length).toBe(1)
    })

    test('creates sequential with multiple modules', () => {
      const module1 = new Identity()
      const module2 = new Doubler()
      const seq = new Sequential(module1, module2)

      expect(seq.length).toBe(2)
    })

    test('throws error when no modules provided', () => {
      expect(() => new Sequential()).toThrow('Sequential requires at least one module')
    })

    test('registers all modules', () => {
      const module1 = new Identity()
      const module2 = new Doubler()
      const module3 = new Identity()
      const seq = new Sequential(module1, module2, module3)

      // Modules should be registered with numeric keys
      expect(seq.at(0)).toBe(module1)
      expect(seq.at(1)).toBe(module2)
      expect(seq.at(2)).toBe(module3)
    })
  })

  describe('forward pass', () => {
    test('executes single module', () => {
      const module = new Identity<readonly [number, 10]>()
      const seq = new Sequential(module)

      const input = mockTensorFactories.randn([32, 10]) as unknown as Tensor<
        readonly [number, 10]
      >
      const output = seq.forward(input)

      expect(output).toBe(input) // Identity returns input unchanged
    })

    test('executes multiple modules in sequence', () => {
      const module1 = new Identity<readonly [number, 5]>()
      const module2 = new Doubler<readonly [number, 5]>()
      const module3 = new Identity<readonly [number, 5]>()
      const seq = new Sequential(module1, module2, module3)

      const input = mockTensorFactories.randn([16, 5]) as unknown as Tensor<readonly [number, 5]>
      const output = seq.forward(input)

      expect(output).toBeDefined()
    })

    test('passes output of one module as input to next', () => {
      const module1 = new Identity<readonly [number, 8]>()
      const module2 = new Identity<readonly [number, 8]>()
      const seq = new Sequential(module1, module2)

      const input = mockTensorFactories.randn([4, 8]) as unknown as Tensor<readonly [number, 8]>
      const output = seq.forward(input)

      expect(output).toBeDefined()
    })
  })

  describe('module access', () => {
    test('at() returns module at index', () => {
      const module1 = new Identity()
      const module2 = new Doubler()
      const seq = new Sequential(module1, module2)

      expect(seq.at(0)).toBe(module1)
      expect(seq.at(1)).toBe(module2)
    })

    test('at() returns undefined for invalid index', () => {
      const module = new Identity()
      const seq = new Sequential(module)

      expect(seq.at(5)).toBeUndefined()
      expect(seq.at(-1)).toBeUndefined()
    })

    test('length property returns correct count', () => {
      const seq1 = new Sequential(new Identity())
      const seq2 = new Sequential(new Identity(), new Doubler())
      const seq3 = new Sequential(new Identity(), new Doubler(), new Identity())

      expect(seq1.length).toBe(1)
      expect(seq2.length).toBe(2)
      expect(seq3.length).toBe(3)
    })

    test('is iterable', () => {
      const module1 = new Identity()
      const module2 = new Doubler()
      const module3 = new Identity()
      const seq = new Sequential(module1, module2, module3)

      const modules = Array.from(seq)

      expect(modules).toHaveLength(3)
      expect(modules[0]).toBe(module1)
      expect(modules[1]).toBe(module2)
      expect(modules[2]).toBe(module3)
    })
  })

  describe('append', () => {
    test('appends module to sequential', () => {
      const module1 = new Identity<readonly [number, 3]>()
      const seq1 = new Sequential(module1)

      const module2 = new Doubler<readonly [number, 3]>()
      const seq2 = seq1.append(module2)

      expect(seq1.length).toBe(1)
      expect(seq2.length).toBe(2)
    })

    test('returns new Sequential instance', () => {
      const module1 = new Identity<readonly [number, 3]>()
      const seq1 = new Sequential(module1)

      const module2 = new Doubler<readonly [number, 3]>()
      const seq2 = seq1.append(module2)

      expect(seq2).not.toBe(seq1)
      expect(seq2).toBeInstanceOf(Sequential)
    })

    test('can chain multiple appends', () => {
      const module1 = new Identity<readonly [number, 4]>()
      const seq = new Sequential(module1)
        .append(new Doubler<readonly [number, 4]>())
        .append(new Identity<readonly [number, 4]>())

      expect(seq.length).toBe(3)
    })
  })

  describe('training mode', () => {
    test('is in training mode by default', () => {
      const seq = new Sequential(new Identity())

      expect(seq.training).toBe(true)
    })

    test('propagates training mode to submodules', () => {
      const module1 = new Identity()
      const module2 = new Doubler()
      const seq = new Sequential(module1, module2)

      expect(module1.training).toBe(true)
      expect(module2.training).toBe(true)

      seq.eval()

      expect(seq.training).toBe(false)
      expect(module1.training).toBe(false)
      expect(module2.training).toBe(false)
    })

    test('can switch between modes', () => {
      const module1 = new Identity()
      const module2 = new Doubler()
      const seq = new Sequential(module1, module2)

      seq.eval()
      expect(seq.training).toBe(false)
      expect(module1.training).toBe(false)
      expect(module2.training).toBe(false)

      seq.train()
      expect(seq.training).toBe(true)
      expect(module1.training).toBe(true)
      expect(module2.training).toBe(true)
    })
  })

  describe('toString', () => {
    test('returns formatted string representation', () => {
      const module1 = new Identity()
      const module2 = new Doubler()
      const seq = new Sequential(module1, module2)

      const str = seq.toString()

      expect(str).toContain('Sequential')
      expect(str).toContain('(0):')
      expect(str).toContain('(1):')
      expect(str).toContain('Identity()')
      expect(str).toContain('Doubler()')
    })

    test('includes all modules in string', () => {
      const seq = new Sequential(new Identity(), new Doubler(), new Identity())

      const str = seq.toString()

      expect(str).toContain('(0):')
      expect(str).toContain('(1):')
      expect(str).toContain('(2):')
    })
  })
})

describe('SequentialBuilder', () => {
  describe('create and build', () => {
    test('creates empty builder', () => {
      const builder = SequentialBuilder.create<readonly [number, 10]>()

      expect(builder).toBeDefined()
    })

    test('builds sequential with single module', () => {
      const module = new Identity<readonly [number, 10]>()
      const seq = SequentialBuilder.create<readonly [number, 10]>().add(module).build()

      expect(seq.length).toBe(1)
    })

    test('builds sequential with multiple modules', () => {
      const module1 = new Identity<readonly [number, 5]>()
      const module2 = new Doubler<readonly [number, 5]>()
      const seq = SequentialBuilder.create<readonly [number, 5]>().add(module1).add(module2).build()

      expect(seq.length).toBe(2)
    })
  })

  describe('add', () => {
    test('returns new builder with added module', () => {
      const builder1 = SequentialBuilder.create<readonly [number, 8]>()
      const module = new Identity<readonly [number, 8]>()
      const builder2 = builder1.add(module)

      expect(builder2).toBeDefined()
      expect(builder2).not.toBe(builder1)
    })

    test('can chain multiple adds', () => {
      const seq = SequentialBuilder.create<readonly [number, 3]>()
        .add(new Identity<readonly [number, 3]>())
        .add(new Doubler<readonly [number, 3]>())
        .add(new Identity<readonly [number, 3]>())
        .build()

      expect(seq.length).toBe(3)
    })
  })

  describe('type inference', () => {
    test('tracks input and output shapes through chain', () => {
      // This is primarily a type-level test - if it compiles, it works
      const builder = SequentialBuilder.create<readonly [number, 10]>()
        .add(new Identity<readonly [number, 10]>())
        .add(new Doubler<readonly [number, 10]>())

      const seq = builder.build()

      expect(seq).toBeInstanceOf(Sequential)
    })
  })
})

describe('sequential helper function', () => {
  test('creates sequential builder', () => {
    const builder = sequential<readonly [number, 16]>()

    expect(builder).toBeDefined()
    expect(builder).toBeInstanceOf(SequentialBuilder)
  })

  test('can build sequential with helper', () => {
    const seq = sequential<readonly [number, 7]>()
      .add(new Identity<readonly [number, 7]>())
      .add(new Doubler<readonly [number, 7]>())
      .build()

    expect(seq).toBeInstanceOf(Sequential)
    expect(seq.length).toBe(2)
  })

  test('provides clean chaining syntax', () => {
    const seq = sequential<readonly [number, 12]>()
      .add(new Identity<readonly [number, 12]>())
      .add(new Doubler<readonly [number, 12]>())
      .add(new Identity<readonly [number, 12]>())
      .add(new Doubler<readonly [number, 12]>())
      .build()

    expect(seq.length).toBe(4)
  })
})

describe('Sequential composition', () => {
  test('sequential can be used inside another sequential', () => {
    const inner = new Sequential(new Identity<readonly [number, 6]>(), new Doubler<readonly [number, 6]>())
    const outer = new Sequential(inner, new Identity<readonly [number, 6]>())

    expect(outer.length).toBe(2)
  })

  test('sequential can be piped', () => {
    const seq1 = new Sequential(new Identity<readonly [number, 4]>())
    const seq2 = new Sequential(new Doubler<readonly [number, 4]>())

    const piped = seq1.pipe(seq2)

    const input = mockTensorFactories.randn([8, 4]) as unknown as Tensor<readonly [number, 4]>
    const output = piped.forward(input)

    expect(output).toBeDefined()
  })

  test('sequential collects parameters from all modules', () => {
    // This would work with modules that have parameters
    const module1 = new Identity<readonly [number, 3]>()
    const module2 = new Identity<readonly [number, 3]>()
    const seq = new Sequential(module1, module2)

    const params = seq.parameters()

    // Identity modules have no parameters
    expect(params).toHaveLength(0)
  })
})

describe('Sequential edge cases', () => {
  test('handles very long sequences', () => {
    const modules: Module<readonly [number, 2], readonly [number, 2]>[] = []
    for (let i = 0; i < 100; i++) {
      modules.push(new Identity<readonly [number, 2]>())
    }

    const seq = new Sequential(...modules)

    expect(seq.length).toBe(100)
  })

  test('handles complex nesting', () => {
    const seq1 = new Sequential(new Identity<readonly [number, 5]>(), new Doubler<readonly [number, 5]>())
    const seq2 = new Sequential(new Doubler<readonly [number, 5]>(), new Identity<readonly [number, 5]>())
    const outer = new Sequential(seq1, seq2)

    expect(outer.length).toBe(2)
  })

  test('iterator can be used multiple times', () => {
    const seq = new Sequential(new Identity(), new Doubler(), new Identity())

    const modules1 = Array.from(seq)
    const modules2 = Array.from(seq)

    expect(modules1).toHaveLength(3)
    expect(modules2).toHaveLength(3)
    expect(modules1[0]).toBe(modules2[0])
  })
})
