/**
 * Tests for Module base class and Parameter
 */

import { describe, test, expect, beforeEach } from 'vitest'
import { Module, Parameter, PipedModule, type Tensor, type float32 } from '../module.js'
import { mockTensorFactories } from '@ts-torch/test-utils'
import type { Shape, DType } from '@ts-torch/core'

// Simple test module for testing base functionality
class TestModule<S extends Shape = Shape, D extends DType<string> = float32> extends Module<S, S, D> {
  forward(input: Tensor<S, D>): Tensor<S, D> {
    return input
  }
}

// Module with parameters
class ParameterizedModule<D extends DType<string> = float32> extends Module<
  readonly [number, 3],
  readonly [number, 3],
  D
> {
  readonly weight: Parameter<readonly [3, 3], D>
  readonly bias: Parameter<readonly [3], D>

  constructor() {
    super()
    this.weight = new Parameter(
      mockTensorFactories.ones([3, 3]) as unknown as Tensor<readonly [3, 3], D>,
      true
    )
    this.bias = new Parameter(
      mockTensorFactories.zeros([3]) as unknown as Tensor<readonly [3], D>,
      true
    )
    this.registerParameter('weight', this.weight)
    this.registerParameter('bias', this.bias)
  }

  forward(input: Tensor<readonly [number, 3], D>): Tensor<readonly [number, 3], D> {
    return input
  }
}

// Module with submodules
class NestedModule<D extends DType<string> = float32> extends Module<
  readonly [number, 3],
  readonly [number, 3],
  D
> {
  readonly layer1: ParameterizedModule<D>
  readonly layer2: ParameterizedModule<D>

  constructor() {
    super()
    this.layer1 = new ParameterizedModule<D>()
    this.layer2 = new ParameterizedModule<D>()
    this.registerModule('layer1', this.layer1)
    this.registerModule('layer2', this.layer2)
  }

  forward(input: Tensor<readonly [number, 3], D>): Tensor<readonly [number, 3], D> {
    return this.layer2.forward(this.layer1.forward(input))
  }
}

describe('Parameter', () => {
  test('creates parameter with gradient tracking enabled by default', () => {
    const tensor = mockTensorFactories.zeros([2, 3]) as unknown as Tensor<readonly [2, 3]>
    const param = new Parameter(tensor)

    expect(param.requiresGrad).toBe(true)
    expect(param.data).toBe(tensor)
  })

  test('creates parameter without gradient tracking when disabled', () => {
    const tensor = mockTensorFactories.zeros([2, 3]) as unknown as Tensor<readonly [2, 3]>
    const param = new Parameter(tensor, false)

    expect(param.requiresGrad).toBe(false)
  })

  test('can change requiresGrad after creation', () => {
    const tensor = mockTensorFactories.zeros([2, 3]) as unknown as Tensor<readonly [2, 3]>
    const param = new Parameter(tensor, true)

    expect(param.requiresGrad).toBe(true)

    param.requiresGrad = false
    expect(param.requiresGrad).toBe(false)

    param.requiresGrad = true
    expect(param.requiresGrad).toBe(true)
  })

  test('provides access to gradient', () => {
    const tensor = mockTensorFactories.zeros([2], true) as unknown as Tensor<readonly [2]>
    const param = new Parameter(tensor, true)

    // Initially no gradient
    expect(param.grad).toBeNull()
  })

  test('can zero out gradients', () => {
    const tensor = mockTensorFactories.zeros([2], true) as unknown as Tensor<readonly [2]>
    const param = new Parameter(tensor, true)

    // This should not throw
    param.zeroGrad()
  })
})

describe('Module', () => {
  let module: TestModule

  beforeEach(() => {
    module = new TestModule()
  })

  test('is in training mode by default', () => {
    expect(module.training).toBe(true)
  })

  test('train() sets training mode', () => {
    module.eval()
    expect(module.training).toBe(false)

    module.train()
    expect(module.training).toBe(true)

    module.train(false)
    expect(module.training).toBe(false)
  })

  test('eval() sets evaluation mode', () => {
    expect(module.training).toBe(true)

    module.eval()
    expect(module.training).toBe(false)
  })

  test('train() returns this for chaining', () => {
    const result = module.train()
    expect(result).toBe(module)
  })

  test('eval() returns this for chaining', () => {
    const result = module.eval()
    expect(result).toBe(module)
  })

  test('forward pass is callable via __call__', () => {
    const input = mockTensorFactories.zeros([2, 3]) as unknown as Tensor<readonly [2, 3]>
    const result = module.__call__(input)

    expect(result).toBe(input) // TestModule returns input unchanged
  })

  test('has string representation', () => {
    const str = module.toString()
    expect(str).toBe('TestModule()')
  })
})

describe('Module.parameters()', () => {
  test('returns empty array for module without parameters', () => {
    const module = new TestModule()
    const params = module.parameters()

    expect(params).toEqual([])
  })

  test('returns parameters for module with parameters', () => {
    const module = new ParameterizedModule()
    const params = module.parameters()

    expect(params).toHaveLength(2)
    expect(params).toContain(module.weight)
    expect(params).toContain(module.bias)
  })

  test('returns parameters from nested modules', () => {
    const module = new NestedModule()
    const params = module.parameters()

    // 2 layers * 2 params each = 4 params
    expect(params).toHaveLength(4)
    expect(params).toContain(module.layer1.weight)
    expect(params).toContain(module.layer1.bias)
    expect(params).toContain(module.layer2.weight)
    expect(params).toContain(module.layer2.bias)
  })
})

describe('Module.namedParameters()', () => {
  test('returns empty map for module without parameters', () => {
    const module = new TestModule()
    const namedParams = module.namedParameters()

    expect(namedParams.size).toBe(0)
  })

  test('returns named parameters for module with parameters', () => {
    const module = new ParameterizedModule()
    const namedParams = module.namedParameters()

    expect(namedParams.size).toBe(2)
    expect(namedParams.get('weight')).toBe(module.weight)
    expect(namedParams.get('bias')).toBe(module.bias)
  })

  test('uses dot notation for nested module parameters', () => {
    const module = new NestedModule()
    const namedParams = module.namedParameters()

    expect(namedParams.size).toBe(4)
    expect(namedParams.get('layer1.weight')).toBe(module.layer1.weight)
    expect(namedParams.get('layer1.bias')).toBe(module.layer1.bias)
    expect(namedParams.get('layer2.weight')).toBe(module.layer2.weight)
    expect(namedParams.get('layer2.bias')).toBe(module.layer2.bias)
  })
})

describe('Module.zeroGrad()', () => {
  test('zeroes gradients for all parameters', () => {
    const module = new ParameterizedModule()

    // This should not throw
    module.zeroGrad()
  })

  test('zeroes gradients for nested module parameters', () => {
    const module = new NestedModule()

    // This should not throw
    module.zeroGrad()
  })
})

describe('Module.train() with nested modules', () => {
  test('propagates training mode to submodules', () => {
    const module = new NestedModule()

    expect(module.training).toBe(true)
    expect(module.layer1.training).toBe(true)
    expect(module.layer2.training).toBe(true)

    module.eval()

    expect(module.training).toBe(false)
    expect(module.layer1.training).toBe(false)
    expect(module.layer2.training).toBe(false)

    module.train()

    expect(module.training).toBe(true)
    expect(module.layer1.training).toBe(true)
    expect(module.layer2.training).toBe(true)
  })
})

describe('Module.to()', () => {
  test('returns this for chaining', () => {
    const module = new TestModule()
    const result = module.to('cpu')

    expect(result).toBe(module)
  })
})

describe('PipedModule', () => {
  test('creates piped module from two modules', () => {
    const module1 = new TestModule<readonly [number, 3]>()
    const module2 = new TestModule<readonly [number, 3]>()

    const piped = module1.pipe(module2)

    expect(piped).toBeInstanceOf(PipedModule)
  })

  test('executes forward pass through both modules', () => {
    const module1 = new TestModule<readonly [number, 3]>()
    const module2 = new TestModule<readonly [number, 3]>()
    const piped = module1.pipe(module2)

    const input = mockTensorFactories.zeros([2, 3]) as unknown as Tensor<readonly [number, 3]>
    const output = piped.forward(input)

    expect(output).toBe(input) // Both TestModules return input unchanged
  })

  test('can chain multiple pipes', () => {
    const module1 = new TestModule<readonly [number, 3]>()
    const module2 = new TestModule<readonly [number, 3]>()
    const module3 = new TestModule<readonly [number, 3]>()

    const piped = module1.pipe(module2).pipe(module3)

    expect(piped).toBeInstanceOf(PipedModule)
  })

  test('registers submodules correctly', () => {
    const module1 = new TestModule<readonly [number, 3]>()
    const module2 = new TestModule<readonly [number, 3]>()
    const piped = module1.pipe(module2)

    const params = piped.parameters()
    expect(params).toEqual([]) // TestModules have no parameters
  })

  test('propagates training mode to piped modules', () => {
    const module1 = new TestModule<readonly [number, 3]>()
    const module2 = new TestModule<readonly [number, 3]>()
    const piped = module1.pipe(module2)

    expect(piped.training).toBe(true)
    expect(module1.training).toBe(true)
    expect(module2.training).toBe(true)

    piped.eval()

    expect(piped.training).toBe(false)
    expect(module1.training).toBe(false)
    expect(module2.training).toBe(false)
  })

  test('has descriptive string representation', () => {
    const module1 = new TestModule<readonly [number, 3]>()
    const module2 = new TestModule<readonly [number, 3]>()
    const piped = module1.pipe(module2)

    const str = piped.toString()
    expect(str).toContain('PipedModule')
    expect(str).toContain('TestModule')
  })
})
