/**
 * Tests for Linear layer
 */

import { describe, test, expect } from 'vitest'
import { Linear, type LinearOptions } from '../linear.js'
import { mockTensorFactories } from '@ts-torch/test-utils'
import type { Tensor } from '../../module.js'

describe('Linear', () => {
  describe('constructor', () => {
    test('creates linear layer with default options', () => {
      const linear = new Linear(784, 128)

      expect(linear.inFeatures).toBe(784)
      expect(linear.outFeatures).toBe(128)
      expect(linear.weight).toBeDefined()
      expect(linear.bias).toBeDefined()
      expect(linear.bias).not.toBeNull()
    })

    test('creates linear layer without bias when disabled', () => {
      const linear = new Linear(784, 128, { bias: false })

      expect(linear.inFeatures).toBe(784)
      expect(linear.outFeatures).toBe(128)
      expect(linear.weight).toBeDefined()
      expect(linear.bias).toBeNull()
    })

    test('weight has correct shape [outFeatures, inFeatures]', () => {
      const linear = new Linear(784, 128)

      expect(linear.weight.data.shape).toEqual([128, 784])
    })

    test('bias has correct shape [outFeatures]', () => {
      const linear = new Linear(784, 128)

      expect(linear.bias).not.toBeNull()
      expect(linear.bias?.data.shape).toEqual([128])
    })

    test('parameters are registered', () => {
      const linear = new Linear(784, 128)
      const params = linear.parameters()

      expect(params).toHaveLength(2)
      expect(params).toContain(linear.weight)
      expect(params).toContain(linear.bias)
    })

    test('parameters are registered (no bias)', () => {
      const linear = new Linear(784, 128, { bias: false })
      const params = linear.parameters()

      expect(params).toHaveLength(1)
      expect(params).toContain(linear.weight)
    })

    test('named parameters use correct names', () => {
      const linear = new Linear(784, 128)
      const namedParams = linear.namedParameters()

      expect(namedParams.get('weight')).toBe(linear.weight)
      expect(namedParams.get('bias')).toBe(linear.bias)
    })
  })

  describe('initialization strategies', () => {
    test('kaiming_uniform initialization', () => {
      const linear = new Linear(10, 5, { init: 'kaiming_uniform' })

      expect(linear.weight.data).toBeDefined()
      expect(linear.weight.requiresGrad).toBe(true)
    })

    test('kaiming_normal initialization', () => {
      const linear = new Linear(10, 5, { init: 'kaiming_normal' })

      expect(linear.weight.data).toBeDefined()
      expect(linear.weight.requiresGrad).toBe(true)
    })

    test('xavier_uniform initialization', () => {
      const linear = new Linear(10, 5, { init: 'xavier_uniform' })

      expect(linear.weight.data).toBeDefined()
      expect(linear.weight.requiresGrad).toBe(true)
    })

    test('xavier_normal initialization', () => {
      const linear = new Linear(10, 5, { init: 'xavier_normal' })

      expect(linear.weight.data).toBeDefined()
      expect(linear.weight.requiresGrad).toBe(true)
    })

    test('zeros initialization', () => {
      const linear = new Linear(10, 5, { init: 'zeros' })

      expect(linear.weight.data).toBeDefined()
      expect(linear.weight.requiresGrad).toBe(true)
    })
  })

  describe('forward pass', () => {
    test('transforms input shape correctly', () => {
      const linear = new Linear(784, 128)

      // Create mock input [Batch=32, InFeatures=784]
      const input = mockTensorFactories.randn([32, 784]) as unknown as Tensor<
        readonly [number, 784]
      >

      const output = linear.forward(input)

      // Output should be [Batch=32, OutFeatures=128]
      expect(output.shape).toEqual([32, 128])
    })

    test('handles single sample input', () => {
      const linear = new Linear(10, 5)

      // Create mock input [Batch=1, InFeatures=10]
      const input = mockTensorFactories.randn([1, 10]) as unknown as Tensor<readonly [number, 10]>

      const output = linear.forward(input)

      expect(output.shape).toEqual([1, 5])
    })

    test('handles large batch input', () => {
      const linear = new Linear(128, 64)

      // Create mock input [Batch=256, InFeatures=128]
      const input = mockTensorFactories.randn([256, 128]) as unknown as Tensor<
        readonly [number, 128]
      >

      const output = linear.forward(input)

      expect(output.shape).toEqual([256, 64])
    })

    test('works without bias', () => {
      const linear = new Linear(10, 5, { bias: false })

      const input = mockTensorFactories.randn([4, 10]) as unknown as Tensor<readonly [number, 10]>

      const output = linear.forward(input)

      expect(output.shape).toEqual([4, 5])
    })
  })

  describe('training mode', () => {
    test('is in training mode by default', () => {
      const linear = new Linear(10, 5)

      expect(linear.training).toBe(true)
    })

    test('can be set to evaluation mode', () => {
      const linear = new Linear(10, 5)

      linear.eval()

      expect(linear.training).toBe(false)
    })

    test('can switch between modes', () => {
      const linear = new Linear(10, 5)

      expect(linear.training).toBe(true)

      linear.eval()
      expect(linear.training).toBe(false)

      linear.train()
      expect(linear.training).toBe(true)
    })
  })

  describe('parameter management', () => {
    test('parameters require gradient by default', () => {
      const linear = new Linear(10, 5)

      expect(linear.weight.requiresGrad).toBe(true)
      expect(linear.bias?.requiresGrad).toBe(true)
    })

    test('can zero gradients', () => {
      const linear = new Linear(10, 5)

      // This should not throw
      linear.zeroGrad()
    })
  })

  describe('composition', () => {
    test('can be piped with other modules', () => {
      const layer1 = new Linear(784, 128)
      const layer2 = new Linear(128, 64)

      const piped = layer1.pipe(layer2)

      expect(piped).toBeDefined()

      // Test forward pass
      const input = mockTensorFactories.randn([32, 784]) as unknown as Tensor<
        readonly [number, 784]
      >
      const output = piped.forward(input)

      expect(output.shape).toEqual([32, 64])
    })

    test('piped modules share training mode', () => {
      const layer1 = new Linear(784, 128)
      const layer2 = new Linear(128, 64)
      const piped = layer1.pipe(layer2)

      expect(layer1.training).toBe(true)
      expect(layer2.training).toBe(true)

      piped.eval()

      expect(layer1.training).toBe(false)
      expect(layer2.training).toBe(false)
    })

    test('can chain multiple layers', () => {
      const layer1 = new Linear(784, 256)
      const layer2 = new Linear(256, 128)
      const layer3 = new Linear(128, 64)

      const piped = layer1.pipe(layer2).pipe(layer3)

      const input = mockTensorFactories.randn([16, 784]) as unknown as Tensor<
        readonly [number, 784]
      >
      const output = piped.forward(input)

      expect(output.shape).toEqual([16, 64])
    })
  })

  describe('toString', () => {
    test('returns descriptive string with bias', () => {
      const linear = new Linear(784, 128)
      const str = linear.toString()

      expect(str).toContain('Linear')
      expect(str).toContain('784')
      expect(str).toContain('128')
      expect(str).toContain('bias=true')
    })

    test('returns descriptive string without bias', () => {
      const linear = new Linear(784, 128, { bias: false })
      const str = linear.toString()

      expect(str).toContain('Linear')
      expect(str).toContain('784')
      expect(str).toContain('128')
      expect(str).toContain('bias=false')
    })
  })

  describe('edge cases', () => {
    test('handles small dimensions', () => {
      const linear = new Linear(1, 1)

      expect(linear.weight.data.shape).toEqual([1, 1])
      expect(linear.bias?.data.shape).toEqual([1])
    })

    test('handles large input features', () => {
      const linear = new Linear(1000, 10)

      expect(linear.weight.data.shape).toEqual([10, 1000])
      expect(linear.bias?.data.shape).toEqual([10])
    })

    test('handles large output features', () => {
      const linear = new Linear(10, 1000)

      expect(linear.weight.data.shape).toEqual([1000, 10])
      expect(linear.bias?.data.shape).toEqual([1000])
    })
  })
})
