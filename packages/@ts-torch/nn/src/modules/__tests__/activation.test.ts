/**
 * Tests for activation function modules
 */

import { describe, test, expect } from 'vitest'
import { ReLU, Sigmoid, Tanh, Softmax, LeakyReLU, GELU } from '../activation.js'
import { mockTensorFactories } from '@ts-torch/test-utils'
import type { Tensor } from '../../module.js'

describe('ReLU', () => {
  test('creates ReLU with default options', () => {
    const relu = new ReLU()

    expect(relu.inplace).toBe(false)
  })

  test('creates ReLU with inplace option', () => {
    const relu = new ReLU(true)

    expect(relu.inplace).toBe(true)
  })

  test('preserves input shape', () => {
    const relu = new ReLU()
    const input = mockTensorFactories.randn([32, 128]) as unknown as Tensor<readonly [32, 128]>

    const output = relu.forward(input)

    expect(output.shape).toEqual([32, 128])
  })

  test('can be used in pipeline', () => {
    const relu = new ReLU()
    const input = mockTensorFactories.randn([16, 64]) as unknown as Tensor<readonly [16, 64]>

    const output = relu.forward(input)

    expect(output).toBeDefined()
  })

  test('has no parameters', () => {
    const relu = new ReLU()
    const params = relu.parameters()

    expect(params).toHaveLength(0)
  })

  test('is in training mode by default', () => {
    const relu = new ReLU()

    expect(relu.training).toBe(true)
  })

  test('toString includes inplace option', () => {
    const relu1 = new ReLU(false)
    const relu2 = new ReLU(true)

    expect(relu1.toString()).toContain('inplace=false')
    expect(relu2.toString()).toContain('inplace=true')
  })
})

describe('Sigmoid', () => {
  test('creates Sigmoid activation', () => {
    const sigmoid = new Sigmoid()

    expect(sigmoid).toBeDefined()
  })

  test('preserves input shape', () => {
    const sigmoid = new Sigmoid()
    const input = mockTensorFactories.randn([32, 10]) as unknown as Tensor<readonly [32, 10]>

    const output = sigmoid.forward(input)

    expect(output.shape).toEqual([32, 10])
  })

  test('has no parameters', () => {
    const sigmoid = new Sigmoid()
    const params = sigmoid.parameters()

    expect(params).toHaveLength(0)
  })

  test('toString returns Sigmoid()', () => {
    const sigmoid = new Sigmoid()

    expect(sigmoid.toString()).toBe('Sigmoid()')
  })

  test('works with different tensor shapes', () => {
    const sigmoid = new Sigmoid()

    const scalar = mockTensorFactories.randn([]) as unknown as Tensor<readonly []>
    const vector = mockTensorFactories.randn([10]) as unknown as Tensor<readonly [10]>
    const matrix = mockTensorFactories.randn([5, 5]) as unknown as Tensor<readonly [5, 5]>

    expect(sigmoid.forward(scalar).shape).toEqual([])
    expect(sigmoid.forward(vector).shape).toEqual([10])
    expect(sigmoid.forward(matrix).shape).toEqual([5, 5])
  })
})

describe('Tanh', () => {
  test('creates Tanh activation', () => {
    const tanh = new Tanh()

    expect(tanh).toBeDefined()
  })

  test('preserves input shape', () => {
    const tanh = new Tanh()
    const input = mockTensorFactories.randn([32, 64]) as unknown as Tensor<readonly [32, 64]>

    const output = tanh.forward(input)

    expect(output.shape).toEqual([32, 64])
  })

  test('has no parameters', () => {
    const tanh = new Tanh()
    const params = tanh.parameters()

    expect(params).toHaveLength(0)
  })

  test('toString returns Tanh()', () => {
    const tanh = new Tanh()

    expect(tanh.toString()).toBe('Tanh()')
  })

  test('works with different tensor shapes', () => {
    const tanh = new Tanh()

    const vector = mockTensorFactories.randn([100]) as unknown as Tensor<readonly [100]>
    const batch = mockTensorFactories.randn([16, 50]) as unknown as Tensor<readonly [16, 50]>

    expect(tanh.forward(vector).shape).toEqual([100])
    expect(tanh.forward(batch).shape).toEqual([16, 50])
  })
})

describe('Softmax', () => {
  test('creates Softmax with default dimension', () => {
    const softmax = new Softmax()

    expect(softmax.dim).toBe(-1)
  })

  test('creates Softmax with specified dimension', () => {
    const softmax = new Softmax(1)

    expect(softmax.dim).toBe(1)
  })

  test('preserves input shape', () => {
    const softmax = new Softmax()
    const input = mockTensorFactories.randn([32, 10]) as unknown as Tensor<readonly [32, 10]>

    const output = softmax.forward(input)

    expect(output.shape).toEqual([32, 10])
  })

  test('has no parameters', () => {
    const softmax = new Softmax()
    const params = softmax.parameters()

    expect(params).toHaveLength(0)
  })

  test('toString includes dimension', () => {
    const softmax1 = new Softmax()
    const softmax2 = new Softmax(0)

    expect(softmax1.toString()).toContain('dim=-1')
    expect(softmax2.toString()).toContain('dim=0')
  })

  test('works with different dimensions', () => {
    const softmax0 = new Softmax(0)
    const softmax1 = new Softmax(1)
    const softmaxNeg1 = new Softmax(-1)

    const input = mockTensorFactories.randn([10, 20]) as unknown as Tensor<readonly [10, 20]>

    expect(softmax0.forward(input).shape).toEqual([10, 20])
    expect(softmax1.forward(input).shape).toEqual([10, 20])
    expect(softmaxNeg1.forward(input).shape).toEqual([10, 20])
  })
})

describe('LeakyReLU', () => {
  test('creates LeakyReLU with default negative slope', () => {
    const leaky = new LeakyReLU()

    expect(leaky.negativeSlope).toBe(0.01)
    expect(leaky.inplace).toBe(false)
  })

  test('creates LeakyReLU with custom negative slope', () => {
    const leaky = new LeakyReLU(0.2)

    expect(leaky.negativeSlope).toBe(0.2)
  })

  test('creates LeakyReLU with inplace option', () => {
    const leaky = new LeakyReLU(0.01, true)

    expect(leaky.negativeSlope).toBe(0.01)
    expect(leaky.inplace).toBe(true)
  })

  test('preserves input shape', () => {
    const leaky = new LeakyReLU()
    const input = mockTensorFactories.randn([32, 128]) as unknown as Tensor<readonly [32, 128]>

    const output = leaky.forward(input)

    expect(output.shape).toEqual([32, 128])
  })

  test('has no parameters', () => {
    const leaky = new LeakyReLU()
    const params = leaky.parameters()

    expect(params).toHaveLength(0)
  })

  test('toString includes negative slope and inplace', () => {
    const leaky1 = new LeakyReLU()
    const leaky2 = new LeakyReLU(0.2, true)

    expect(leaky1.toString()).toContain('negative_slope=0.01')
    expect(leaky1.toString()).toContain('inplace=false')
    expect(leaky2.toString()).toContain('negative_slope=0.2')
    expect(leaky2.toString()).toContain('inplace=true')
  })

  test('works with different tensor shapes', () => {
    const leaky = new LeakyReLU(0.1)

    const vector = mockTensorFactories.randn([256]) as unknown as Tensor<readonly [256]>
    const batch = mockTensorFactories.randn([64, 128]) as unknown as Tensor<readonly [64, 128]>

    expect(leaky.forward(vector).shape).toEqual([256])
    expect(leaky.forward(batch).shape).toEqual([64, 128])
  })
})

describe('GELU', () => {
  test('creates GELU activation', () => {
    const gelu = new GELU()

    expect(gelu).toBeDefined()
  })

  test('preserves input shape', () => {
    const gelu = new GELU()
    const input = mockTensorFactories.randn([32, 768]) as unknown as Tensor<readonly [32, 768]>

    const output = gelu.forward(input)

    expect(output.shape).toEqual([32, 768])
  })

  test('has no parameters', () => {
    const gelu = new GELU()
    const params = gelu.parameters()

    expect(params).toHaveLength(0)
  })

  test('toString returns GELU()', () => {
    const gelu = new GELU()

    expect(gelu.toString()).toBe('GELU()')
  })

  test('works with transformer-like dimensions', () => {
    const gelu = new GELU()

    // Common transformer dimensions
    const input1 = mockTensorFactories.randn([32, 512]) as unknown as Tensor<readonly [32, 512]>
    const input2 = mockTensorFactories.randn([16, 1024]) as unknown as Tensor<
      readonly [16, 1024]
    >
    const input3 = mockTensorFactories.randn([8, 2048]) as unknown as Tensor<readonly [8, 2048]>

    expect(gelu.forward(input1).shape).toEqual([32, 512])
    expect(gelu.forward(input2).shape).toEqual([16, 1024])
    expect(gelu.forward(input3).shape).toEqual([8, 2048])
  })
})

describe('Activation composition', () => {
  test('activations can be chained with pipe', () => {
    const relu = new ReLU<readonly [number, 128]>()
    const sigmoid = new Sigmoid<readonly [number, 128]>()

    const piped = relu.pipe(sigmoid)

    const input = mockTensorFactories.randn([32, 128]) as unknown as Tensor<
      readonly [number, 128]
    >
    const output = piped.forward(input)

    expect(output.shape).toEqual([32, 128])
  })

  test('training mode propagates through piped activations', () => {
    const relu = new ReLU<readonly [number, 64]>()
    const tanh = new Tanh<readonly [number, 64]>()
    const piped = relu.pipe(tanh)

    expect(relu.training).toBe(true)
    expect(tanh.training).toBe(true)

    piped.eval()

    expect(relu.training).toBe(false)
    expect(tanh.training).toBe(false)
  })

  test('multiple activations can be chained', () => {
    const relu = new ReLU<readonly [number, 32]>()
    const tanh = new Tanh<readonly [number, 32]>()
    const sigmoid = new Sigmoid<readonly [number, 32]>()

    const piped = relu.pipe(tanh).pipe(sigmoid)

    const input = mockTensorFactories.randn([16, 32]) as unknown as Tensor<readonly [number, 32]>
    const output = piped.forward(input)

    expect(output.shape).toEqual([16, 32])
  })
})

describe('Activation edge cases', () => {
  test('activations work with scalar tensors', () => {
    const relu = new ReLU<readonly []>()
    const sigmoid = new Sigmoid<readonly []>()
    const tanh = new Tanh<readonly []>()

    const scalar = mockTensorFactories.randn([]) as unknown as Tensor<readonly []>

    expect(relu.forward(scalar).shape).toEqual([])
    expect(sigmoid.forward(scalar).shape).toEqual([])
    expect(tanh.forward(scalar).shape).toEqual([])
  })

  test('activations work with high-dimensional tensors', () => {
    const relu = new ReLU<readonly [number, 3, 224, 224]>()

    const input = mockTensorFactories.randn([8, 3, 224, 224]) as unknown as Tensor<
      readonly [number, 3, 224, 224]
    >
    const output = relu.forward(input)

    expect(output.shape).toEqual([8, 3, 224, 224])
  })

  test('activations work with batch size 1', () => {
    const sigmoid = new Sigmoid<readonly [1, 10]>()

    const input = mockTensorFactories.randn([1, 10]) as unknown as Tensor<readonly [1, 10]>
    const output = sigmoid.forward(input)

    expect(output.shape).toEqual([1, 10])
  })

  test('activations work with large batch sizes', () => {
    const gelu = new GELU<readonly [number, 512]>()

    const input = mockTensorFactories.randn([1024, 512]) as unknown as Tensor<
      readonly [number, 512]
    >
    const output = gelu.forward(input)

    expect(output.shape).toEqual([1024, 512])
  })
})
