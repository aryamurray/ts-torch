/**
 * Tests for extended activation functions
 * (ELU, SELU, SiLU, Mish, Hardswish, Hardsigmoid, etc.)
 *
 * Note: Forward pass tests require the native library to be built.
 * These tests focus on constructor/property validation.
 */

import { describe, test, expect } from 'vitest'
import {
  ELU,
  SELU,
  SiLU,
  Swish,
  Mish,
  Hardswish,
  Hardsigmoid,
  Hardtanh,
  ReLU6,
  PReLU,
  Softplus,
  Softsign,
  LogSoftmax,
} from '../activation.js'

describe('ELU', () => {
  test('creates ELU with default alpha', () => {
    const elu = new ELU()

    expect(elu.alpha).toBe(1.0)
  })

  test('creates ELU with custom alpha', () => {
    const elu = new ELU(0.5)

    expect(elu.alpha).toBe(0.5)
  })

  test('has no parameters', () => {
    const elu = new ELU()
    const params = elu.parameters()

    expect(params).toHaveLength(0)
  })

  test('is in training mode by default', () => {
    const elu = new ELU()

    expect(elu.training).toBe(true)
  })

  test('toString includes alpha', () => {
    const elu = new ELU(0.5)
    const str = elu.toString()

    expect(str).toContain('ELU')
    expect(str).toContain('alpha=0.5')
  })
})

describe('SELU', () => {
  test('creates SELU activation', () => {
    const selu = new SELU()

    expect(selu).toBeDefined()
  })

  test('has no parameters', () => {
    const selu = new SELU()
    const params = selu.parameters()

    expect(params).toHaveLength(0)
  })

  test('is in training mode by default', () => {
    const selu = new SELU()

    expect(selu.training).toBe(true)
  })

  test('toString returns SELU()', () => {
    const selu = new SELU()

    expect(selu.toString()).toBe('SELU()')
  })
})

describe('SiLU (Swish)', () => {
  test('creates SiLU activation', () => {
    const silu = new SiLU()

    expect(silu).toBeDefined()
  })

  test('Swish is an alias for SiLU', () => {
    const swish = new Swish()

    expect(swish).toBeDefined()
  })

  test('has no parameters', () => {
    const silu = new SiLU()
    const params = silu.parameters()

    expect(params).toHaveLength(0)
  })

  test('is in training mode by default', () => {
    const silu = new SiLU()

    expect(silu.training).toBe(true)
  })

  test('toString returns SiLU()', () => {
    const silu = new SiLU()

    expect(silu.toString()).toBe('SiLU()')
  })
})

describe('Mish', () => {
  test('creates Mish activation', () => {
    const mish = new Mish()

    expect(mish).toBeDefined()
  })

  test('has no parameters', () => {
    const mish = new Mish()
    const params = mish.parameters()

    expect(params).toHaveLength(0)
  })

  test('is in training mode by default', () => {
    const mish = new Mish()

    expect(mish.training).toBe(true)
  })

  test('toString returns Mish()', () => {
    const mish = new Mish()

    expect(mish.toString()).toBe('Mish()')
  })
})

describe('Hardswish', () => {
  test('creates Hardswish activation', () => {
    const hardswish = new Hardswish()

    expect(hardswish).toBeDefined()
  })

  test('has no parameters', () => {
    const hardswish = new Hardswish()
    const params = hardswish.parameters()

    expect(params).toHaveLength(0)
  })

  test('is in training mode by default', () => {
    const hardswish = new Hardswish()

    expect(hardswish.training).toBe(true)
  })

  test('toString returns Hardswish()', () => {
    const hardswish = new Hardswish()

    expect(hardswish.toString()).toBe('Hardswish()')
  })
})

describe('Hardsigmoid', () => {
  test('creates Hardsigmoid activation', () => {
    const hardsigmoid = new Hardsigmoid()

    expect(hardsigmoid).toBeDefined()
  })

  test('has no parameters', () => {
    const hardsigmoid = new Hardsigmoid()
    const params = hardsigmoid.parameters()

    expect(params).toHaveLength(0)
  })

  test('is in training mode by default', () => {
    const hardsigmoid = new Hardsigmoid()

    expect(hardsigmoid.training).toBe(true)
  })

  test('toString returns Hardsigmoid()', () => {
    const hardsigmoid = new Hardsigmoid()

    expect(hardsigmoid.toString()).toBe('Hardsigmoid()')
  })
})

describe('Hardtanh', () => {
  test('creates Hardtanh with default bounds', () => {
    const hardtanh = new Hardtanh()

    expect(hardtanh.minVal).toBe(-1)
    expect(hardtanh.maxVal).toBe(1)
  })

  test('creates Hardtanh with custom bounds', () => {
    const hardtanh = new Hardtanh(-2, 2)

    expect(hardtanh.minVal).toBe(-2)
    expect(hardtanh.maxVal).toBe(2)
  })

  test('has no parameters', () => {
    const hardtanh = new Hardtanh()
    const params = hardtanh.parameters()

    expect(params).toHaveLength(0)
  })

  test('is in training mode by default', () => {
    const hardtanh = new Hardtanh()

    expect(hardtanh.training).toBe(true)
  })

  test('toString includes bounds', () => {
    const hardtanh = new Hardtanh(-2, 2)
    const str = hardtanh.toString()

    expect(str).toContain('Hardtanh')
    expect(str).toContain('min_val=-2')
    expect(str).toContain('max_val=2')
  })
})

describe('ReLU6', () => {
  test('creates ReLU6 activation', () => {
    const relu6 = new ReLU6()

    expect(relu6).toBeDefined()
  })

  test('has no parameters', () => {
    const relu6 = new ReLU6()
    const params = relu6.parameters()

    expect(params).toHaveLength(0)
  })

  test('is in training mode by default', () => {
    const relu6 = new ReLU6()

    expect(relu6.training).toBe(true)
  })

  test('toString returns ReLU6()', () => {
    const relu6 = new ReLU6()

    expect(relu6.toString()).toBe('ReLU6()')
  })
})

describe('PReLU', () => {
  test('creates PReLU with default parameters', () => {
    const prelu = new PReLU()

    expect(prelu.numParameters).toBe(1)
  })

  test('creates PReLU with multiple parameters', () => {
    const prelu = new PReLU(64)

    expect(prelu.numParameters).toBe(64)
  })

  test('has learnable weight parameter', () => {
    const prelu = new PReLU(64)
    const params = prelu.parameters()

    expect(params).toHaveLength(1)
    expect(params[0].data.shape).toEqual([64])
  })

  test('weight requires gradient', () => {
    const prelu = new PReLU()
    const params = prelu.parameters()

    expect(params[0].requiresGrad).toBe(true)
  })

  test('is in training mode by default', () => {
    const prelu = new PReLU()

    expect(prelu.training).toBe(true)
  })

  test('toString includes parameters', () => {
    const prelu = new PReLU(64)
    const str = prelu.toString()

    expect(str).toContain('PReLU')
    expect(str).toContain('num_parameters=64')
  })
})

describe('Softplus', () => {
  test('creates Softplus with default parameters', () => {
    const softplus = new Softplus()

    expect(softplus.beta).toBe(1)
    expect(softplus.threshold).toBe(20)
  })

  test('creates Softplus with custom parameters', () => {
    const softplus = new Softplus(2, 10)

    expect(softplus.beta).toBe(2)
    expect(softplus.threshold).toBe(10)
  })

  test('has no parameters', () => {
    const softplus = new Softplus()
    const params = softplus.parameters()

    expect(params).toHaveLength(0)
  })

  test('is in training mode by default', () => {
    const softplus = new Softplus()

    expect(softplus.training).toBe(true)
  })

  test('toString includes beta and threshold', () => {
    const softplus = new Softplus(2, 10)
    const str = softplus.toString()

    expect(str).toContain('Softplus')
    expect(str).toContain('beta=2')
    expect(str).toContain('threshold=10')
  })
})

describe('Softsign', () => {
  test('creates Softsign activation', () => {
    const softsign = new Softsign()

    expect(softsign).toBeDefined()
  })

  test('has no parameters', () => {
    const softsign = new Softsign()
    const params = softsign.parameters()

    expect(params).toHaveLength(0)
  })

  test('is in training mode by default', () => {
    const softsign = new Softsign()

    expect(softsign.training).toBe(true)
  })

  test('toString returns Softsign()', () => {
    const softsign = new Softsign()

    expect(softsign.toString()).toBe('Softsign()')
  })
})

describe('LogSoftmax', () => {
  test('creates LogSoftmax with default dimension', () => {
    const logsoftmax = new LogSoftmax()

    expect(logsoftmax.dim).toBe(-1)
  })

  test('creates LogSoftmax with specified dimension', () => {
    const logsoftmax = new LogSoftmax(1)

    expect(logsoftmax.dim).toBe(1)
  })

  test('has no parameters', () => {
    const logsoftmax = new LogSoftmax()
    const params = logsoftmax.parameters()

    expect(params).toHaveLength(0)
  })

  test('is in training mode by default', () => {
    const logsoftmax = new LogSoftmax()

    expect(logsoftmax.training).toBe(true)
  })

  test('toString includes dimension', () => {
    const logsoftmax = new LogSoftmax(1)
    const str = logsoftmax.toString()

    expect(str).toContain('LogSoftmax')
    expect(str).toContain('dim=1')
  })
})
