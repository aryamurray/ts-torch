import { describe, test, expect } from 'vitest'
import { device, run } from '@ts-torch/core'
import { Flatten } from '../flatten.js'

const cpu = device.cpu()

describe('Flatten', () => {
  test('flattens spatial dims with default args (startDim=1, endDim=-1)', () => {
    run(() => {
      const flatten = new Flatten()
      const input = cpu.zeros([2, 3, 4, 5])
      const output = flatten.forward(input)
      expect([...output.shape]).toEqual([2, 60])
    })
  })

  test('flattens all dims when startDim=0', () => {
    run(() => {
      const flatten = new Flatten(0)
      const input = cpu.zeros([2, 3, 4])
      const output = flatten.forward(input)
      expect([...output.shape]).toEqual([24])
    })
  })

  test('flattens partial range', () => {
    run(() => {
      const flatten = new Flatten(1, 2)
      const input = cpu.zeros([2, 3, 4, 5])
      const output = flatten.forward(input)
      expect([...output.shape]).toEqual([2, 12, 5])
    })
  })

  test('toString', () => {
    const flatten = new Flatten()
    expect(flatten.toString()).toBe('Flatten(start_dim=1, end_dim=-1)')
  })

  test('custom startDim and endDim in toString', () => {
    const flatten = new Flatten(0, 2)
    expect(flatten.toString()).toBe('Flatten(start_dim=0, end_dim=2)')
  })
})
