/**
 * Tests for dataset transforms
 */

import { describe, it, expect } from 'vitest'
import { device } from '@ts-torch/core'
import { Normalize, Pad, RandomHorizontalFlip, Resize, ToTensor } from '../transforms.js'

const cpu = device.cpu()

describe('Transforms', () => {
  it('normalizes per-channel tensors', () => {
    const tensor = cpu.tensor([1, 2, 3, 4, 5, 6], [3, 1, 2] as const)
    const normalize = new Normalize([1, 2, 3], [1, 1, 1])

    const result = normalize.apply(tensor)
    expect(result).toBeCloseTo([0, 1, 1, 2, 2, 3])
  })

  it('pads tensors with constant values', () => {
    const tensor = cpu.tensor([1, 2, 3, 4], [1, 2, 2] as const)
    const pad = new Pad([1, 1, 1, 1], -1)

    const result = pad.apply(tensor)
    expect(result.shape).toEqual([1, 4, 4])
    expect(result.toArray()).toHaveLength(16)
  })

  it('flips tensors horizontally when p=1', () => {
    const tensor = cpu.tensor([1, 2, 3, 4], [1, 2, 2] as const)
    const flip = new RandomHorizontalFlip(1)

    const result = flip.apply(tensor)
    expect(result).toBeCloseTo([2, 1, 4, 3])
  })

  it('resizes tensors with nearest neighbor', () => {
    const tensor = cpu.tensor([1, 2, 3, 4], [1, 2, 2] as const)
    const resize = new Resize([1, 4])

    const result = resize.apply(tensor)
    expect(result.shape).toEqual([1, 1, 4])
  })

  it('converts nested arrays to tensor', () => {
    const toTensor = new ToTensor()
    const tensor = toTensor.apply([[1, 2], [3, 4]])

    expect(tensor.shape).toEqual([2, 2])
    expect(tensor).toBeCloseTo([1, 2, 3, 4])
  })
})
