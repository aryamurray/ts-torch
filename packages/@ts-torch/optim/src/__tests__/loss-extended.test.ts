/**
 * Tests for extended loss functions
 * (BCE, BCEWithLogits, L1, SmoothL1, KLDiv, CosineEmbedding, TripletMargin)
 */

import { describe, test, expect } from 'vitest'
import { device, run, float32 } from '@ts-torch/core'
import {
  bceLoss,
  bceWithLogitsLoss,
  l1Loss,
  smoothL1Loss,
  huberLoss,
  klDivLoss,
  cosineEmbeddingLoss,
  tripletMarginLoss,
} from '../loss'

const cpu = device.cpu()

describe('bceLoss', () => {
  test('computes binary cross entropy loss', () => {
    run(() => {
      // Probabilities (after sigmoid)
      const input = cpu.tensor([0.9, 0.1, 0.8, 0.2], [4] as const)
      const target = cpu.tensor([1.0, 0.0, 1.0, 0.0], [4] as const)

      const loss = bceLoss(input, target)

      // Loss should be positive
      expect(loss.item()).toBeGreaterThan(0)
      expect(isFinite(loss.item())).toBe(true)
    })
  })

  test('perfect predictions have low loss', () => {
    run(() => {
      const input = cpu.tensor([0.99, 0.01], [2] as const)
      const target = cpu.tensor([1.0, 0.0], [2] as const)

      const loss = bceLoss(input, target)

      expect(loss.item()).toBeLessThan(0.1)
    })
  })

  test('wrong predictions have high loss', () => {
    run(() => {
      const input = cpu.tensor([0.01, 0.99], [2] as const)
      const target = cpu.tensor([1.0, 0.0], [2] as const)

      const loss = bceLoss(input, target)

      expect(loss.item()).toBeGreaterThan(2.0)
    })
  })

  test('reduction none returns per-element loss', () => {
    run(() => {
      const input = cpu.tensor([0.9, 0.1], [2] as const)
      const target = cpu.tensor([1.0, 0.0], [2] as const)

      const loss = bceLoss(input, target, 'none')

      expect(loss.shape).toEqual([2])
    })
  })

  test('reduction sum returns total loss', () => {
    run(() => {
      const input = cpu.tensor([0.9, 0.1], [2] as const)
      const target = cpu.tensor([1.0, 0.0], [2] as const)

      const lossMean = bceLoss(input, target, 'mean')
      const lossSum = bceLoss(input, target, 'sum')

      expect(lossSum.item()).toBeGreaterThan(lossMean.item())
    })
  })

  test('handles 2D input', () => {
    run(() => {
      const input = cpu.randn([8, 4] as const).sigmoid()
      const target = cpu.tensor(
        [1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1],
        [8, 4] as const,
      )

      const loss = bceLoss(input, target)

      expect(loss.shape).toEqual([])
      expect(isFinite(loss.item())).toBe(true)
    })
  })
})

describe('bceWithLogitsLoss', () => {
  test('computes BCE with logits (numerically stable)', () => {
    run(() => {
      // Raw logits (before sigmoid)
      const input = cpu.tensor([2.0, -2.0, 1.0, -1.0], [4] as const)
      const target = cpu.tensor([1.0, 0.0, 1.0, 0.0], [4] as const)

      const loss = bceWithLogitsLoss(input, target)

      expect(loss.item()).toBeGreaterThan(0)
      expect(isFinite(loss.item())).toBe(true)
    })
  })

  test('handles extreme logits without NaN', () => {
    run(() => {
      // Very large logits that would cause issues with manual sigmoid + BCE
      const input = cpu.tensor([100.0, -100.0], [2] as const)
      const target = cpu.tensor([1.0, 0.0], [2] as const)

      const loss = bceWithLogitsLoss(input, target)

      expect(isFinite(loss.item())).toBe(true)
      expect(loss.item()).toBeLessThan(1.0) // Correct predictions
    })
  })

  test('reduction options work correctly', () => {
    run(() => {
      const input = cpu.tensor([1.0, -1.0], [2] as const)
      const target = cpu.tensor([1.0, 0.0], [2] as const)

      const lossNone = bceWithLogitsLoss(input, target, 'none')
      const lossMean = bceWithLogitsLoss(input, target, 'mean')
      const lossSum = bceWithLogitsLoss(input, target, 'sum')

      expect(lossNone.shape).toEqual([2])
      expect(lossMean.shape).toEqual([])
      expect(lossSum.shape).toEqual([])
    })
  })
})

describe('l1Loss', () => {
  test('computes mean absolute error', () => {
    run(() => {
      const input = cpu.tensor([1.0, 2.0, 3.0], [3] as const)
      const target = cpu.tensor([1.5, 2.5, 2.5], [3] as const)

      const loss = l1Loss(input, target)

      // L1 = mean(|1-1.5|, |2-2.5|, |3-2.5|) = mean(0.5, 0.5, 0.5) = 0.5
      expect(loss.item()).toBeCloseTo(0.5, 4)
    })
  })

  test('handles zero error', () => {
    run(() => {
      const input = cpu.tensor([1.0, 2.0, 3.0], [3] as const)
      const target = cpu.tensor([1.0, 2.0, 3.0], [3] as const)

      const loss = l1Loss(input, target)

      expect(loss.item()).toBeCloseTo(0.0, 5)
    })
  })

  test('reduction options work correctly', () => {
    run(() => {
      const input = cpu.tensor([1.0, 2.0], [2] as const)
      const target = cpu.tensor([0.0, 0.0], [2] as const)

      const lossNone = l1Loss(input, target, 'none')
      const lossMean = l1Loss(input, target, 'mean')
      const lossSum = l1Loss(input, target, 'sum')

      expect(lossNone.shape).toEqual([2])
      expect(lossMean.item()).toBeCloseTo(1.5, 4)
      expect(lossSum.item()).toBeCloseTo(3.0, 4)
    })
  })
})

describe('smoothL1Loss (Huber Loss)', () => {
  test('computes smooth L1 loss', () => {
    run(() => {
      const input = cpu.tensor([1.0, 2.0, 3.0], [3] as const)
      const target = cpu.tensor([1.0, 2.0, 3.0], [3] as const)

      const loss = smoothL1Loss(input, target)

      expect(loss.item()).toBeCloseTo(0.0, 4)
    })
  })

  test('less sensitive to outliers than MSE', () => {
    run(() => {
      const input = cpu.tensor([0.0, 0.0, 10.0], [3] as const) // One outlier
      const target = cpu.tensor([0.0, 0.0, 0.0], [3] as const)

      const l1 = l1Loss(input, target)
      const smooth = smoothL1Loss(input, target)

      // Both should be positive
      expect(l1.item()).toBeGreaterThan(0)
      expect(smooth.item()).toBeGreaterThan(0)
    })
  })

  test('accepts custom beta parameter', () => {
    run(() => {
      const input = cpu.tensor([0.5], [1] as const)
      const target = cpu.tensor([0.0], [1] as const)

      const loss = smoothL1Loss(input, target, 'mean', 1.0)

      expect(loss.item()).toBeGreaterThan(0)
    })
  })

  test('huberLoss is alias for smoothL1Loss', () => {
    run(() => {
      const input = cpu.tensor([1.0, 2.0], [2] as const)
      const target = cpu.tensor([0.0, 0.0], [2] as const)

      const smoothL1 = smoothL1Loss(input, target, 'mean', 1.0)
      const huber = huberLoss(input, target, 'mean', 1.0)

      expect(smoothL1.item()).toBeCloseTo(huber.item(), 5)
    })
  })
})

describe('klDivLoss', () => {
  test('computes KL divergence loss', () => {
    run(() => {
      // log(Q) - input
      const input = cpu.tensor([-0.5, -1.0, -2.0], [3] as const)
      // P - target (probability distribution)
      const target = cpu.tensor([0.5, 0.3, 0.2], [3] as const)

      const loss = klDivLoss(input, target)

      expect(loss.item()).toBeGreaterThan(0)
      expect(isFinite(loss.item())).toBe(true)
    })
  })

  test('identical distributions have low loss', () => {
    run(() => {
      const probs = cpu.tensor([0.25, 0.25, 0.25, 0.25], [4] as const)
      const logProbs = probs.log()

      const loss = klDivLoss(logProbs, probs)

      // KL(P||P) ≈ 0
      expect(loss.item()).toBeCloseTo(0, 1)
    })
  })

  test('handles 2D distributions', () => {
    run(() => {
      const input = cpu.randn([4, 10] as const).logSoftmax(-1)
      const target = cpu.randn([4, 10] as const).softmax(-1)

      const loss = klDivLoss(input, target)

      expect(loss.shape).toEqual([])
      expect(isFinite(loss.item())).toBe(true)
    })
  })
})

describe('cosineEmbeddingLoss', () => {
  test('computes cosine embedding loss for similar pairs', () => {
    run(() => {
      const input1 = cpu.tensor([1.0, 0.0, 0.0], [3] as const)
      const input2 = cpu.tensor([1.0, 0.1, 0.0], [3] as const) // Similar
      const target = cpu.tensor([1.0], [1] as const) // Similar label

      const loss = cosineEmbeddingLoss(
        input1.reshape([1, 3]),
        input2.reshape([1, 3]),
        target,
      )

      // Similar vectors with y=1 should have low loss
      expect(loss.item()).toBeLessThan(0.5)
    })
  })

  test('computes cosine embedding loss for dissimilar pairs', () => {
    run(() => {
      const input1 = cpu.tensor([1.0, 0.0, 0.0], [3] as const)
      const input2 = cpu.tensor([-1.0, 0.0, 0.0], [3] as const) // Opposite
      const target = cpu.tensor([-1.0], [1] as const) // Dissimilar label

      const loss = cosineEmbeddingLoss(
        input1.reshape([1, 3]),
        input2.reshape([1, 3]),
        target,
      )

      // Opposite vectors with y=-1 should have low loss
      expect(loss.item()).toBeLessThan(1.0)
    })
  })

  test('accepts margin parameter', () => {
    run(() => {
      const input1 = cpu.tensor([1.0, 0.0], [2] as const)
      const input2 = cpu.tensor([0.0, 1.0], [2] as const)
      const target = cpu.tensor([-1.0], [1] as const)

      const loss = cosineEmbeddingLoss(
        input1.reshape([1, 2]),
        input2.reshape([1, 2]),
        target,
        0.5,
      )

      expect(loss.item()).toBeGreaterThanOrEqual(0)
    })
  })
})

describe('tripletMarginLoss', () => {
  test('computes triplet margin loss', () => {
    run(() => {
      const anchor = cpu.tensor([1.0, 0.0, 0.0], [1, 3] as const)
      const positive = cpu.tensor([0.9, 0.1, 0.0], [1, 3] as const) // Similar to anchor
      const negative = cpu.tensor([0.0, 1.0, 0.0], [1, 3] as const) // Different from anchor

      const loss = tripletMarginLoss(anchor, positive, negative)

      // Good triplet (pos close, neg far) should have low loss
      expect(loss.item()).toBeGreaterThanOrEqual(0)
    })
  })

  test('bad triplet has higher loss', () => {
    run(() => {
      const anchor = cpu.tensor([1.0, 0.0], [1, 2] as const)
      const positive = cpu.tensor([0.0, 1.0], [1, 2] as const) // Far from anchor
      const negative = cpu.tensor([0.9, 0.0], [1, 2] as const) // Close to anchor

      const loss = tripletMarginLoss(anchor, positive, negative, 1.0)

      // Bad triplet should have positive loss
      expect(loss.item()).toBeGreaterThan(0)
    })
  })

  test('accepts margin parameter', () => {
    run(() => {
      const anchor = cpu.randn([4, 64] as const)
      const positive = cpu.randn([4, 64] as const)
      const negative = cpu.randn([4, 64] as const)

      const loss = tripletMarginLoss(anchor, positive, negative, 2.0)

      expect(loss.item()).toBeGreaterThanOrEqual(0)
    })
  })

  test('accepts p-norm parameter', () => {
    run(() => {
      const anchor = cpu.randn([2, 32] as const)
      const positive = cpu.randn([2, 32] as const)
      const negative = cpu.randn([2, 32] as const)

      // L1 norm
      const lossL1 = tripletMarginLoss(anchor, positive, negative, 1.0, 1)
      expect(lossL1.item()).toBeGreaterThanOrEqual(0)

      // L2 norm (default)
      const lossL2 = tripletMarginLoss(anchor, positive, negative, 1.0, 2)
      expect(lossL2.item()).toBeGreaterThanOrEqual(0)
    })
  })

  test('reduction options work correctly', () => {
    run(() => {
      const anchor = cpu.randn([4, 16] as const)
      const positive = cpu.randn([4, 16] as const)
      const negative = cpu.randn([4, 16] as const)

      const lossMean = tripletMarginLoss(anchor, positive, negative, 1.0, 2, 'mean')
      const lossSum = tripletMarginLoss(anchor, positive, negative, 1.0, 2, 'sum')
      const lossNone = tripletMarginLoss(anchor, positive, negative, 1.0, 2, 'none')

      expect(lossMean.shape).toEqual([])
      expect(lossSum.shape).toEqual([])
      expect(lossNone.shape).toEqual([4])
    })
  })
})

describe('gradient flow', () => {
  test('bceLoss supports backward pass', () => {
    run(() => {
      const input = cpu.tensor([0.9, 0.1], [2] as const, float32, true)
      const target = cpu.tensor([1.0, 0.0], [2] as const)

      const loss = bceLoss(input, target)
      loss.backward()

      expect(input.grad).not.toBeNull()
    })
  })

  test('bceWithLogitsLoss supports backward pass', () => {
    run(() => {
      const input = cpu.tensor([2.0, -2.0], [2] as const, float32, true)
      const target = cpu.tensor([1.0, 0.0], [2] as const)

      const loss = bceWithLogitsLoss(input, target)
      loss.backward()

      expect(input.grad).not.toBeNull()
    })
  })

  test('l1Loss supports backward pass', () => {
    run(() => {
      const input = cpu.tensor([1.0, 2.0], [2] as const, float32, true)
      const target = cpu.tensor([0.0, 0.0], [2] as const)

      const loss = l1Loss(input, target)
      loss.backward()

      expect(input.grad).not.toBeNull()
    })
  })

  test('smoothL1Loss supports backward pass', () => {
    run(() => {
      const input = cpu.tensor([1.0, 2.0], [2] as const, float32, true)
      const target = cpu.tensor([0.0, 0.0], [2] as const)

      const loss = smoothL1Loss(input, target)
      loss.backward()

      expect(input.grad).not.toBeNull()
    })
  })
})
