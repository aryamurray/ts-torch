/**
 * Tests for loss functions using real tensors
 */

import { describe, test, expect } from 'vitest'
import { torch } from '@ts-torch/core'
import {
  mseLoss,
  crossEntropyLoss,
  nllLoss,
} from '../loss'

describe('Loss Functions', () => {
  describe('mseLoss', () => {
    test('computes mean squared error with default reduction', () => {
      torch.run(() => {
        const input = torch.tensor([2.5, 0.0, 2.0, 8.0], [4] as const)
        const target = torch.tensor([3.0, -0.5, 2.0, 7.0], [4] as const)

        const loss = mseLoss(input, target)

        // MSE = mean([(2.5-3)^2, (0-(-0.5))^2, (2-2)^2, (8-7)^2])
        // MSE = mean([0.25, 0.25, 0, 1]) = 1.5 / 4 = 0.375
        expect(loss.item()).toBeCloseTo(0.375, 4)
      })
    })

    test('handles zero error', () => {
      torch.run(() => {
        const input = torch.tensor([1.0, 2.0, 3.0], [3] as const)
        const target = torch.tensor([1.0, 2.0, 3.0], [3] as const)

        const loss = mseLoss(input, target)

        expect(loss.item()).toBeCloseTo(0.0, 5)
      })
    })

    test('handles single element tensors', () => {
      torch.run(() => {
        const input = torch.tensor([1.5], [1] as const)
        const target = torch.tensor([1.0], [1] as const)

        const loss = mseLoss(input, target)

        expect(loss.item()).toBeCloseTo(0.25, 5)
      })
    })

    test('handles 2D tensors', () => {
      torch.run(() => {
        const input = torch.tensor([1.0, 2.0, 3.0, 4.0], [2, 2] as const)
        const target = torch.tensor([0.0, 0.0, 0.0, 0.0], [2, 2] as const)

        const loss = mseLoss(input, target)

        // MSE = mean([1, 4, 9, 16]) = 30/4 = 7.5
        expect(loss.item()).toBeCloseTo(7.5, 4)
      })
    })
  })

  describe('crossEntropyLoss', () => {
    test('computes cross entropy for multi-class classification', () => {
      torch.run(() => {
        // Logits for 2 samples, 3 classes
        const logits = torch.tensor([2.0, 1.0, 0.1, 0.5, 2.5, 0.2], [2, 3] as const)
        const targets = torch.tensor([0, 1], [2] as const, torch.int64)

        const loss = crossEntropyLoss(logits, targets)

        // Loss should be positive and finite
        expect(loss.item()).toBeGreaterThan(0)
        expect(isFinite(loss.item())).toBe(true)
      })
    })

    test('computes CE for single sample', () => {
      torch.run(() => {
        const logits = torch.tensor([1.0, 2.0, 3.0], [1, 3] as const)
        const targets = torch.tensor([2], [1] as const, torch.int64)

        const loss = crossEntropyLoss(logits, targets)

        // When target is the highest logit, loss should be relatively low
        expect(loss.item()).toBeGreaterThan(0)
        expect(loss.item()).toBeLessThan(2.0)
      })
    })

    test('higher loss for wrong predictions', () => {
      torch.run(() => {
        // Logits strongly predict class 0, but target is class 2
        const logits = torch.tensor([10.0, 0.0, 0.0], [1, 3] as const)
        const targets = torch.tensor([2], [1] as const, torch.int64)

        const loss = crossEntropyLoss(logits, targets)

        // Loss should be high since prediction is wrong
        expect(loss.item()).toBeGreaterThan(5.0)
      })
    })

    test('lower loss for correct predictions', () => {
      torch.run(() => {
        // Logits strongly predict class 0, target is class 0
        const logits = torch.tensor([10.0, 0.0, 0.0], [1, 3] as const)
        const targets = torch.tensor([0], [1] as const, torch.int64)

        const loss = crossEntropyLoss(logits, targets)

        // Loss should be low since prediction is correct
        expect(loss.item()).toBeLessThan(0.1)
      })
    })

    test('handles batch of samples', () => {
      torch.run(() => {
        // 4 samples, 5 classes
        const logits = torch.randn([4, 5] as const)
        const targets = torch.tensor([0, 1, 2, 3], [4] as const, torch.int64)

        const loss = crossEntropyLoss(logits, targets)

        expect(loss.item()).toBeGreaterThan(0)
        expect(isFinite(loss.item())).toBe(true)
      })
    })
  })

  describe('nllLoss', () => {
    test('computes negative log likelihood loss', () => {
      torch.run(() => {
        // Log probabilities (output of log_softmax)
        const logits = torch.tensor([2.0, 1.0, 0.1], [1, 3] as const)
        const logProbs = logits.logSoftmax(1)
        const targets = torch.tensor([0], [1] as const, torch.int64)

        const loss = nllLoss(logProbs, targets)

        expect(loss.item()).toBeGreaterThan(0)
        expect(isFinite(loss.item())).toBe(true)
      })
    })

    test('handles batch of samples', () => {
      torch.run(() => {
        const logits = torch.randn([3, 4] as const)
        const logProbs = logits.logSoftmax(1)
        const targets = torch.tensor([0, 1, 2], [3] as const, torch.int64)

        const loss = nllLoss(logProbs, targets)

        expect(loss.item()).toBeGreaterThan(0)
        expect(isFinite(loss.item())).toBe(true)
      })
    })
  })

  describe('gradient flow', () => {
    test('mseLoss supports backward pass', () => {
      torch.run(() => {
        const input = torch.tensor([1.0, 2.0, 3.0], [3] as const, torch.float32, true)
        const target = torch.tensor([0.0, 0.0, 0.0], [3] as const)

        const loss = mseLoss(input, target)
        loss.backward()

        expect(input.grad).not.toBeNull()
      })
    })

    test('crossEntropyLoss supports backward pass', () => {
      torch.run(() => {
        const logits = torch.tensor([1.0, 2.0, 3.0], [1, 3] as const, torch.float32, true)
        const targets = torch.tensor([0], [1] as const, torch.int64)

        const loss = crossEntropyLoss(logits, targets)
        loss.backward()

        expect(logits.grad).not.toBeNull()
      })
    })
  })
})
