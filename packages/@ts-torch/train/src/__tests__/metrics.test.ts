/**
 * Tests for Training Metrics
 */

import { describe, test, expect, beforeEach } from 'vitest'
import {
  LossMetric,
  AccuracyMetric,
  TopKAccuracyMetric,
  CustomMetric,
  createMetrics,
  computeMetrics,
  resetMetrics,
  type Metric,
  type MetricFn,
} from '../metrics'

/**
 * Mock tensor for testing
 */
class MockTensor {
  private values: number[]
  shape: readonly number[]

  constructor(values: number[], shape: readonly number[]) {
    this.values = values
    this.shape = shape
  }

  item(): number {
    return this.values[0]
  }

  toArray(): number[] {
    return this.values
  }

  argmax(_dim?: number): MockTensor {
    // For a 2D tensor [batch, classes], return argmax along dim 1
    const batchSize = this.shape[0]
    const numClasses = this.shape[1] ?? 1
    const results: number[] = []
    for (let i = 0; i < batchSize; i++) {
      let maxIdx = 0
      let maxVal = this.values[i * numClasses]
      for (let j = 1; j < numClasses; j++) {
        if (this.values[i * numClasses + j] > maxVal) {
          maxVal = this.values[i * numClasses + j]
          maxIdx = j
        }
      }
      results.push(maxIdx)
    }
    return new MockTensor(results, [batchSize])
  }

  eq(other: MockTensor): MockTensor {
    const results = this.values.map((v, i) => (v === other.values[i] ? 1 : 0))
    return new MockTensor(results, this.shape)
  }

  sum(): MockTensor {
    const total = this.values.reduce((a, b) => a + b, 0)
    return new MockTensor([total], [1])
  }
}

describe('LossMetric', () => {
  let metric: LossMetric

  beforeEach(() => {
    metric = new LossMetric()
  })

  test('has correct name', () => {
    expect(metric.name).toBe('loss')
  })

  test('starts with zero compute value', () => {
    expect(metric.compute()).toBe(0)
  })

  test('accumulates loss values', () => {
    const predictions = new MockTensor([0.1, 0.9], [1, 2])
    const targets = new MockTensor([1], [1])
    const loss1 = new MockTensor([0.5], [1])
    const loss2 = new MockTensor([0.3], [1])

    metric.update(predictions as any, targets as any, loss1 as any)
    metric.update(predictions as any, targets as any, loss2 as any)

    expect(metric.compute()).toBe(0.4) // (0.5 + 0.3) / 2
  })

  test('ignores updates without loss tensor', () => {
    const predictions = new MockTensor([0.1, 0.9], [1, 2])
    const targets = new MockTensor([1], [1])

    metric.update(predictions as any, targets as any)

    expect(metric.compute()).toBe(0)
  })

  test('reset clears accumulated values', () => {
    const predictions = new MockTensor([0.1, 0.9], [1, 2])
    const targets = new MockTensor([1], [1])
    const loss = new MockTensor([0.5], [1])

    metric.update(predictions as any, targets as any, loss as any)
    expect(metric.compute()).toBe(0.5)

    metric.reset()
    expect(metric.compute()).toBe(0)
  })
})

describe('AccuracyMetric', () => {
  let metric: AccuracyMetric

  beforeEach(() => {
    metric = new AccuracyMetric()
  })

  test('has correct name', () => {
    expect(metric.name).toBe('accuracy')
  })

  test('starts with zero accuracy', () => {
    expect(metric.compute()).toBe(0)
  })

  test('computes accuracy from argmax predictions', () => {
    // Predictions: batch of 4, 3 classes each
    // Sample 0: class 2 (highest at index 2)
    // Sample 1: class 0 (highest at index 0)
    // Sample 2: class 1 (highest at index 1)
    // Sample 3: class 2 (highest at index 2)
    const predictions = new MockTensor(
      [0.1, 0.2, 0.7, 0.8, 0.1, 0.1, 0.1, 0.9, 0.0, 0.2, 0.1, 0.7],
      [4, 3],
    )
    // Targets: [2, 0, 0, 2] - 3 correct (0, 1, 3), 1 wrong (2)
    const targets = new MockTensor([2, 0, 0, 2], [4])

    metric.update(predictions as any, targets as any)

    expect(metric.compute()).toBe(75) // 3/4 = 75%
  })

  test('accumulates across multiple batches', () => {
    // First batch: 2/2 correct
    const pred1 = new MockTensor([0.1, 0.9, 0.9, 0.1], [2, 2])
    const targets1 = new MockTensor([1, 0], [2])

    // Second batch: 1/2 correct
    const pred2 = new MockTensor([0.9, 0.1, 0.9, 0.1], [2, 2])
    const targets2 = new MockTensor([0, 1], [2])

    metric.update(pred1 as any, targets1 as any)
    metric.update(pred2 as any, targets2 as any)

    expect(metric.compute()).toBe(75) // 3/4 = 75%
  })

  test('reset clears accumulated values', () => {
    const predictions = new MockTensor([0.1, 0.9], [1, 2])
    const targets = new MockTensor([1], [1])

    metric.update(predictions as any, targets as any)
    metric.reset()

    expect(metric.compute()).toBe(0)
  })
})

describe('TopKAccuracyMetric', () => {
  test('has correct name with k value', () => {
    const metric = new TopKAccuracyMetric(5)
    expect(metric.name).toBe('top5_accuracy')
  })

  test('starts with zero accuracy', () => {
    const metric = new TopKAccuracyMetric(3)
    expect(metric.compute()).toBe(0)
  })

  test('reset clears accumulated values', () => {
    const metric = new TopKAccuracyMetric(3)
    // Just verify reset doesn't throw
    metric.reset()
    expect(metric.compute()).toBe(0)
  })
})

describe('CustomMetric', () => {
  test('has custom name', () => {
    const fn: MetricFn = () => 0
    const metric = new CustomMetric('myMetric', fn)
    expect(metric.name).toBe('myMetric')
  })

  test('executes custom function', () => {
    const fn: MetricFn = (pred, _target) => {
      // Simple mock: return sum of first prediction value
      return (pred as any).toArray()[0]
    }
    const metric = new CustomMetric('custom', fn)

    const predictions = new MockTensor([0.5, 0.5], [1, 2])
    const targets = new MockTensor([1], [1])

    metric.update(predictions as any, targets as any)

    expect(metric.compute()).toBe(0.5)
  })

  test('averages multiple updates', () => {
    const fn: MetricFn = (pred) => (pred as any).toArray()[0]
    const metric = new CustomMetric('custom', fn)

    metric.update(new MockTensor([0.2], [1]) as any, new MockTensor([0], [1]) as any)
    metric.update(new MockTensor([0.8], [1]) as any, new MockTensor([0], [1]) as any)

    expect(metric.compute()).toBe(0.5) // (0.2 + 0.8) / 2
  })

  test('handles errors gracefully', () => {
    const fn: MetricFn = () => {
      throw new Error('Test error')
    }
    const metric = new CustomMetric('errorMetric', fn)

    const predictions = new MockTensor([0.5], [1])
    const targets = new MockTensor([1], [1])

    // Should not throw, just skip the update
    expect(() => metric.update(predictions as any, targets as any)).not.toThrow()
    expect(metric.compute()).toBe(0)
  })

  test('skips non-finite values', () => {
    const fn: MetricFn = () => NaN
    const metric = new CustomMetric('nanMetric', fn)

    metric.update(new MockTensor([0.5], [1]) as any, new MockTensor([1], [1]) as any)

    expect(metric.compute()).toBe(0)
  })

  test('reset clears accumulated values', () => {
    const fn: MetricFn = () => 1
    const metric = new CustomMetric('custom', fn)

    metric.update(new MockTensor([0.5], [1]) as any, new MockTensor([1], [1]) as any)
    expect(metric.compute()).toBe(1)

    metric.reset()
    expect(metric.compute()).toBe(0)
  })
})

describe('createMetrics', () => {
  test('creates empty array for empty config', () => {
    const metrics = createMetrics({})
    expect(metrics).toHaveLength(0)
  })

  test('creates LossMetric when loss: true', () => {
    const metrics = createMetrics({ loss: true })
    expect(metrics).toHaveLength(1)
    expect(metrics[0].name).toBe('loss')
  })

  test('creates AccuracyMetric when accuracy: true', () => {
    const metrics = createMetrics({ accuracy: true })
    expect(metrics).toHaveLength(1)
    expect(metrics[0].name).toBe('accuracy')
  })

  test('creates TopKAccuracyMetric for each k in topK array', () => {
    const metrics = createMetrics({ topK: [1, 5, 10] })
    expect(metrics).toHaveLength(3)
    expect(metrics[0].name).toBe('top1_accuracy')
    expect(metrics[1].name).toBe('top5_accuracy')
    expect(metrics[2].name).toBe('top10_accuracy')
  })

  test('creates CustomMetric for function values', () => {
    const customFn: MetricFn = () => 42
    const metrics = createMetrics({ myCustom: customFn })
    expect(metrics).toHaveLength(1)
    expect(metrics[0].name).toBe('myCustom')
  })

  test('combines multiple metric types', () => {
    const customFn: MetricFn = () => 0
    const metrics = createMetrics({
      loss: true,
      accuracy: true,
      topK: [5],
      f1: customFn,
    })
    expect(metrics).toHaveLength(4)
    const names = metrics.map((m) => m.name)
    expect(names).toContain('loss')
    expect(names).toContain('accuracy')
    expect(names).toContain('top5_accuracy')
    expect(names).toContain('f1')
  })

  test('ignores false boolean values', () => {
    const metrics = createMetrics({ loss: false, accuracy: false })
    expect(metrics).toHaveLength(0)
  })
})

describe('computeMetrics', () => {
  test('returns empty object for empty metrics array', () => {
    const result = computeMetrics([])
    expect(result).toEqual({})
  })

  test('computes all metrics and returns results', () => {
    const metrics: Metric[] = [new LossMetric(), new AccuracyMetric()]

    // Add some data
    const predictions = new MockTensor([0.1, 0.9], [1, 2])
    const targets = new MockTensor([1], [1])
    const loss = new MockTensor([0.5], [1])

    metrics[0].update(predictions as any, targets as any, loss as any)
    metrics[1].update(predictions as any, targets as any)

    const result = computeMetrics(metrics)

    expect(result.loss).toBe(0.5)
    expect(result.accuracy).toBe(100)
  })
})

describe('resetMetrics', () => {
  test('resets all metrics in array', () => {
    const metrics: Metric[] = [new LossMetric(), new AccuracyMetric()]

    // Add some data
    const predictions = new MockTensor([0.1, 0.9], [1, 2])
    const targets = new MockTensor([1], [1])
    const loss = new MockTensor([0.5], [1])

    metrics[0].update(predictions as any, targets as any, loss as any)
    metrics[1].update(predictions as any, targets as any)

    resetMetrics(metrics)

    expect(metrics[0].compute()).toBe(0)
    expect(metrics[1].compute()).toBe(0)
  })

  test('handles empty array', () => {
    expect(() => resetMetrics([])).not.toThrow()
  })
})
