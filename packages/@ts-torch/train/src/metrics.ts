/**
 * Training Metrics
 *
 * Metric accumulators for tracking training progress.
 * All metrics accumulate on GPU and only sync scalars at compute time.
 */

import type { Tensor } from '@ts-torch/core'

/**
 * Base metric interface
 */
export interface Metric {
  /** Metric name */
  readonly name: string

  /** Reset accumulated values */
  reset(): void

  /** Update with new predictions and targets */
  update(predictions: Tensor, targets: Tensor, loss?: Tensor): void

  /** Compute final metric value */
  compute(): number | Record<string, number>
}

/**
 * Loss metric - tracks average loss
 */
export class LossMetric implements Metric {
  readonly name = 'loss'
  private total = 0
  private count = 0

  reset(): void {
    this.total = 0
    this.count = 0
  }

  update(_predictions: Tensor, _targets: Tensor, loss?: Tensor): void {
    if (loss) {
      // Get scalar value from loss tensor
      const lossValue = (loss as any).item?.() ?? (loss as any).toArray?.()[0] ?? 0
      this.total += lossValue
      this.count += 1
    }
  }

  compute(): number {
    return this.count > 0 ? this.total / this.count : 0
  }
}

/**
 * Accuracy metric - tracks classification accuracy
 *
 * @remarks
 * Intermediate tensors (argmax, eq, sum) are created within the scope
 * and will be automatically freed when the scope exits.
 */
export class AccuracyMetric implements Metric {
  readonly name = 'accuracy'
  private correct = 0
  private total = 0

  reset(): void {
    this.correct = 0
    this.total = 0
  }

  update(predictions: Tensor, targets: Tensor): void {
    // Get predicted classes (argmax along last dimension)
    const predicted = (predictions as any).argmax?.(1) ?? predictions

    // Compare predictions with targets
    const eq = (predicted as any).eq?.(targets)
    if (eq) {
      // Sum correct predictions
      const correctBatch = (eq as any).sum?.()
      const correctValue = (correctBatch as any).item?.() ?? (correctBatch as any).toArray?.()[0] ?? 0
      this.correct += correctValue
    }

    // Get batch size
    const batchSize = predictions.shape[0] ?? 1
    this.total += batchSize
  }

  compute(): number {
    return this.total > 0 ? (this.correct / this.total) * 100 : 0
  }
}

/**
 * Top-K Accuracy metric
 */
export class TopKAccuracyMetric implements Metric {
  readonly name: string
  private correct = 0
  private total = 0

  constructor(private k: number) {
    this.name = `top${k}_accuracy`
  }

  reset(): void {
    this.correct = 0
    this.total = 0
  }

  update(predictions: Tensor, targets: Tensor): void {
    // Get top-k predictions
    const topk = (predictions as any).topk?.(this.k, 1)
    if (topk) {
      const topkIndices = topk.indices ?? topk[1]

      // Check if target is in top-k
      // Expand targets to match topk shape for comparison
      const targetsExpanded = (targets as any).unsqueeze?.(1).expand?.(-1, this.k)
      if (targetsExpanded) {
        const correct = (topkIndices as any).eq?.(targetsExpanded)?.any?.(1)
        if (correct) {
          const correctSum = (correct as any).sum?.()
          const correctValue = (correctSum as any).item?.() ?? 0
          this.correct += correctValue
        }
      }
    }

    const batchSize = predictions.shape[0] ?? 1
    this.total += batchSize
  }

  compute(): number {
    return this.total > 0 ? (this.correct / this.total) * 100 : 0
  }
}

/**
 * Custom metric function signature
 *
 * @param predictions - Model predictions tensor
 * @param targets - Ground truth targets tensor
 * @returns Metric value (scalar)
 *
 * @example
 * ```ts
 * const myMetric = (pred, target) => pred.argmax(1).eq(target).mean().item()
 * ```
 */
export type MetricFn = (predictions: Tensor, targets: Tensor) => number

/**
 * Custom metric - wraps a user-provided function
 */
export class CustomMetric implements Metric {
  readonly name: string
  private fn: MetricFn
  private total = 0
  private count = 0

  constructor(name: string, fn: MetricFn) {
    this.name = name
    this.fn = fn
  }

  reset(): void {
    this.total = 0
    this.count = 0
  }

  update(predictions: Tensor, targets: Tensor): void {
    try {
      const value = this.fn(predictions, targets)
      if (!Number.isFinite(value)) {
        console.warn(`[ts-torch] Metric '${this.name}' returned non-finite value: ${value}`)
        return
      }
      this.total += value
      this.count += 1
    } catch (error) {
      console.warn(`[ts-torch] Error computing metric '${this.name}':`, error instanceof Error ? error.message : error)
      // Continue training but skip this update
    }
  }

  compute(): number {
    return this.count > 0 ? this.total / this.count : 0
  }
}

/**
 * Built-in metric names
 */
export type BuiltinMetricName = 'loss' | 'accuracy'

/**
 * A named custom metric
 */
export interface NamedMetric {
  name: string
  fn: MetricFn
}

/**
 * Metric specification - either a built-in name or a custom named metric
 */
export type MetricSpec = BuiltinMetricName | NamedMetric

/**
 * Create a named custom metric (convenience factory)
 *
 * @example
 * ```ts
 * metrics: ['loss', 'accuracy', metric('f1', computeF1)]
 * ```
 */
export function metric(name: string, fn: MetricFn): NamedMetric {
  return { name, fn }
}

/**
 * Create metrics from an array of metric specs.
 * Always includes 'loss' even if not explicitly listed.
 *
 * @param specs - Array of metric specs
 * @returns Array of metric instances
 *
 * @example
 * ```ts
 * const metrics = createMetrics(['loss', 'accuracy'])
 * const metrics = createMetrics(['loss', metric('f1', computeF1)])
 * ```
 */
export function createMetrics(specs: MetricSpec[]): Metric[] {
  const metrics: Metric[] = []
  const seen = new Set<string>()

  for (const spec of specs) {
    if (typeof spec === 'string') {
      if (seen.has(spec)) continue
      seen.add(spec)
      switch (spec) {
        case 'loss':
          metrics.push(new LossMetric())
          break
        case 'accuracy':
          metrics.push(new AccuracyMetric())
          break
        default:
          throw new Error(`Unknown built-in metric: ${spec}`)
      }
    } else {
      if (seen.has(spec.name)) continue
      seen.add(spec.name)
      metrics.push(new CustomMetric(spec.name, spec.fn))
    }
  }

  // Always include loss if not already present
  if (!seen.has('loss')) {
    metrics.unshift(new LossMetric())
  }

  return metrics
}

/**
 * Compute all metrics and return results
 *
 * @param metrics - Array of metrics
 * @returns Object with metric names as keys and values
 */
export function computeMetrics(metrics: Metric[]): Record<string, number> {
  const result: Record<string, number> = {}

  for (const metric of metrics) {
    const value = metric.compute()
    if (typeof value === 'number') {
      result[metric.name] = value
    } else {
      Object.assign(result, value)
    }
  }

  return result
}

/**
 * Reset all metrics
 *
 * @param metrics - Array of metrics to reset
 */
export function resetMetrics(metrics: Metric[]): void {
  for (const metric of metrics) {
    metric.reset()
  }
}
