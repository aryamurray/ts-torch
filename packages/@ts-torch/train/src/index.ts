/**
 * @ts-torch/train - Declarative training API
 *
 * Provides a declarative, configuration-driven approach to training neural networks.
 * Instead of writing imperative training loops, describe WHAT you want:
 *
 * @example
 * ```ts
 * import { Trainer, Adam } from '@ts-torch/train'
 *
 * const trainer = new Trainer(model)
 *
 * await trainer.fit(trainLoader, {
 *   epochs: 10,
 *   optimizer: Adam({ lr: 1e-3 }),
 *   loss: 'crossEntropy',
 *   metrics: { accuracy: true },
 *   validateOn: testLoader,
 *   onEpochEnd: ({ metrics }) => console.log(metrics)
 * })
 * ```
 */

// Trainer
export { Trainer } from './trainer.js'
export type { FitOptions, EvaluateOptions, EpochContext, BatchContext, History, LossType, LossFn, TrainerConfig } from './trainer.js'

// Optimizer factories
export { Adam, SGD, AdamW, RMSprop } from './optimizers.js'
export type { OptimizerConfig, AdamConfig, SGDConfig, AdamWConfig, RMSpropConfig } from './optimizers.js'

// Metrics
export { LossMetric, AccuracyMetric, TopKAccuracyMetric, CustomMetric, createMetrics, computeMetrics, resetMetrics } from './metrics.js'
export type { Metric, MetricsConfig, MetricsResult, MetricFn } from './metrics.js'
