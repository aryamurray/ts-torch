/**
 * @ts-torch/train - Declarative training API
 *
 * Provides a declarative, configuration-driven approach to training neural networks.
 * Instead of writing imperative training loops, describe WHAT you want:
 *
 * @example
 * ```ts
 * import { Trainer, Adam, loss, logger } from '@ts-torch/train'
 *
 * const trainer = new Trainer({
 *   model,
 *   data: trainLoader,
 *   epochs: 10,
 *   optimizer: Adam({ lr: 1e-3 }),
 *   loss: loss.crossEntropy(),
 *   metrics: ['loss', 'accuracy'],
 *   validation: testLoader,
 *   callbacks: [logger.console()],
 * })
 *
 * const history = await trainer.fit()
 * ```
 */

// Trainer
export { Trainer } from './trainer.js'
export type { TrainerOptions, EvaluateOptions, Batch, TensorTree } from './trainer.js'
export { mapTensorTree } from './trainer.js'

// Loss
export { loss, resolveLoss } from './loss.js'
export type { LossConfig } from './loss.js'

// Callbacks
export { consoleLogger, earlyStop, checkpoint, logger } from './callbacks.js'
export type { Callback, CallbackResult, EpochContext, BatchContext, History, EpochRecord } from './callbacks.js'

// Dashboard
export { createDashboardCallback } from './dashboard-callback.js'

// Optimizer factories
export { Adam, SGD, AdamW, RMSprop } from './optimizers.js'
export type { OptimizerConfig, AdamConfig, SGDConfig, AdamWConfig, RMSpropConfig } from './optimizers.js'

// Metrics
export {
  LossMetric,
  AccuracyMetric,
  TopKAccuracyMetric,
  CustomMetric,
  createMetrics,
  computeMetrics,
  resetMetrics,
  metric,
} from './metrics.js'
export type { Metric, MetricFn, MetricSpec, BuiltinMetricName, NamedMetric } from './metrics.js'

// Learning rate schedulers
export {
  StepLR,
  MultiStepLR,
  ExponentialLR,
  CosineAnnealingLR,
  CosineAnnealingWarmRestarts,
  ReduceLROnPlateau,
  LinearWarmup,
} from './schedulers.js'
export type {
  SchedulerConfig,
  StepLRConfig,
  MultiStepLRConfig,
  ExponentialLRConfig,
  CosineAnnealingLRConfig,
  CosineAnnealingWarmRestartsConfig,
  ReduceLROnPlateauConfig,
  LinearWarmupConfig,
} from './schedulers.js'
