/**
 * Declarative LR Scheduler Factories
 *
 * Factory functions that create scheduler configurations for use with Trainer.fit().
 *
 * @example
 * ```ts
 * await trainer.fit(data, {
 *   optimizer: Adam({ lr: 1e-3 }),
 *   scheduler: StepLR({ stepSize: 30, gamma: 0.1 }),
 *   // ...
 * })
 * ```
 */

import {
  StepLR as StepLRScheduler,
  MultiStepLR as MultiStepLRScheduler,
  ExponentialLR as ExponentialLRScheduler,
  CosineAnnealingLR as CosineAnnealingLRScheduler,
  CosineAnnealingWarmRestarts as CosineAnnealingWarmRestartsScheduler,
  ReduceLROnPlateau as ReduceLROnPlateauScheduler,
  LinearWarmup as LinearWarmupScheduler,
  type LRScheduler,
} from '@ts-torch/optim'
import type { Optimizer } from '@ts-torch/optim'

/**
 * Scheduler configuration - used by Trainer to create scheduler
 */
export interface SchedulerConfig {
  /** Factory function that creates the scheduler given optimizer */
  create: (optimizer: Optimizer) => LRScheduler
  /** Display name */
  name: string
  /** Whether this scheduler needs metrics (like ReduceLROnPlateau) */
  needsMetrics?: boolean
  /** When to step: 'epoch' (default) or 'batch' */
  stepOn?: 'epoch' | 'batch'
}

/**
 * StepLR scheduler configuration
 */
export interface StepLRConfig {
  /** Period of learning rate decay */
  stepSize: number
  /** Multiplicative factor of learning rate decay (default: 0.1) */
  gamma?: number
}

/**
 * Create a StepLR scheduler configuration
 *
 * Decays the learning rate by gamma every stepSize epochs.
 *
 * @param config - StepLR configuration
 * @returns SchedulerConfig for use with Trainer
 *
 * @example
 * ```ts
 * await trainer.fit(data, {
 *   optimizer: Adam({ lr: 0.1 }),
 *   scheduler: StepLR({ stepSize: 30, gamma: 0.1 }),
 *   // LR: 0.1 -> 0.01 (epoch 30) -> 0.001 (epoch 60) -> ...
 * })
 * ```
 */
export function StepLR(config: StepLRConfig): SchedulerConfig {
  return {
    name: `StepLR(stepSize=${config.stepSize}, gamma=${config.gamma ?? 0.1})`,
    create: (optimizer) => new StepLRScheduler(optimizer, config.stepSize, config.gamma ?? 0.1),
  }
}

/**
 * MultiStepLR scheduler configuration
 */
export interface MultiStepLRConfig {
  /** List of epoch indices at which to decay the learning rate */
  milestones: number[]
  /** Multiplicative factor of learning rate decay (default: 0.1) */
  gamma?: number
}

/**
 * Create a MultiStepLR scheduler configuration
 *
 * Decays the learning rate by gamma once the number of epochs reaches one of the milestones.
 *
 * @param config - MultiStepLR configuration
 * @returns SchedulerConfig for use with Trainer
 *
 * @example
 * ```ts
 * await trainer.fit(data, {
 *   optimizer: Adam({ lr: 0.1 }),
 *   scheduler: MultiStepLR({ milestones: [30, 80], gamma: 0.1 }),
 *   // LR: 0.1 -> 0.01 (epoch 30) -> 0.001 (epoch 80)
 * })
 * ```
 */
export function MultiStepLR(config: MultiStepLRConfig): SchedulerConfig {
  return {
    name: `MultiStepLR(milestones=[${config.milestones}], gamma=${config.gamma ?? 0.1})`,
    create: (optimizer) => new MultiStepLRScheduler(optimizer, config.milestones, config.gamma ?? 0.1),
  }
}

/**
 * ExponentialLR scheduler configuration
 */
export interface ExponentialLRConfig {
  /** Multiplicative factor of learning rate decay */
  gamma: number
}

/**
 * Create an ExponentialLR scheduler configuration
 *
 * Decays the learning rate by gamma every epoch.
 *
 * @param config - ExponentialLR configuration
 * @returns SchedulerConfig for use with Trainer
 *
 * @example
 * ```ts
 * await trainer.fit(data, {
 *   optimizer: Adam({ lr: 0.1 }),
 *   scheduler: ExponentialLR({ gamma: 0.95 }),
 *   // LR = 0.1 * 0.95^epoch
 * })
 * ```
 */
export function ExponentialLR(config: ExponentialLRConfig): SchedulerConfig {
  return {
    name: `ExponentialLR(gamma=${config.gamma})`,
    create: (optimizer) => new ExponentialLRScheduler(optimizer, config.gamma),
  }
}

/**
 * CosineAnnealingLR scheduler configuration
 */
export interface CosineAnnealingLRConfig {
  /** Maximum number of iterations (epochs) */
  tMax: number
  /** Minimum learning rate (default: 0) */
  etaMin?: number
}

/**
 * Create a CosineAnnealingLR scheduler configuration
 *
 * Sets the learning rate using a cosine annealing schedule.
 *
 * @param config - CosineAnnealingLR configuration
 * @returns SchedulerConfig for use with Trainer
 *
 * @example
 * ```ts
 * await trainer.fit(data, {
 *   epochs: 50,
 *   optimizer: Adam({ lr: 0.1 }),
 *   scheduler: CosineAnnealingLR({ tMax: 50, etaMin: 0.001 }),
 *   // LR follows cosine curve from 0.1 to 0.001
 * })
 * ```
 */
export function CosineAnnealingLR(config: CosineAnnealingLRConfig): SchedulerConfig {
  return {
    name: `CosineAnnealingLR(T_max=${config.tMax}, eta_min=${config.etaMin ?? 0})`,
    create: (optimizer) => new CosineAnnealingLRScheduler(optimizer, config.tMax, config.etaMin ?? 0),
  }
}

/**
 * CosineAnnealingWarmRestarts scheduler configuration
 */
export interface CosineAnnealingWarmRestartsConfig {
  /** Number of iterations for the first restart */
  t0: number
  /** A factor increases t_i after a restart (default: 1) */
  tMult?: number
  /** Minimum learning rate (default: 0) */
  etaMin?: number
}

/**
 * Create a CosineAnnealingWarmRestarts scheduler configuration
 *
 * Sets the learning rate using a cosine annealing schedule with periodic restarts.
 *
 * @param config - CosineAnnealingWarmRestarts configuration
 * @returns SchedulerConfig for use with Trainer
 *
 * @example
 * ```ts
 * await trainer.fit(data, {
 *   optimizer: Adam({ lr: 0.1 }),
 *   scheduler: CosineAnnealingWarmRestarts({ t0: 10, tMult: 2 }),
 *   // LR restarts every T_0 * T_mult epochs
 * })
 * ```
 */
export function CosineAnnealingWarmRestarts(config: CosineAnnealingWarmRestartsConfig): SchedulerConfig {
  return {
    name: `CosineAnnealingWarmRestarts(T_0=${config.t0}, T_mult=${config.tMult ?? 1})`,
    create: (optimizer) =>
      new CosineAnnealingWarmRestartsScheduler(optimizer, config.t0, config.tMult ?? 1, config.etaMin ?? 0),
  }
}

/**
 * ReduceLROnPlateau scheduler configuration
 */
export interface ReduceLROnPlateauConfig {
  /** Mode: 'min' or 'max' (default: 'min') */
  mode?: 'min' | 'max'
  /** Factor by which to reduce learning rate (default: 0.1) */
  factor?: number
  /** Number of epochs with no improvement after which LR will be reduced (default: 10) */
  patience?: number
  /** Threshold for measuring the new optimum (default: 1e-4) */
  threshold?: number
  /** One of 'rel', 'abs' (default: 'rel') */
  thresholdMode?: 'rel' | 'abs'
  /** Number of epochs to wait before resuming normal operation (default: 0) */
  cooldown?: number
  /** Minimum learning rate (default: 0) */
  minLr?: number
}

/**
 * Create a ReduceLROnPlateau scheduler configuration
 *
 * Reduces learning rate when a metric has stopped improving.
 *
 * @param config - ReduceLROnPlateau configuration
 * @returns SchedulerConfig for use with Trainer
 *
 * @example
 * ```ts
 * await trainer.fit(data, {
 *   optimizer: Adam({ lr: 0.1 }),
 *   scheduler: ReduceLROnPlateau({ mode: 'min', factor: 0.1, patience: 10 }),
 *   // LR reduced by factor when validation loss doesn't improve for patience epochs
 * })
 * ```
 */
export function ReduceLROnPlateau(config: ReduceLROnPlateauConfig = {}): SchedulerConfig {
  return {
    name: `ReduceLROnPlateau(mode=${config.mode ?? 'min'}, factor=${config.factor ?? 0.1}, patience=${config.patience ?? 10})`,
    needsMetrics: true,
    create: (optimizer) =>
      new ReduceLROnPlateauScheduler(optimizer, config.mode ?? 'min', {
        factor: config.factor ?? 0.1,
        patience: config.patience ?? 10,
        threshold: config.threshold ?? 1e-4,
        thresholdMode: config.thresholdMode ?? 'rel',
        cooldown: config.cooldown ?? 0,
        minLr: config.minLr ?? 0,
      }),
  }
}

/**
 * LinearWarmup scheduler configuration
 */
export interface LinearWarmupConfig {
  /** Number of warmup steps (or epochs if stepOn is 'epoch') */
  warmupSteps: number
  /** When to step: 'batch' or 'epoch' (default: 'batch') */
  stepOn?: 'batch' | 'epoch'
}

/**
 * Create a LinearWarmup scheduler configuration
 *
 * Linearly increases the learning rate from 0 to the base learning rate.
 *
 * @param config - LinearWarmup configuration
 * @returns SchedulerConfig for use with Trainer
 *
 * @example
 * ```ts
 * await trainer.fit(data, {
 *   optimizer: Adam({ lr: 0.001 }),
 *   scheduler: LinearWarmup({ warmupSteps: 1000, stepOn: 'batch' }),
 *   // LR linearly increases from 0 to 0.001 over first 1000 batches
 * })
 * ```
 */
export function LinearWarmup(config: LinearWarmupConfig): SchedulerConfig {
  return {
    name: `LinearWarmup(warmup_steps=${config.warmupSteps})`,
    stepOn: config.stepOn ?? 'batch',
    create: (optimizer) => new LinearWarmupScheduler(optimizer, config.warmupSteps),
  }
}
