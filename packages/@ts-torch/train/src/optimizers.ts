/**
 * Declarative Optimizer Factories
 *
 * Factory functions that create optimizer configurations for use with Trainer.fit().
 *
 * @example
 * ```ts
 * await trainer.fit(data, {
 *   optimizer: Adam({ lr: 1e-3, weightDecay: 1e-4 }),
 *   // ...
 * })
 * ```
 */

import { Adam as AdamOptimizer, SGD as SGDOptimizer, AdamW as AdamWOptimizer, RMSprop as RMSpropOptimizer } from '@ts-torch/optim'
import type { Optimizer } from '@ts-torch/optim'
import type { Module } from '@ts-torch/nn'

/**
 * Optimizer configuration - used by Trainer to create optimizer
 */
export interface OptimizerConfig {
  /** Factory function that creates the optimizer given model parameters */
  create: (model: Module<any, any, any>) => Optimizer
  /** Display name */
  name: string
}

/**
 * Adam optimizer configuration
 */
export interface AdamConfig {
  /** Learning rate */
  lr: number
  /** Beta coefficients for computing running averages (default: [0.9, 0.999]) */
  betas?: [number, number]
  /** Term added to denominator for numerical stability (default: 1e-8) */
  eps?: number
  /** Weight decay (L2 penalty) (default: 0) */
  weightDecay?: number
  /** Whether to use AMSGrad variant (default: false) */
  amsgrad?: boolean
  /** Index signature for optimizer compatibility */
  [key: string]: unknown
}

/**
 * Create an Adam optimizer configuration
 *
 * @param config - Adam configuration
 * @returns OptimizerConfig for use with Trainer
 *
 * @example
 * ```ts
 * await trainer.fit(data, {
 *   optimizer: Adam({ lr: 1e-3, weightDecay: 1e-4 }),
 *   // ...
 * })
 * ```
 */
export function Adam(config: AdamConfig): OptimizerConfig {
  return {
    name: `Adam(lr=${config.lr})`,
    create: (model) => {
      const params = model.parameters().map((p) => p.data) as any
      return new AdamOptimizer(params, config)
    },
  }
}

/**
 * SGD optimizer configuration
 */
export interface SGDConfig {
  /** Learning rate */
  lr: number
  /** Momentum factor (default: 0) */
  momentum?: number
  /** Weight decay (L2 penalty) (default: 0) */
  weightDecay?: number
  /** Dampening for momentum (default: 0) */
  dampening?: number
  /** Whether to use Nesterov momentum (default: false) */
  nesterov?: boolean
  /** Index signature for optimizer compatibility */
  [key: string]: unknown
}

/**
 * Create an SGD optimizer configuration
 *
 * @param config - SGD configuration
 * @returns OptimizerConfig for use with Trainer
 *
 * @example
 * ```ts
 * await trainer.fit(data, {
 *   optimizer: SGD({ lr: 0.01, momentum: 0.9 }),
 *   // ...
 * })
 * ```
 */
export function SGD(config: SGDConfig): OptimizerConfig {
  return {
    name: `SGD(lr=${config.lr}, momentum=${config.momentum ?? 0})`,
    create: (model) => {
      const params = model.parameters().map((p) => p.data) as any
      return new SGDOptimizer(params, config)
    },
  }
}

/**
 * AdamW optimizer configuration
 */
export interface AdamWConfig {
  /** Learning rate */
  lr: number
  /** Beta coefficients for computing running averages (default: [0.9, 0.999]) */
  betas?: [number, number]
  /** Term added to denominator for numerical stability (default: 1e-8) */
  eps?: number
  /** Weight decay coefficient (default: 0.01) */
  weightDecay?: number
  /** Whether to use AMSGrad variant (default: false) */
  amsgrad?: boolean
  /** Index signature for optimizer compatibility */
  [key: string]: unknown
}

/**
 * Create an AdamW optimizer configuration
 *
 * AdamW uses decoupled weight decay regularization, which often performs
 * better than Adam with L2 regularization.
 *
 * @param config - AdamW configuration
 * @returns OptimizerConfig for use with Trainer
 *
 * @example
 * ```ts
 * await trainer.fit(data, {
 *   optimizer: AdamW({ lr: 1e-3, weightDecay: 0.01 }),
 *   // ...
 * })
 * ```
 */
export function AdamW(config: AdamWConfig): OptimizerConfig {
  return {
    name: `AdamW(lr=${config.lr}, weight_decay=${config.weightDecay ?? 0.01})`,
    create: (model) => {
      const params = model.parameters().map((p) => p.data) as any
      return new AdamWOptimizer(params, config)
    },
  }
}

/**
 * RMSprop optimizer configuration
 */
export interface RMSpropConfig {
  /** Learning rate */
  lr: number
  /** Smoothing constant (default: 0.99) */
  alpha?: number
  /** Term added to denominator for numerical stability (default: 1e-8) */
  eps?: number
  /** Weight decay (L2 penalty) (default: 0) */
  weightDecay?: number
  /** Momentum factor (default: 0) */
  momentum?: number
  /** If true, center the gradient by subtracting the mean (default: false) */
  centered?: boolean
  /** Index signature for optimizer compatibility */
  [key: string]: unknown
}

/**
 * Create an RMSprop optimizer configuration
 *
 * @param config - RMSprop configuration
 * @returns OptimizerConfig for use with Trainer
 *
 * @example
 * ```ts
 * await trainer.fit(data, {
 *   optimizer: RMSprop({ lr: 0.01 }),
 *   // ...
 * })
 * ```
 */
export function RMSprop(config: RMSpropConfig): OptimizerConfig {
  return {
    name: `RMSprop(lr=${config.lr})`,
    create: (model) => {
      const params = model.parameters().map((p) => p.data) as any
      return new RMSpropOptimizer(params, config)
    },
  }
}
