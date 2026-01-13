/**
 * Declarative Trainer
 *
 * The Trainer class provides a declarative API for training neural networks.
 * Instead of manually writing training loops, you describe WHAT you want:
 *
 * @example
 * ```ts
 * const trainer = new Trainer(model)
 *
 * await trainer.fit(trainLoader, {
 *   epochs: 10,
 *   optimizer: Adam({ lr: 1e-3 }),
 *   loss: 'crossEntropy',
 *   metrics: { loss: true, accuracy: true },
 *   validateOn: testLoader,
 *   onEpochEnd: ({ epoch, metrics }) => console.log(metrics)
 * })
 * ```
 */

import type { Tensor } from '@ts-torch/core'
import { run } from '@ts-torch/core'
import type { Module } from '@ts-torch/nn'
import type { Optimizer } from '@ts-torch/optim'
import { crossEntropyLoss, mseLoss, nllLoss } from '@ts-torch/optim'
import type { OptimizerConfig } from './optimizers.js'
import { createMetrics, computeMetrics, resetMetrics, type Metric, type MetricsConfig, type MetricsResult } from './metrics.js'

/**
 * Loss function type
 */
export type LossType = 'crossEntropy' | 'mse' | 'nll'

/**
 * Custom loss function signature
 */
export type LossFn = (predictions: Tensor, targets: Tensor) => Tensor

/**
 * Epoch context passed to callbacks
 */
export interface EpochContext {
  /** Current epoch number (1-indexed) */
  epoch: number
  /** Metrics for this epoch */
  metrics: MetricsResult
  /** Validation metrics (if validateOn provided) */
  valMetrics?: MetricsResult | undefined
  /** Total epochs */
  totalEpochs: number
}

/**
 * Batch context passed to callbacks
 */
export interface BatchContext {
  /** Current step (batch number, 0-indexed) */
  step: number
  /** Current epoch (1-indexed) */
  epoch: number
  /** Loss for this batch */
  loss: number
  /** Total batches in epoch */
  totalBatches: number
}

/**
 * Training history
 */
export interface History {
  /** Loss per epoch */
  loss: number[]
  /** Validation loss per epoch (if validation data provided) */
  valLoss?: number[]
  /** All metrics per epoch */
  metrics: MetricsResult[]
  /** Validation metrics per epoch */
  valMetrics?: MetricsResult[]
}

/**
 * Trainer configuration - passed to constructor
 */
export interface TrainerConfig {
  /** Default optimizer (can be overridden in fit()) */
  optimizer?: OptimizerConfig
  /** Default loss function (can be overridden in fit()) */
  loss?: LossType | LossFn
}

/**
 * Fit options - declarative training configuration
 */
export interface FitOptions {
  /** Number of training epochs */
  epochs: number

  /** Optimizer configuration (optional if set in constructor) */
  optimizer?: OptimizerConfig

  /** Loss function (optional if set in constructor, default: 'crossEntropy') */
  loss?: LossType | LossFn

  /** Metrics to track */
  metrics?: MetricsConfig

  // ==================== Policies ====================

  /** Enable automatic mixed precision */
  amp?: boolean

  /** Precision mode */
  precision?: 'fp32' | 'fp16' | 'bf16'

  /** Gradient accumulation steps */
  accumulate?: number

  /** Gradient clipping max norm */
  clipGradNorm?: number

  // ==================== Validation ====================

  /** Validation data loader */
  validateOn?: AsyncIterable<any>

  /** Validate every N epochs (default: 1) */
  validateEvery?: number

  // ==================== Callbacks ====================

  /** Called at the end of each epoch */
  onEpochEnd?: (ctx: EpochContext) => void | Promise<void>

  /** Called at the end of each batch */
  onBatchEnd?: (ctx: BatchContext) => void | Promise<void>

  /** Log metrics every N batches (0 = no logging) */
  logEvery?: number

  // ==================== Debug ====================

  /** Debug options */
  debug?: {
    /** Log tensor shapes */
    tensors?: boolean
    /** Track memory usage */
    memory?: boolean
    /** Track timing */
    timing?: boolean
    /** Check gradients for NaN/Inf */
    gradients?: boolean
  }
}

/**
 * Evaluation options
 */
export interface EvaluateOptions {
  /** Metrics to compute */
  metrics?: MetricsConfig
}

/**
 * Declarative Trainer for neural network training
 *
 * @example
 * ```ts
 * // Option 1: Pass optimizer in fit()
 * const trainer = new Trainer(model)
 * await trainer.fit(trainLoader, {
 *   epochs: 5,
 *   optimizer: Adam({ lr: 1e-3 }),
 *   loss: 'crossEntropy',
 * })
 *
 * // Option 2: Pass optimizer in constructor (cleaner for reuse)
 * const trainer = new Trainer(model, {
 *   optimizer: Adam({ lr: 1e-3 }),
 *   loss: 'crossEntropy',
 * })
 * await trainer.fit(trainLoader, { epochs: 5 })
 * ```
 */
export class Trainer<M extends Module<any, any, any>> {
  private model: M
  private optimizer: Optimizer | null = null
  private metrics: Metric[] = []
  private lossFn: (predictions: Tensor, targets: Tensor) => Tensor = crossEntropyLoss as any
  private defaultConfig: TrainerConfig

  constructor(model: M, config?: TrainerConfig) {
    this.model = model
    this.defaultConfig = config ?? {}
    if (config?.loss) {
      this.setupLoss(config.loss)
    }
  }

  /**
   * Train the model
   *
   * @param data - Training data loader
   * @param options - Training configuration
   * @returns Training history
   *
   * @example
   * ```ts
   * const history = await trainer.fit(trainLoader, {
   *   epochs: 10,
   *   optimizer: Adam({ lr: 1e-3 }),
   *   loss: 'crossEntropy',
   *   metrics: { loss: true, accuracy: true }
   * })
   * ```
   */
  async fit(data: AsyncIterable<any>, options: FitOptions): Promise<History> {
    const {
      epochs,
      optimizer: optimizerConfig,
      loss,
      metrics: metricsConfig = { loss: true },
      accumulate = 1,
      clipGradNorm,
      validateOn,
      validateEvery = 1,
      onEpochEnd,
      onBatchEnd,
      logEvery = 0,
      debug,
    } = options

    // Resolve optimizer (fit options override constructor config)
    const resolvedOptimizer = optimizerConfig ?? this.defaultConfig.optimizer
    if (!resolvedOptimizer) {
      throw new Error('Optimizer must be provided either in Trainer constructor or fit() options')
    }

    // Setup loss function (fit options override constructor config)
    const resolvedLoss = loss ?? this.defaultConfig.loss ?? 'crossEntropy'
    this.setupLoss(resolvedLoss)

    // Create optimizer
    this.optimizer = resolvedOptimizer.create(this.model)

    // Create metrics
    this.metrics = createMetrics({ ...metricsConfig, loss: true })

    // Training history
    const history: History = {
      loss: [],
      metrics: [],
    }

    if (validateOn) {
      history.valLoss = []
      history.valMetrics = []
    }

    // Set model to training mode
    this.model.train()

    // Training loop
    for (let epoch = 1; epoch <= epochs; epoch++) {
      resetMetrics(this.metrics)

      let batchIdx = 0
      let epochLoss = 0
      let batchCount = 0

      // Iterate over batches
      for await (const batch of data) {
        const batchLoss = await this.trainStep(batch, accumulate, clipGradNorm, batchIdx, debug)

        epochLoss += batchLoss
        batchCount++

        // Update metrics with batch
        for (const metric of this.metrics) {
          if (metric.name === 'loss') {
            // Loss metric updated in trainStep
          }
        }

        // Batch callback
        if (onBatchEnd) {
          await onBatchEnd({
            step: batchIdx,
            epoch,
            loss: batchLoss,
            totalBatches: -1, // Unknown until iteration complete
          })
        }

        // Logging
        if (logEvery > 0 && batchIdx % logEvery === 0) {
          console.log(`Epoch ${epoch} | Batch ${batchIdx} | Loss: ${batchLoss.toFixed(4)}`)
        }

        batchIdx++
      }

      // Compute epoch metrics
      const epochMetrics = computeMetrics(this.metrics)
      epochMetrics.loss = epochLoss / batchCount

      history.loss.push(epochMetrics.loss)
      history.metrics.push(epochMetrics)

      // Validation
      let valMetrics: MetricsResult | undefined
      if (validateOn && epoch % validateEvery === 0) {
        valMetrics = await this.evaluate(validateOn, { metrics: metricsConfig })
        history.valLoss!.push(valMetrics.loss ?? 0)
        history.valMetrics!.push(valMetrics)
      }

      // Epoch callback
      if (onEpochEnd) {
        await onEpochEnd({
          epoch,
          metrics: epochMetrics,
          valMetrics,
          totalEpochs: epochs,
        })
      }
    }

    return history
  }

  /**
   * Evaluate the model on data
   *
   * @param data - Evaluation data loader
   * @param options - Evaluation options
   * @returns Metrics
   *
   * @example
   * ```ts
   * const metrics = await trainer.evaluate(testLoader)
   * console.log(`Accuracy: ${metrics.accuracy}%`)
   * ```
   */
  async evaluate(data: AsyncIterable<any>, options: EvaluateOptions = {}): Promise<MetricsResult> {
    const { metrics: metricsConfig = { loss: true, accuracy: true } } = options

    // Create metrics
    const evalMetrics = createMetrics(metricsConfig)

    // Set model to evaluation mode
    this.model.eval()

    try {
      for await (const batch of data) {
        run(() => {
          const { predictions, targets, loss } = this.forwardBatch(batch)

          // Update metrics
          for (const metric of evalMetrics) {
            metric.update(predictions, targets, loss)
          }
        })
      }

      return computeMetrics(evalMetrics)
    } finally {
      // Restore training mode
      this.model.train()
    }
  }

  /**
   * Setup loss function from type or custom function
   * @internal
   */
  private setupLoss(loss: LossType | LossFn): void {
    if (typeof loss === 'function') {
      this.lossFn = loss
    } else {
      switch (loss) {
        case 'crossEntropy':
          this.lossFn = crossEntropyLoss as any
          break
        case 'mse':
          this.lossFn = mseLoss as any
          break
        case 'nll':
          this.lossFn = nllLoss as any
          break
        default:
          throw new Error(`Unknown loss type: ${loss}`)
      }
    }
  }

  /**
   * Execute a single training step
   * @internal
   */
  private async trainStep(
    batch: any,
    accumulate: number,
    clipGradNorm: number | undefined,
    batchIdx: number,
    _debug?: FitOptions['debug'],
  ): Promise<number> {
    const shouldStep = (batchIdx + 1) % accumulate === 0

    let lossValue = 0

    run(() => {
      // Zero gradients at start of accumulation window
      if (batchIdx % accumulate === 0) {
        this.optimizer!.zeroGrad()
      }

      // Forward pass
      const { predictions, targets } = this.forwardBatch(batch)

      // Compute loss
      const loss = this.lossFn(predictions, targets)

      // Scale loss for gradient accumulation
      const scaledLoss = accumulate > 1 ? (loss as any).divScalar(accumulate) : loss

      // Backward pass
      ;(scaledLoss as any).backward()

      // Get loss value for reporting
      lossValue = (loss as any).item?.() ?? (loss as any).toArray?.()[0] ?? 0

      // Update metrics
      for (const metric of this.metrics) {
        metric.update(predictions, targets, loss)
      }
    })

    // Optimizer step
    if (shouldStep) {
      // Gradient clipping
      if (clipGradNorm) {
        this.clipGradients(clipGradNorm)
      }

      this.optimizer!.step()
    }

    return lossValue
  }

  /**
   * Process a batch and return predictions and targets
   * @internal
   */
  private forwardBatch(batch: any): { predictions: Tensor; targets: Tensor; loss?: Tensor } {
    let data: Tensor
    let targets: Tensor

    // Handle different batch formats
    if (Array.isArray(batch)) {
      // Array of {data, label} objects
      if (batch[0] && 'data' in batch[0] && 'label' in batch[0]) {
        // Stack batch items - for now assume pre-batched tensors
        data = batch[0].data
        targets = batch[0].label
      } else if (batch.length === 2) {
        // [data, targets] tuple
        data = batch[0]
        targets = batch[1]
      } else {
        throw new Error('Unsupported batch format')
      }
    } else if (batch && 'data' in batch && 'label' in batch) {
      // Single {data, label} object
      data = batch.data
      targets = batch.label
    } else if (batch && 'data' in batch && 'targets' in batch) {
      // {data, targets} object
      data = batch.data
      targets = batch.targets
    } else {
      throw new Error('Unsupported batch format')
    }

    // Forward pass
    const predictions = this.model.forward(data as any) as unknown as Tensor

    // Compute loss if metrics need it
    const loss = this.lossFn(predictions as any, targets as any) as Tensor

    return { predictions, targets, loss }
  }

  /**
   * Clip gradients by norm
   * @internal
   */
  private clipGradients(maxNorm: number): void {
    const params = this.model.parameters()

    // Compute total gradient norm
    let totalNorm = 0
    for (const param of params) {
      const grad = (param.data as any).grad
      if (grad) {
        const gradNorm = (grad as any).norm?.()?.item?.() ?? 0
        totalNorm += gradNorm * gradNorm
      }
    }
    totalNorm = Math.sqrt(totalNorm)

    // Clip if necessary
    if (totalNorm > maxNorm) {
      const clipCoef = maxNorm / (totalNorm + 1e-6)
      for (const param of params) {
        const grad = (param.data as any).grad
        if (grad) {
          ;(grad as any).mulScalarInplace?.(clipCoef)
        }
      }
    }
  }
}
