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

import type { Tensor, DeviceType } from '@ts-torch/core'
import { run, runAsync, cat } from '@ts-torch/core'
import type { Module } from '@ts-torch/nn'
import type { Optimizer } from '@ts-torch/optim'
import { crossEntropyLoss, mseLoss, nllLoss } from '@ts-torch/optim'
import type { OptimizerConfig } from './optimizers.js'
import type { SchedulerConfig } from './schedulers.js'
import type { LRScheduler } from '@ts-torch/optim'
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

  // ==================== Learning Rate Scheduling ====================

  /** Learning rate scheduler configuration */
  scheduler?: SchedulerConfig

  // ==================== Validation ====================

  /** Validation data loader */
  validateOn?: AsyncIterable<any>

  /** Validate every N epochs (default: 1) */
  validateEvery?: number

  // ==================== Callbacks ====================

  /** Called at the start of each epoch */
  onEpochStart?: (ctx: { epoch: number; totalEpochs: number }) => void | Promise<void>

  /** Called at the end of each epoch */
  onEpochEnd?: (ctx: EpochContext) => void | Promise<void>

  /** Called at the start of each batch, before forward pass */
  onBatchStart?: (ctx: { batch: any; step: number; epoch: number }) => void | Promise<void>

  /** Called at the end of each batch */
  onBatchEnd?: (ctx: BatchContext) => void | Promise<void>

  /**
   * Called after forward pass with predictions and targets.
   * Can modify or replace the loss before backward pass.
   *
   * @example
   * ```ts
   * onForward: ({ predictions, targets, loss }) => {
   *   // Add custom regularization
   *   return loss.add(myRegularizationTerm)
   * }
   * ```
   */
  onForward?: (ctx: {
    predictions: Tensor
    targets: Tensor
    loss: Tensor
    batch: any
  }) => Tensor | void | Promise<Tensor | void>

  /**
   * Called after backward pass, before optimizer step.
   * Useful for gradient inspection or custom gradient manipulation.
   */
  onBackward?: (ctx: { loss: number; step: number; epoch: number }) => void | Promise<void>

  /**
   * Called before optimizer.step(). Return false to skip the optimizer step.
   * Useful for implementing custom gradient accumulation or conditional updates.
   */
  onBeforeOptimizerStep?: (ctx: { step: number; epoch: number }) => boolean | void | Promise<boolean | void>

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
 * Provides a high-level API for training models without manual training loops.
 * Handles optimizer setup, forward/backward passes, and metrics tracking.
 *
 * @template M - Model type (must extend Module with any device type)
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
export class Trainer<M extends Module<any, any, any, DeviceType>> {
  private model: M
  private optimizer: Optimizer | null = null
  private scheduler: LRScheduler | null = null
  private schedulerConfig: SchedulerConfig | null = null
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
      scheduler: schedulerConfig,
      loss,
      metrics: metricsConfig = { loss: true },
      accumulate = 1,
      clipGradNorm,
      validateOn,
      validateEvery = 1,
      onEpochStart,
      onEpochEnd,
      onBatchStart,
      onBatchEnd,
      onForward,
      onBackward,
      onBeforeOptimizerStep,
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

    // Create scheduler if provided
    this.schedulerConfig = schedulerConfig ?? null
    this.scheduler = schedulerConfig ? schedulerConfig.create(this.optimizer) : null

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

      // Epoch start callback
      if (onEpochStart) {
        await onEpochStart({ epoch, totalEpochs: epochs })
      }

      let batchIdx = 0
      let epochLoss = 0
      let batchCount = 0

      // Iterate over batches
      for await (const batch of data) {
        // Batch start callback
        if (onBatchStart) {
          await onBatchStart({ batch, step: batchIdx, epoch })
        }

        const batchLoss = await this.trainStep(
          batch,
          accumulate,
          clipGradNorm,
          batchIdx,
          epoch,
          { onForward, onBackward, onBeforeOptimizerStep },
          debug,
        )

        // FREE batch tensors after use to prevent memory leak
        // Batch tensors are created by the pipeline and must be explicitly freed
        this.freeBatchTensors(batch)

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

        // Step scheduler if configured for batch-level stepping
        if (this.scheduler && this.schedulerConfig?.stepOn === 'batch' && !this.schedulerConfig.needsMetrics) {
          this.scheduler.step()
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

      // Step scheduler at epoch end (default behavior)
      if (this.scheduler && this.schedulerConfig?.stepOn !== 'batch') {
        if (this.schedulerConfig?.needsMetrics) {
          // ReduceLROnPlateau needs metrics - use validation loss if available, otherwise training loss
          const metricValue = valMetrics?.loss ?? epochMetrics.loss
          // Cast to any to call step with metric (ReduceLROnPlateau overrides step to accept metric)
          ;(this.scheduler as any).step(metricValue)
        } else {
          this.scheduler.step()
        }
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

        // FREE batch tensors after use to prevent memory leak
        this.freeBatchTensors(batch)
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
    epoch: number,
    hooks: {
      onForward?: FitOptions['onForward']
      onBackward?: FitOptions['onBackward']
      onBeforeOptimizerStep?: FitOptions['onBeforeOptimizerStep']
    },
    _debug?: FitOptions['debug'],
  ): Promise<number> {
    const shouldStep = (batchIdx + 1) % accumulate === 0
    const hasAsyncHooks = !!(hooks.onForward || hooks.onBackward || hooks.onBeforeOptimizerStep)

    let lossValue = 0

    if (!hasAsyncHooks) {
      // Fast path: single scope keeps entire computation graph alive through backward pass.
      // Previously forward and backward were in separate scopes, which caused intermediate
      // computation graph tensors to be freed before backward() could traverse them (use-after-free).
      run(() => {
        if (batchIdx % accumulate === 0) {
          this.optimizer!.zeroGrad()
        }

        const { predictions, targets } = this.forwardBatch(batch)
        const loss = this.lossFn(predictions, targets)

        const scaledLoss = accumulate > 1 ? (loss as any).divScalar(accumulate) : loss
        ;(scaledLoss as any).backward()

        lossValue = (loss as any).item?.() ?? (loss as any).toArray?.()[0] ?? 0

        for (const metric of this.metrics) {
          metric.update(predictions, targets, loss)
        }

        if (shouldStep) {
          if (clipGradNorm) this.clipGradients(clipGradNorm)
          this.optimizer!.step()
        }
      })
    } else {
      // Hook path: forward + backward in one async scope to support async hooks
      // while keeping the computation graph alive through backward pass.
      await runAsync(async () => {
        if (batchIdx % accumulate === 0) {
          this.optimizer!.zeroGrad()
        }

        const { predictions, targets } = this.forwardBatch(batch)
        let loss: Tensor = this.lossFn(predictions, targets)

        // onForward hook - can modify or replace the loss
        if (hooks.onForward) {
          const modifiedLoss = await hooks.onForward({ predictions, targets, loss, batch })
          if (modifiedLoss) {
            loss = modifiedLoss
          }
        }

        const scaledLoss = accumulate > 1 ? (loss as any).divScalar(accumulate) : loss
        ;(scaledLoss as any).backward()

        lossValue = (loss as any).item?.() ?? (loss as any).toArray?.()[0] ?? 0

        for (const metric of this.metrics) {
          metric.update(predictions, targets, loss)
        }
      })

      // onBackward hook (async, outside scope - loss value already extracted)
      if (hooks.onBackward) {
        await hooks.onBackward({ loss: lossValue, step: batchIdx, epoch })
      }

      // Optimizer step in separate scope (only touches model params/grads, not computation graph)
      if (shouldStep) {
        let shouldDoStep = true
        if (hooks.onBeforeOptimizerStep) {
          const result = await hooks.onBeforeOptimizerStep({ step: batchIdx, epoch })
          if (result === false) {
            shouldDoStep = false
          }
        }

        if (shouldDoStep) {
          run(() => {
            if (clipGradNorm) this.clipGradients(clipGradNorm)
            this.optimizer!.step()
          })
        }
      }
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
      // Array of {data, label} objects - concatenate into batch tensors
      if (batch[0] && 'data' in batch[0] && 'label' in batch[0]) {
        // Extract all data and label tensors
        const dataTensors = batch.map((item: any) => item.data)
        const labelTensors = batch.map((item: any) => item.label)

        // Concatenate along batch dimension (dim 0)
        data = cat(dataTensors as Tensor[], 0) as Tensor
        targets = cat(labelTensors as Tensor[], 0) as Tensor
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

    // Move data to model's device (just-in-time transfer)
    const modelDevice = this.getModelDevice()
    if (modelDevice && modelDevice !== 'cpu') {
      data = this.moveToDevice(data, modelDevice)
      targets = this.moveToDevice(targets, modelDevice)
    }

    // Forward pass
    const predictions = this.model.forward(data as any) as unknown as Tensor

    // Compute loss if metrics need it
    const loss = this.lossFn(predictions as any, targets as any) as Tensor

    return { predictions, targets, loss }
  }

  /**
   * Get the device the model is on
   * @internal
   */
  private getModelDevice(): DeviceType | undefined {
    // Try to get device from first parameter
    const params = this.model.parameters()
    if (params.length > 0) {
      const firstParam = params[0]
      // Parameter.data is a Tensor which has a .device property
      return (firstParam?.data as any)?.device
    }
    return undefined
  }

  /**
   * Move a tensor to the specified device
   * @internal
   */
  private moveToDevice(tensor: Tensor, device: DeviceType): Tensor {
    // Tensor has a .device property (string like 'cpu' or 'cuda')
    const currentDevice = (tensor as any).device ?? 'cpu'
    if (currentDevice === device) {
      return tensor
    }
    // Use .move() method to transfer to target device
    if (typeof (tensor as any).move === 'function') {
      return (tensor as any).move(device)
    }
    return tensor
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

  /**
   * Free tensors in a batch to prevent memory leaks
   *
   * Batch tensors are created by the data pipeline and transferred to GPU.
   * They must be explicitly freed after each training/evaluation step.
   *
   * @param batch - Batch to free
   * @internal
   */
  private freeBatchTensors(batch: any): void {
    if (Array.isArray(batch)) {
      // Array of {data, label} objects
      for (const item of batch) {
        if (item && 'data' in item && 'label' in item) {
          item.data?.free?.()
          item.label?.free?.()
        }
      }
    } else if (batch && 'data' in batch && 'label' in batch) {
      // Single {data, label} object
      batch.data?.free?.()
      batch.label?.free?.()
    } else if (batch && 'data' in batch && 'targets' in batch) {
      // {data, targets} object
      batch.data?.free?.()
      batch.targets?.free?.()
    }
  }
}
