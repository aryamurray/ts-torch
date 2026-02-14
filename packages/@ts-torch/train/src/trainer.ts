/**
 * Declarative Trainer
 *
 * The Trainer class provides a declarative API for training neural networks.
 * All configuration is passed via a single options object to the constructor.
 *
 * @example
 * ```ts
 * const trainer = new Trainer({
 *   model,
 *   data: trainLoader,
 *   epochs: 3,
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

import type { Tensor, DeviceType } from '@ts-torch/core'
import { run } from '@ts-torch/core'
import type { Module } from '@ts-torch/nn'
import type { Optimizer } from '@ts-torch/optim'
import type { OptimizerConfig } from './optimizers.js'
import type { SchedulerConfig } from './schedulers.js'
import type { LRScheduler } from '@ts-torch/optim'
import type { LossConfig } from './loss.js'
import { resolveLoss, type LossFn } from './loss.js'
import { createMetrics, computeMetrics, resetMetrics, type Metric, type MetricSpec } from './metrics.js'
import type { Callback, EpochContext, History, EpochRecord } from './callbacks.js'

// ==================== Batch & TensorTree ====================

/**
 * A tree of tensors — single tensor, array, or named record.
 */
export type TensorTree = Tensor | Tensor[] | Record<string, Tensor>

/**
 * A typed batch with input and target tensor trees.
 * Defaults to single tensors (the common case).
 */
export interface Batch<I extends TensorTree = Tensor, T extends TensorTree = Tensor> {
  input: I
  target: T
}

/**
 * Map a function over every tensor in a TensorTree, preserving structure.
 */
export function mapTensorTree<T extends TensorTree>(tree: T, fn: (t: Tensor) => Tensor): T {
  if (Array.isArray(tree)) {
    return tree.map(fn) as T
  }
  if (tree && typeof tree === 'object' && !('shape' in tree)) {
    // Record<string, Tensor>
    const result: Record<string, Tensor> = {}
    for (const [key, val] of Object.entries(tree)) {
      result[key] = fn(val as Tensor)
    }
    return result as T
  }
  // Single tensor
  return fn(tree as Tensor) as T
}

/**
 * Apply a side-effecting function to every tensor in a TensorTree.
 */
function forEachTensor(tree: TensorTree, fn: (t: Tensor) => void): void {
  if (Array.isArray(tree)) {
    for (const t of tree) fn(t)
  } else if (tree && typeof tree === 'object' && !('shape' in tree)) {
    for (const val of Object.values(tree)) fn(val as Tensor)
  } else {
    fn(tree as Tensor)
  }
}

// ==================== TrainerOptions ====================

/**
 * Trainer options — all configuration in one place
 */
export interface TrainerOptions<M = Module<any, any, any, DeviceType>> {
  model: M
  data: AsyncIterable<Batch>
  epochs: number
  optimizer: OptimizerConfig
  loss: LossConfig
  metrics?: MetricSpec[]
  validation?: AsyncIterable<Batch>
  validateEvery?: number
  callbacks?: Callback[]
  /** Shorthand sugar — wraps into an anonymous Callback internally */
  onEpochEnd?: (ctx: EpochContext) => void | Promise<void>
  scheduler?: SchedulerConfig
  accumulate?: number
  clipGradNorm?: number
  amp?: boolean
  precision?: 'fp32' | 'fp16' | 'bf16'
  debug?: { tensors?: boolean; memory?: boolean; timing?: boolean; gradients?: boolean }
}

/**
 * Evaluation options (for evaluate overloads)
 */
export interface EvaluateOptions {
  metrics?: MetricSpec[]
}

// ==================== Trainer ====================

/**
 * Declarative Trainer for neural network training.
 *
 * All configuration is passed to the constructor. `.fit()` is zero-arg.
 */
export class Trainer<M extends Module<any, any, any, DeviceType>> {
  private model: M
  private data: AsyncIterable<Batch>
  private epochs: number
  private optimizerConfig: OptimizerConfig
  private lossFn: LossFn
  private lossConfig: LossConfig
  private metricsSpecs: MetricSpec[]
  private validation?: AsyncIterable<Batch>
  private validateEvery: number
  private callbacks: Callback[]
  private schedulerConfig: SchedulerConfig | null
  private accumulate: number
  private clipGradNorm?: number

  private optimizer: Optimizer | null = null
  private scheduler: LRScheduler | null = null
  private metrics: Metric[] = []

  constructor(options: TrainerOptions<M>) {
    this.model = options.model
    this.data = options.data
    this.epochs = options.epochs
    this.optimizerConfig = options.optimizer
    this.lossConfig = options.loss
    this.lossFn = resolveLoss(options.loss) // fail fast
    this.metricsSpecs = options.metrics ?? ['loss']
    this.validation = options.validation
    this.validateEvery = options.validateEvery ?? 1
    this.schedulerConfig = options.scheduler ?? null
    this.accumulate = options.accumulate ?? 1
    this.clipGradNorm = options.clipGradNorm

    // Build callbacks array
    this.callbacks = [...(options.callbacks ?? [])]

    // Wrap onEpochEnd shorthand into a Callback
    if (options.onEpochEnd) {
      const fn = options.onEpochEnd
      this.callbacks.push({ onEpochEnd: fn })
    }
  }

  /**
   * Train the model. Returns training history.
   */
  async fit(): Promise<History> {
    const startTime = Date.now()

    // Create optimizer
    this.optimizer = this.optimizerConfig.create(this.model)

    // Create scheduler
    this.scheduler = this.schedulerConfig ? this.schedulerConfig.create(this.optimizer) : null

    // Create metrics
    this.metrics = createMetrics(this.metricsSpecs)

    // Build serializable config for history
    const serializableConfig: Record<string, unknown> = {
      epochs: this.epochs,
      optimizer: this.optimizerConfig.name,
      loss: this.lossConfig,
      metrics: this.metricsSpecs,
      accumulate: this.accumulate,
    }
    if (this.clipGradNorm) serializableConfig.clipGradNorm = this.clipGradNorm

    const history: History = {
      epochs: [],
      totalTime: 0,
      config: serializableConfig,
    }

    // Set model to training mode
    this.model.train()

    // onTrainStart
    for (const cb of this.callbacks) {
      if (cb.onTrainStart) await cb.onTrainStart()
    }

    // Training loop
    for (let epoch = 1; epoch <= this.epochs; epoch++) {
      const epochStart = Date.now()
      resetMetrics(this.metrics)

      // onEpochStart
      for (const cb of this.callbacks) {
        if (cb.onEpochStart) await cb.onEpochStart({ epoch })
      }

      let batchIdx = 0
      let epochLoss = 0
      let batchCount = 0

      for await (const batch of this.data) {
        // onBatchStart
        for (const cb of this.callbacks) {
          if (cb.onBatchStart) await cb.onBatchStart({ step: batchIdx, epoch, loss: 0 })
        }

        const batchLoss = this.trainStep(batch, batchIdx)

        // Free batch tensors
        this.freeBatchTensors(batch)

        epochLoss += batchLoss
        batchCount++

        // onBatchEnd
        for (const cb of this.callbacks) {
          if (cb.onBatchEnd) await cb.onBatchEnd({ step: batchIdx, epoch, loss: batchLoss })
        }

        // Step scheduler at batch level
        if (this.scheduler && this.schedulerConfig?.stepOn === 'batch' && !this.schedulerConfig.needsMetrics) {
          this.scheduler.step()
        }

        batchIdx++
      }

      // Compute epoch metrics
      const epochMetrics = computeMetrics(this.metrics)
      epochMetrics.loss = epochLoss / batchCount

      // Validation
      let valMetrics: Record<string, number> | undefined
      if (this.validation && epoch % this.validateEvery === 0) {
        valMetrics = await this.evaluate(this.validation)
      }

      // Step scheduler at epoch end
      if (this.scheduler && this.schedulerConfig?.stepOn !== 'batch') {
        if (this.schedulerConfig?.needsMetrics) {
          const metricValue = valMetrics?.loss ?? epochMetrics.loss
          ;(this.scheduler as any).step(metricValue)
        } else {
          this.scheduler.step()
        }
      }

      const epochTime = (Date.now() - epochStart) / 1000
      const elapsed = (Date.now() - startTime) / 1000

      const epochRecord: EpochRecord = {
        epoch,
        train: { metrics: { ...epochMetrics } },
        val: valMetrics ? { metrics: { ...valMetrics } } : undefined,
        time: epochTime,
      }
      history.epochs.push(epochRecord)
      history.totalTime = elapsed

      // Build EpochContext for callbacks
      const epochCtx: EpochContext = {
        epoch,
        metrics: epochMetrics,
        valMetrics,
        history,
        elapsed,
        model: this.model as any,
      }

      // onEpochEnd callbacks — check for stop signal
      let shouldStop = false
      for (const cb of this.callbacks) {
        if (cb.onEpochEnd) {
          const result = await cb.onEpochEnd(epochCtx)
          if (result && typeof result === 'object' && result.stop) {
            shouldStop = true
          }
        }
      }

      if (shouldStop) break
    }

    // onTrainEnd
    for (const cb of this.callbacks) {
      if (cb.onTrainEnd) await cb.onTrainEnd()
    }

    return history
  }

  /**
   * Evaluate the model on data.
   *
   * - Zero-arg: uses configured `validation` data and `metrics`
   * - `evaluate(data)`: override data, use configured metrics
   * - `evaluate(data, { metrics })`: override both
   */
  async evaluate(data?: AsyncIterable<Batch>, options?: EvaluateOptions): Promise<Record<string, number>> {
    const evalData = data ?? this.validation
    if (!evalData) {
      throw new Error('No validation data configured. Pass data to evaluate() or set validation in TrainerOptions.')
    }

    const evalSpecs = options?.metrics ?? this.metricsSpecs
    const evalMetrics = createMetrics(evalSpecs)

    this.model.eval()

    try {
      for await (const batch of evalData) {
        run(() => {
          const { predictions, targets, loss } = this.forwardBatch(batch)
          for (const metric of evalMetrics) {
            metric.update(predictions, targets, loss)
          }
        })

        this.freeBatchTensors(batch)
      }

      return computeMetrics(evalMetrics)
    } finally {
      this.model.train()
    }
  }

  /**
   * Execute a single training step
   * @internal
   */
  private trainStep(batch: Batch, batchIdx: number): number {
    const shouldStep = (batchIdx + 1) % this.accumulate === 0
    let lossValue = 0

    run(() => {
      if (batchIdx % this.accumulate === 0) {
        this.optimizer!.zeroGrad()
      }

      const { predictions, targets } = this.forwardBatch(batch)
      const loss = this.lossFn(predictions, targets)

      const scaledLoss = this.accumulate > 1 ? (loss as any).divScalar(this.accumulate) : loss
      ;(scaledLoss as any).backward()

      lossValue = (loss as any).item?.() ?? (loss as any).toArray?.()[0] ?? 0

      for (const metric of this.metrics) {
        metric.update(predictions, targets, loss)
      }

      if (shouldStep) {
        if (this.clipGradNorm) this.clipGradients(this.clipGradNorm)
        this.optimizer!.step()
      }
    })

    return lossValue
  }

  /**
   * Process a batch and return predictions, targets, and loss
   * @internal
   */
  private forwardBatch(batch: Batch): { predictions: Tensor; targets: Tensor; loss: Tensor } {
    let inputTensor: Tensor = batch.input as Tensor
    let targetTensor: Tensor = batch.target as Tensor

    // Move to model's device if needed
    const modelDevice = this.getModelDevice()
    if (modelDevice && modelDevice !== 'cpu') {
      inputTensor = this.moveToDevice(inputTensor, modelDevice)
      targetTensor = this.moveToDevice(targetTensor, modelDevice)
    }

    const predictions = this.model.forward(inputTensor as any) as unknown as Tensor
    const loss = this.lossFn(predictions as any, targetTensor as any) as Tensor

    return { predictions, targets: targetTensor, loss }
  }

  /**
   * Get the device the model is on
   * @internal
   */
  private getModelDevice(): DeviceType | undefined {
    const params = this.model.parameters()
    if (params.length > 0) {
      const firstParam = params[0]
      return (firstParam?.data as any)?.device
    }
    return undefined
  }

  /**
   * Move a tensor to the specified device
   * @internal
   */
  private moveToDevice(tensor: Tensor, device: DeviceType): Tensor {
    const currentDevice = (tensor as any).device ?? 'cpu'
    if (currentDevice === device) return tensor
    if (typeof (tensor as any).move === 'function') return (tensor as any).move(device)
    return tensor
  }

  /**
   * Clip gradients by norm
   * @internal
   */
  private clipGradients(maxNorm: number): void {
    const params = this.model.parameters()

    let totalNorm = 0
    for (const param of params) {
      const grad = (param.data as any).grad
      if (grad) {
        const gradNorm = (grad as any).norm?.()?.item?.() ?? 0
        totalNorm += gradNorm * gradNorm
      }
    }
    totalNorm = Math.sqrt(totalNorm)

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
   * @internal
   */
  private freeBatchTensors(batch: Batch): void {
    forEachTensor(batch.input, (t) => (t as any).free?.())
    forEachTensor(batch.target, (t) => (t as any).free?.())
  }
}
