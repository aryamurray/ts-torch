/**
 * Composable Callback Objects
 *
 * Callbacks are composable objects that hook into the training loop.
 * Built-in callbacks replace common boilerplate (logging, early stopping, checkpointing).
 */

/**
 * Context passed to epoch-level callbacks
 */
export interface EpochContext {
  /** Current epoch number (1-indexed) */
  epoch: number
  /** Total number of epochs */
  totalEpochs?: number
  /** Train metrics for this epoch */
  metrics: Record<string, number>
  /** Validation metrics (if validation ran) */
  valMetrics?: Record<string, number>
  /** All epochs so far */
  history: History
  /** Total seconds since fit() started */
  elapsed: number
  /** The model being trained (for checkpoint callbacks) */
  model?: { save(path: string, metadata?: Record<string, unknown>): Promise<void> }
}

/**
 * Context passed to batch-level callbacks
 */
export interface BatchContext {
  /** Current step (batch number, 0-indexed) */
  step: number
  /** Current epoch (1-indexed) */
  epoch: number
  /** Total number of epochs */
  totalEpochs?: number
  /** Total number of batches per epoch (if known) */
  totalBatches?: number
  /** Loss for this batch */
  loss: number
}

/**
 * Result a callback can return to control training
 */
export type CallbackResult = { stop?: boolean } | void

/**
 * Callback interface with optional hooks
 */
export interface Callback {
  onTrainStart?(): CallbackResult | Promise<CallbackResult>
  onTrainEnd?(): CallbackResult | Promise<CallbackResult>
  onEpochStart?(ctx: { epoch: number; totalEpochs?: number }): CallbackResult | Promise<CallbackResult>
  onEpochEnd?(ctx: EpochContext): CallbackResult | Promise<CallbackResult>
  onBatchStart?(ctx: BatchContext): CallbackResult | Promise<CallbackResult>
  onBatchEnd?(ctx: BatchContext): CallbackResult | Promise<CallbackResult>
  onValidationStart?(): CallbackResult | Promise<CallbackResult>
  onValidationEnd?(): CallbackResult | Promise<CallbackResult>
}

// Import History type from trainer — but to avoid circular deps we define a minimal version here
// The actual History is defined in trainer.ts; this file references it via the EpochContext interface.
// We re-export the History and EpochRecord types from trainer.ts through the package index.

/**
 * Record for a single training epoch
 */
export interface EpochRecord {
  epoch: number
  train: { metrics: Record<string, number> }
  val?: { metrics: Record<string, number> }
  time: number
}

/**
 * Training history returned by fit()
 */
export interface History {
  epochs: EpochRecord[]
  totalTime: number
  config: Record<string, unknown> // serializable subset of TrainerOptions
}

/**
 * Console logger callback - auto-formats epoch metrics with elapsed time.
 * Replaces manual onEpochEnd logging boilerplate.
 *
 * @example
 * ```ts
 * new Trainer({ ..., callbacks: [consoleLogger()] })
 * ```
 */
export function consoleLogger(): Callback {
  return {
    onTrainStart() {
      console.log('Training started...\n')
    },
    onEpochEnd(ctx: EpochContext) {
      const parts = [`Epoch ${ctx.epoch} [${ctx.elapsed.toFixed(1)}s]`]

      // Train metrics
      for (const [key, value] of Object.entries(ctx.metrics)) {
        if (key === 'loss') {
          parts.push(`Loss: ${value.toFixed(4)}`)
        } else if (key === 'accuracy') {
          parts.push(`Acc: ${value.toFixed(2)}%`)
        } else {
          parts.push(`${key}: ${value.toFixed(4)}`)
        }
      }

      // Validation metrics
      if (ctx.valMetrics) {
        for (const [key, value] of Object.entries(ctx.valMetrics)) {
          if (key === 'accuracy') {
            parts.push(`Val Acc: ${value.toFixed(2)}%`)
          } else if (key === 'loss') {
            parts.push(`Val Loss: ${value.toFixed(4)}`)
          } else {
            parts.push(`val_${key}: ${value.toFixed(4)}`)
          }
        }
      }

      console.log(parts.join(' | '))
    },
    onTrainEnd() {
      console.log('\nTraining complete.')
    },
  }
}

/**
 * Early stopping callback - stops training when a monitored metric stops improving.
 *
 * @example
 * ```ts
 * callbacks: [earlyStop({ patience: 5, monitor: 'loss', mode: 'min' })]
 * ```
 */
export function earlyStop(opts: { patience: number; monitor?: string; mode?: 'min' | 'max' }): Callback {
  const { patience, monitor = 'loss', mode = 'min' } = opts
  let best = mode === 'min' ? Infinity : -Infinity
  let wait = 0

  return {
    onEpochEnd(ctx: EpochContext): CallbackResult {
      const value = ctx.valMetrics?.[monitor] ?? ctx.metrics[monitor]
      if (value === undefined) return

      const improved = mode === 'min' ? value < best : value > best
      if (improved) {
        best = value
        wait = 0
      } else {
        wait++
        if (wait >= patience) {
          console.log(`Early stopping: ${monitor} did not improve for ${patience} epochs`)
          return { stop: true }
        }
      }
    },
  }
}

/**
 * Checkpoint callback - saves model state at regular intervals.
 *
 * @example
 * ```ts
 * callbacks: [checkpoint({ every: 5, path: './checkpoints' })]
 * ```
 */
export function checkpoint(opts: { every: number; path?: string }): Callback {
  const dir = opts.path ?? './checkpoints'

  return {
    async onEpochEnd(ctx: EpochContext): Promise<void> {
      if (ctx.epoch % opts.every !== 0) return

      if (!ctx.model) {
        console.log(`[Checkpoint] Epoch ${ctx.epoch}: no model in context, skipping`)
        return
      }

      const { join } = await import('node:path')

      const epochDir = join(dir, `epoch-${ctx.epoch}`)

      await ctx.model.save(epochDir, {
        epoch: ctx.epoch,
        metrics: ctx.metrics,
        valMetrics: ctx.valMetrics,
      })

      console.log(`[Checkpoint] Saved epoch ${ctx.epoch} to ${epochDir}`)
    },
  }
}

/**
 * Logger namespace for convenience
 *
 * @example
 * ```ts
 * import { logger } from '@ts-torch/train'
 * callbacks: [logger.console()]
 * ```
 */
export const logger = {
  console: consoleLogger,
}
