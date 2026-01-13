/**
 * Types for Worker-based DataLoader
 */

import type { Tensor } from '@ts-torch/core'

/**
 * Options for WorkerDataLoader
 */
export interface WorkerDataLoaderOptions {
  /** Number of worker threads (default: os.cpus().length) */
  numWorkers?: number

  /** Batch size */
  batchSize: number

  /** Number of batches to prefetch (default: 2) */
  prefetchCount?: number

  /** Shuffle data each epoch */
  shuffle?: boolean

  /** Drop incomplete last batch */
  dropLast?: boolean

  /** Transform configurations (serializable) */
  transforms?: TransformConfig[]

  /** Maximum bytes per batch slot for buffer sizing */
  maxBatchBytes?: number

  /** Timeout for worker tasks in ms (default: 30000) */
  workerTimeout?: number
}

/**
 * Slot states for ring buffer
 */
export const enum SlotState {
  EMPTY = 0,
  LOADING = 1,
  READY = 2,
  CONSUMING = 3,
}

/**
 * Metadata for a batch stored in a buffer slot
 */
export interface BatchMetadata {
  /** Sample indices in this batch */
  indices: number[]

  /** Shape of data tensor [batchSize, ...itemShape] */
  dataShape: number[]

  /** Shape of label tensor [batchSize, ...labelShape] */
  labelShape: number[]

  /** Data type for data tensor */
  dataDtype: DType

  /** Data type for label tensor */
  labelDtype: DType

  /** Byte offset where data starts in the slot */
  dataByteOffset: number

  /** Byte size of data */
  dataByteSize: number

  /** Byte offset where labels start in the slot */
  labelByteOffset: number

  /** Byte size of labels */
  labelByteSize: number
}

export type DType = 'float32' | 'float64' | 'int32' | 'int64' | 'uint8'

/**
 * A slot in the ring buffer
 */
export interface BufferSlot {
  /** Slot index in ring buffer */
  index: number

  /** Byte offset in SharedArrayBuffer where this slot's data region starts */
  byteOffset: number

  /** Maximum bytes this slot can hold */
  maxBytes: number
}

/**
 * A slot with its data, returned from waitForReady
 */
export interface ReadySlot extends BufferSlot {
  /** The SharedArrayBuffer containing the data */
  buffer: SharedArrayBuffer

  /** Batch metadata */
  metadata: BatchMetadata
}

/**
 * A task to load a batch
 */
export interface BatchTask {
  /** Unique task ID */
  taskId: number

  /** Sample indices to load */
  indices: number[]

  /** Slot index in ring buffer to write to */
  slotIndex: number

  /** Byte offset in the SharedArrayBuffer for this slot */
  byteOffset: number

  /** Maximum bytes available in this slot */
  maxBytes: number
}

/**
 * Configuration passed to workers on initialization
 */
export interface WorkerConfig {
  /** Path or module to load dataset from */
  datasetModule: string

  /** Export name of the dataset factory */
  datasetExport: string

  /** Arguments to pass to dataset factory */
  datasetArgs: unknown[]

  /** Transform configurations */
  transforms: TransformConfig[]
}

/**
 * Messages from main thread to worker
 */
export type MainToWorkerMessage =
  | { type: 'loadBatch'; task: BatchTask; buffer: SharedArrayBuffer }
  | { type: 'shutdown' }

/**
 * Messages from worker to main thread
 */
export type WorkerToMainMessage =
  | { type: 'ready' }
  | { type: 'batchReady'; taskId: number; metadata: BatchMetadata }
  | { type: 'error'; taskId: number; error: string }

/**
 * Transform configuration (serializable)
 */
export interface TransformConfig {
  type: TransformType
  params: Record<string, unknown>
}

export type TransformType =
  | 'resize'
  | 'normalize'
  | 'randomCrop'
  | 'centerCrop'
  | 'randomHorizontalFlip'
  | 'randomVerticalFlip'
  | 'randomRotation'
  | 'colorJitter'
  | 'toTensor'
  | 'custom'

/**
 * Dataset interface that workers need
 */
export interface WorkerDataset<T = unknown> {
  getItem(index: number): T | Promise<T>
  readonly length: number
}

/**
 * Result batch format
 */
export interface TensorPair {
  data: Tensor
  label: Tensor
}

/**
 * Raw batch data before tensor conversion
 */
export interface RawBatchData {
  data: Float32Array | Float64Array | Int32Array | BigInt64Array | Uint8Array
  label: Float32Array | Float64Array | Int32Array | BigInt64Array | Uint8Array
  dataShape: number[]
  labelShape: number[]
  dataDtype: DType
  labelDtype: DType
}
