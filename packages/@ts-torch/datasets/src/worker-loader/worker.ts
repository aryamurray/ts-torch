/**
 * Worker thread entry point for data loading
 *
 * This file runs in a worker thread and handles batch loading tasks.
 */

import { parentPort, workerData } from 'node:worker_threads'
import type {
  BatchMetadata,
  BatchTask,
  DType,
  MainToWorkerMessage,
  TransformConfig,
  WorkerConfig,
  WorkerDataset,
  WorkerToMainMessage,
} from './types.js'

// Dataset and transforms are initialized from workerData
let dataset: WorkerDataset | null = null
let transforms: Array<(input: unknown) => unknown | Promise<unknown>> = []
let isInitialized = false

/**
 * Initialize the worker with dataset and transforms
 */
async function initialize(config: WorkerConfig): Promise<void> {
  if (isInitialized) return

  // Dynamic import of the dataset module
  // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
  const datasetModule = await import(config.datasetModule)
  // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-member-access
  const datasetFactory = datasetModule[config.datasetExport]

  if (typeof datasetFactory !== 'function') {
    throw new Error(
      `Dataset export "${config.datasetExport}" is not a function in module "${config.datasetModule}"`,
    )
  }

  // eslint-disable-next-line @typescript-eslint/no-unsafe-argument
  dataset = (await datasetFactory(...config.datasetArgs)) as WorkerDataset

  // Initialize transforms from configs
  transforms = await createTransformsFromConfigs(config.transforms)

  isInitialized = true
}

/**
 * Create transform functions from serializable configs
 */
async function createTransformsFromConfigs(
  configs: TransformConfig[],
): Promise<Array<(input: unknown) => unknown | Promise<unknown>>> {
  const result: Array<(input: unknown) => unknown | Promise<unknown>> = []

  for (const config of configs) {
    const transform = createTransform(config)
    if (transform) {
      result.push(transform)
    }
  }

  return result
}

/**
 * Create a transform function from a config
 */
function createTransform(
  config: TransformConfig,
): ((input: unknown) => unknown | Promise<unknown>) | null {
  switch (config.type) {
    case 'normalize': {
      const mean = config.params['mean'] as number[] | number
      const std = config.params['std'] as number[] | number
      return (input: unknown) => normalizeTransform(input, mean, std)
    }

    case 'resize': {
      const size = config.params['size'] as [number, number]
      return (input: unknown) => resizeTransform(input, size)
    }

    case 'randomHorizontalFlip': {
      const p = (config.params['p'] as number) ?? 0.5
      return (input: unknown) => (Math.random() < p ? flipHorizontal(input) : input)
    }

    case 'randomVerticalFlip': {
      const p = (config.params['p'] as number) ?? 0.5
      return (input: unknown) => (Math.random() < p ? flipVertical(input) : input)
    }

    case 'toTensor':
      // Already tensor-like, pass through
      return null

    default:
      console.warn(`Unknown transform type: ${config.type}`)
      return null
  }
}

// Transform implementations (simplified - real implementations would be more complex)
function normalizeTransform(
  input: unknown,
  _mean: number[] | number,
  _std: number[] | number,
): unknown {
  // For now, pass through - real implementation would normalize array data
  return input
}

function resizeTransform(input: unknown, _size: [number, number]): unknown {
  // For now, pass through - real implementation would resize image data
  return input
}

function flipHorizontal(input: unknown): unknown {
  // For now, pass through - real implementation would flip image data
  return input
}

function flipVertical(input: unknown): unknown {
  // For now, pass through - real implementation would flip image data
  return input
}

/**
 * Load a batch of samples and write to SharedArrayBuffer
 */
async function loadBatch(
  task: BatchTask,
  buffer: SharedArrayBuffer,
): Promise<BatchMetadata> {
  if (!dataset) {
    throw new Error('Dataset not initialized')
  }

  const { indices, byteOffset, maxBytes } = task

  // Load all samples
  const samples: Array<{ data: unknown; label: unknown }> = []
  for (const idx of indices) {
    let sample = await Promise.resolve(dataset.getItem(idx))

    // Apply transforms
    for (const transform of transforms) {
      sample = await Promise.resolve(transform(sample))
    }

    samples.push(sample as { data: unknown; label: unknown })
  }

  // Determine data format from first sample
  const firstSample = samples[0]
  if (!firstSample) {
    throw new Error('Empty batch')
  }

  // Extract and flatten data
  const { dataArrays, dataShape, dataDtype } = extractData(samples, 'data')
  const { dataArrays: labelArrays, dataShape: labelShape, dataDtype: labelDtype } = extractData(
    samples,
    'label',
  )

  // Calculate sizes
  const dataBytesPerElement = getBytesPerElement(dataDtype)
  const labelBytesPerElement = getBytesPerElement(labelDtype)
  const dataElements = dataArrays.reduce((sum, arr) => sum + arr.length, 0)
  const labelElements = labelArrays.reduce((sum, arr) => sum + arr.length, 0)
  const dataByteSize = dataElements * dataBytesPerElement
  const labelByteSize = labelElements * labelBytesPerElement

  if (dataByteSize + labelByteSize > maxBytes) {
    throw new Error(
      `Batch data (${dataByteSize + labelByteSize} bytes) exceeds slot size (${maxBytes} bytes)`,
    )
  }

  // Write data to shared buffer
  const dataByteOffset = byteOffset
  const labelByteOffset = byteOffset + dataByteSize

  writeToBuffer(buffer, dataByteOffset, dataArrays, dataDtype)
  writeToBuffer(buffer, labelByteOffset, labelArrays, labelDtype)

  return {
    indices,
    dataShape: [indices.length, ...dataShape],
    labelShape: [indices.length, ...labelShape],
    dataDtype,
    labelDtype,
    dataByteOffset,
    dataByteSize,
    labelByteOffset,
    labelByteSize,
  }
}

/**
 * Extract data from samples and determine shape/dtype
 */
function extractData(
  samples: Array<{ data: unknown; label: unknown }>,
  key: 'data' | 'label',
): { dataArrays: number[][]; dataShape: number[]; dataDtype: DType } {
  const dataArrays: number[][] = []
  let dataShape: number[] = []
  let dataDtype: DType = 'float32'

  for (const sample of samples) {
    const value = sample[key]

    if (typeof value === 'number') {
      dataArrays.push([value])
      dataShape = []
    } else if (Array.isArray(value)) {
      const flat = flattenArray(value)
      dataArrays.push(flat.data)
      dataShape = flat.shape
    } else if (value instanceof Float32Array) {
      dataArrays.push(Array.from(value))
      dataShape = [value.length]
    } else if (value instanceof Float64Array) {
      dataArrays.push(Array.from(value))
      dataShape = [value.length]
      dataDtype = 'float64'
    } else if (value instanceof Int32Array) {
      dataArrays.push(Array.from(value))
      dataShape = [value.length]
      dataDtype = 'int32'
    } else if (value instanceof Uint8Array) {
      dataArrays.push(Array.from(value))
      dataShape = [value.length]
      dataDtype = 'uint8'
    } else if (isTypedArrayLike(value)) {
      // Handle tensor-like objects with toArray method
      const arr = (value as { toArray: () => number[] }).toArray()
      dataArrays.push(arr)
      dataShape = (value as { shape: number[] }).shape?.slice(0) ?? [arr.length]
    } else {
      throw new Error(`Unsupported data type for ${key}: ${typeof value}`)
    }
  }

  return { dataArrays, dataShape, dataDtype }
}

function isTypedArrayLike(value: unknown): boolean {
  return (
    value !== null &&
    typeof value === 'object' &&
    'toArray' in value &&
    typeof (value as { toArray: unknown }).toArray === 'function'
  )
}

/**
 * Flatten a nested array and return shape
 */
function flattenArray(arr: unknown[]): { data: number[]; shape: number[] } {
  const shape: number[] = []
  let current: unknown = arr

  while (Array.isArray(current)) {
    shape.push(current.length)
    current = current[0]
  }

  const data: number[] = []
  const flatten = (a: unknown): void => {
    if (Array.isArray(a)) {
      for (const item of a) {
        flatten(item)
      }
    } else {
      data.push(a as number)
    }
  }
  flatten(arr)

  return { data, shape }
}

/**
 * Get bytes per element for a dtype
 */
function getBytesPerElement(dtype: DType): number {
  switch (dtype) {
    case 'float32':
    case 'int32':
      return 4
    case 'float64':
    case 'int64':
      return 8
    case 'uint8':
      return 1
    default:
      return 4
  }
}

/**
 * Write arrays to SharedArrayBuffer
 */
function writeToBuffer(
  buffer: SharedArrayBuffer,
  byteOffset: number,
  arrays: number[][],
  dtype: DType,
): void {
  let offset = 0

  for (const arr of arrays) {
    switch (dtype) {
      case 'float32': {
        const view = new Float32Array(buffer, byteOffset + offset, arr.length)
        view.set(arr)
        offset += arr.length * 4
        break
      }
      case 'float64': {
        const view = new Float64Array(buffer, byteOffset + offset, arr.length)
        view.set(arr)
        offset += arr.length * 8
        break
      }
      case 'int32': {
        const view = new Int32Array(buffer, byteOffset + offset, arr.length)
        view.set(arr)
        offset += arr.length * 4
        break
      }
      case 'int64': {
        const view = new BigInt64Array(buffer, byteOffset + offset, arr.length)
        view.set(arr.map(BigInt))
        offset += arr.length * 8
        break
      }
      case 'uint8': {
        const view = new Uint8Array(buffer, byteOffset + offset, arr.length)
        view.set(arr)
        offset += arr.length
        break
      }
    }
  }
}

/**
 * Send a message to the main thread
 */
function postMessage(message: WorkerToMainMessage): void {
  parentPort?.postMessage(message)
}

// Initialize from workerData if available
const config = workerData as WorkerConfig | undefined

if (config) {
  initialize(config)
    .then(() => {
      postMessage({ type: 'ready' })
    })
    .catch((error) => {
      console.error('Worker initialization failed:', error)
      process.exit(1)
    })
}

// Handle messages from main thread
parentPort?.on('message', async (message: MainToWorkerMessage) => {
  switch (message.type) {
    case 'loadBatch':
      try {
        const metadata = await loadBatch(message.task, message.buffer)
        postMessage({
          type: 'batchReady',
          taskId: message.task.taskId,
          metadata,
        })
      } catch (error) {
        postMessage({
          type: 'error',
          taskId: message.task.taskId,
          error: error instanceof Error ? error.message : String(error),
        })
      }
      break

    case 'shutdown':
      process.exit(0)
      break
  }
})
