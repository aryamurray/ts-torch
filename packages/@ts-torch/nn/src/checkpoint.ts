/**
 * Model Checkpointing
 *
 * Custom binary format for saving and loading model state.
 * Designed for fast I/O and compatibility with the ts-torch ecosystem.
 *
 * Binary Format:
 * ```
 * ┌─────────────────────────────────────────────────────────┐
 * │ Magic bytes: "TSNN" (4 bytes)                           │
 * │ Version: uint32 (4 bytes)                               │
 * │ Metadata length: uint32 (4 bytes)                       │
 * │ Metadata JSON: UTF-8 string                             │
 * ├─────────────────────────────────────────────────────────┤
 * │ Tensor count: uint32                                    │
 * │ For each tensor:                                        │
 * │   - Name length: uint32                                 │
 * │   - Name: UTF-8 string                                  │
 * │   - Shape length: uint32                                │
 * │   - Shape: uint32[]                                     │
 * │   - Dtype: uint8 (0=f32, 1=f64, 2=i32, 3=u8)            │
 * │   - Data length: uint32                                 │
 * │   - Data: raw bytes                                     │
 * └─────────────────────────────────────────────────────────┘
 * ```
 *
 * @example
 * ```ts
 * import { saveCheckpoint, loadCheckpoint } from '@ts-torch/nn'
 *
 * // Save model state
 * await saveCheckpoint('./model.ckpt', {
 *   tensors: model.stateDict(),
 *   metadata: { epoch: 10, loss: 0.05 }
 * })
 *
 * // Load model state
 * const checkpoint = await loadCheckpoint('./model.ckpt')
 * model.loadStateDict(checkpoint.tensors)
 * ```
 */

import { readFile, writeFile } from 'node:fs/promises'

// ==================== Constants ====================

/** Magic bytes to identify checkpoint files */
const MAGIC = new Uint8Array([0x54, 0x53, 0x4e, 0x4e]) // "TSNN"

/** Current format version */
const VERSION = 1

/** Data type identifiers */
const DTYPE = {
  float32: 0,
  float64: 1,
  int32: 2,
  uint8: 3,
} as const

type DTypeName = keyof typeof DTYPE

// ==================== Types ====================

/**
 * Tensor data for serialization
 */
export interface TensorData {
  /** Raw data */
  data: Float32Array | Float64Array | Int32Array | Uint8Array
  /** Tensor shape */
  shape: number[]
  /** Data type */
  dtype: DTypeName
}

/**
 * Checkpoint data structure
 */
export interface CheckpointData {
  /** Named tensor parameters */
  tensors: Record<string, TensorData>
  /** Arbitrary metadata (JSON-serializable) */
  metadata?: Record<string, unknown>
}

/**
 * State dict type - maps parameter names to tensor data
 */
export type StateDict = Record<string, TensorData>

// ==================== Serialization ====================

/**
 * Save checkpoint to file
 *
 * @param path - File path to save to
 * @param data - Checkpoint data (tensors + metadata)
 */
export async function saveCheckpoint(path: string, data: CheckpointData): Promise<void> {
  const buffer = encodeCheckpoint(data)
  await writeFile(path, buffer)
}

/**
 * Load checkpoint from file
 *
 * @param path - File path to load from
 * @returns Checkpoint data
 */
export async function loadCheckpoint(path: string): Promise<CheckpointData> {
  const buffer = await readFile(path)
  return decodeCheckpoint(new Uint8Array(buffer))
}

/**
 * Encode checkpoint data to binary buffer
 *
 * @param data - Checkpoint data
 * @returns Binary buffer
 */
export function encodeCheckpoint(data: CheckpointData): Uint8Array {
  const chunks: Uint8Array[] = []

  // 1. Magic bytes
  chunks.push(MAGIC)

  // 2. Version
  chunks.push(encodeUint32(VERSION))

  // 3. Metadata
  const metadataJson = JSON.stringify(data.metadata ?? {})
  const metadataBytes = new TextEncoder().encode(metadataJson)
  chunks.push(encodeUint32(metadataBytes.length))
  chunks.push(metadataBytes)

  // 4. Tensor count
  const tensorNames = Object.keys(data.tensors)
  chunks.push(encodeUint32(tensorNames.length))

  // 5. Each tensor
  for (const name of tensorNames) {
    const tensor = data.tensors[name]!
    chunks.push(encodeTensor(name, tensor))
  }

  // Concatenate all chunks
  const totalLength = chunks.reduce((sum, chunk) => sum + chunk.length, 0)
  const result = new Uint8Array(totalLength)
  let offset = 0
  for (const chunk of chunks) {
    result.set(chunk, offset)
    offset += chunk.length
  }

  return result
}

/**
 * Decode checkpoint data from binary buffer
 *
 * @param buffer - Binary buffer
 * @returns Checkpoint data
 */
export function decodeCheckpoint(buffer: Uint8Array): CheckpointData {
  let offset = 0

  // 1. Verify magic bytes
  const magic = buffer.slice(offset, offset + 4)
  offset += 4
  if (!arraysEqual(magic, MAGIC)) {
    throw new Error('Invalid checkpoint file: bad magic bytes')
  }

  // 2. Read version
  const version = decodeUint32(buffer, offset)
  offset += 4
  if (version !== VERSION) {
    throw new Error(`Unsupported checkpoint version: ${version} (expected ${VERSION})`)
  }

  // 3. Read metadata
  const metadataLength = decodeUint32(buffer, offset)
  offset += 4
  const metadataBytes = buffer.slice(offset, offset + metadataLength)
  offset += metadataLength
  const metadataJson = new TextDecoder().decode(metadataBytes)
  const metadata = metadataJson ? JSON.parse(metadataJson) : {}

  // 4. Read tensor count
  const tensorCount = decodeUint32(buffer, offset)
  offset += 4

  // 5. Read each tensor
  const tensors: Record<string, TensorData> = {}
  for (let i = 0; i < tensorCount; i++) {
    const { name, tensor, bytesRead } = decodeTensor(buffer, offset)
    offset += bytesRead
    tensors[name] = tensor
  }

  return { tensors, metadata }
}

// ==================== Tensor Encoding ====================

/**
 * Encode a single tensor
 */
function encodeTensor(name: string, tensor: TensorData): Uint8Array {
  const chunks: Uint8Array[] = []

  // Name
  const nameBytes = new TextEncoder().encode(name)
  chunks.push(encodeUint32(nameBytes.length))
  chunks.push(nameBytes)

  // Shape
  chunks.push(encodeUint32(tensor.shape.length))
  for (const dim of tensor.shape) {
    chunks.push(encodeUint32(dim))
  }

  // Dtype
  chunks.push(new Uint8Array([DTYPE[tensor.dtype]]))

  // Data
  const dataBytes = new Uint8Array(tensor.data.buffer, tensor.data.byteOffset, tensor.data.byteLength)
  chunks.push(encodeUint32(dataBytes.length))
  chunks.push(dataBytes)

  // Concatenate
  const totalLength = chunks.reduce((sum, chunk) => sum + chunk.length, 0)
  const result = new Uint8Array(totalLength)
  let offset = 0
  for (const chunk of chunks) {
    result.set(chunk, offset)
    offset += chunk.length
  }

  return result
}

/**
 * Decode a single tensor
 */
function decodeTensor(
  buffer: Uint8Array,
  startOffset: number,
): { name: string; tensor: TensorData; bytesRead: number } {
  let offset = startOffset

  // Name
  const nameLength = decodeUint32(buffer, offset)
  offset += 4
  const nameBytes = buffer.slice(offset, offset + nameLength)
  offset += nameLength
  const name = new TextDecoder().decode(nameBytes)

  // Shape
  const shapeLength = decodeUint32(buffer, offset)
  offset += 4
  const shape: number[] = []
  for (let i = 0; i < shapeLength; i++) {
    shape.push(decodeUint32(buffer, offset))
    offset += 4
  }

  // Dtype
  const dtypeId = buffer[offset]!
  offset += 1
  const dtype = getDtypeName(dtypeId)

  // Data
  const dataLength = decodeUint32(buffer, offset)
  offset += 4
  const dataBytes = buffer.slice(offset, offset + dataLength)
  offset += dataLength

  // Create typed array
  const data = createTypedArray(dtype, dataBytes)

  return {
    name,
    tensor: { data, shape, dtype },
    bytesRead: offset - startOffset,
  }
}

// ==================== Utilities ====================

/**
 * Encode uint32 as little-endian bytes
 */
function encodeUint32(value: number): Uint8Array {
  const buffer = new ArrayBuffer(4)
  new DataView(buffer).setUint32(0, value, true) // little-endian
  return new Uint8Array(buffer)
}

/**
 * Decode uint32 from little-endian bytes
 */
function decodeUint32(buffer: Uint8Array, offset: number): number {
  return new DataView(buffer.buffer, buffer.byteOffset + offset, 4).getUint32(0, true)
}

/**
 * Compare two Uint8Arrays
 */
function arraysEqual(a: Uint8Array, b: Uint8Array): boolean {
  if (a.length !== b.length) return false
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false
  }
  return true
}

/**
 * Get dtype name from ID
 */
function getDtypeName(id: number): DTypeName {
  switch (id) {
    case 0:
      return 'float32'
    case 1:
      return 'float64'
    case 2:
      return 'int32'
    case 3:
      return 'uint8'
    default:
      throw new Error(`Unknown dtype ID: ${id}`)
  }
}

/**
 * Create typed array from bytes
 */
function createTypedArray(
  dtype: DTypeName,
  bytes: Uint8Array,
): Float32Array | Float64Array | Int32Array | Uint8Array {
  // Copy to properly aligned buffer
  const alignedBuffer = new ArrayBuffer(bytes.length)
  new Uint8Array(alignedBuffer).set(bytes)

  switch (dtype) {
    case 'float32':
      return new Float32Array(alignedBuffer)
    case 'float64':
      return new Float64Array(alignedBuffer)
    case 'int32':
      return new Int32Array(alignedBuffer)
    case 'uint8':
      return new Uint8Array(alignedBuffer)
  }
}

// ==================== Convenience Functions ====================

/**
 * Convert Float32Array to TensorData
 */
export function float32Tensor(data: Float32Array, shape: number[]): TensorData {
  return { data, shape, dtype: 'float32' }
}

/**
 * Convert model parameters to checkpoint format
 *
 * @param params - Map of parameter name to data array
 * @param shapes - Map of parameter name to shape
 * @returns Record suitable for checkpoint tensors
 */
export function paramsToTensors(
  params: Map<string, Float32Array>,
  shapes: Map<string, number[]>,
): Record<string, TensorData> {
  const tensors: Record<string, TensorData> = {}

  for (const [name, data] of params) {
    const shape = shapes.get(name) ?? [data.length]
    tensors[name] = { data, shape, dtype: 'float32' }
  }

  return tensors
}
