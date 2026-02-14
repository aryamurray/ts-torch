/**
 * Safetensors Format Reader & Writer
 *
 * Implements the HuggingFace safetensors binary format for tensor serialization.
 * Format: [8 bytes: uint64 LE header_size][header_size bytes: JSON header][remaining: raw tensor data]
 *
 * @see https://huggingface.co/docs/safetensors/
 */

import { readFile, writeFile } from 'node:fs/promises'

// ==================== Types ====================

/**
 * Tensor data for serialization
 */
export interface TensorData {
  /** Raw data */
  data: Float32Array | Float64Array | Int32Array | Uint8Array | BigInt64Array | Uint16Array
  /** Tensor shape */
  shape: number[]
  /** Data type */
  dtype: string
}

/**
 * State dict type - maps parameter names to tensor data
 */
export type StateDict = Record<string, TensorData>

// ==================== Dtype Mapping ====================

/** Safetensors dtype string -> ts-torch dtype string */
const SF_TO_TSTORCH: Record<string, string> = {
  F16: 'float16',
  BF16: 'bfloat16',
  F32: 'float32',
  F64: 'float64',
  I32: 'int32',
  I64: 'int64',
  BOOL: 'bool',
  U8: 'uint8',
}

/** ts-torch dtype string -> safetensors dtype string */
const TSTORCH_TO_SF: Record<string, string> = {
  float16: 'F16',
  bfloat16: 'BF16',
  float32: 'F32',
  float64: 'F64',
  int32: 'I32',
  int64: 'I64',
  bool: 'BOOL',
  uint8: 'U8',
}

// ==================== Internal Types ====================

interface SafetensorsHeader {
  [key: string]: {
    dtype: string
    shape: number[]
    data_offsets: [number, number]
  } | Record<string, string>
}

// ==================== Decoder ====================

/**
 * Decode a safetensors buffer into tensors and metadata.
 *
 * @param buffer - Raw bytes of a .safetensors file
 * @returns Object with tensors (StateDict) and metadata (Record<string, string>)
 */
export function decodeSafetensors(buffer: Uint8Array): { tensors: StateDict; metadata: Record<string, string> } {
  if (buffer.byteLength < 8) {
    throw new Error('Invalid safetensors file: too small')
  }

  // Read header size (uint64 LE)
  const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength)
  const headerSize = Number(view.getBigUint64(0, true))

  if (8 + headerSize > buffer.byteLength) {
    throw new Error('Invalid safetensors file: header size exceeds buffer')
  }

  // Parse header JSON
  const headerBytes = buffer.slice(8, 8 + headerSize)
  const headerJson = new TextDecoder().decode(headerBytes)
  const header: SafetensorsHeader = JSON.parse(headerJson)

  // Data starts after the header
  const dataStart = 8 + headerSize

  const state: StateDict = {}
  let metadata: Record<string, string> = {}

  for (const [key, entry] of Object.entries(header)) {
    // Extract metadata entry
    if (key === '__metadata__') {
      metadata = entry as Record<string, string>
      continue
    }

    const tensorMeta = entry as { dtype: string; shape: number[]; data_offsets: [number, number] }

    const tsTorchDtype = SF_TO_TSTORCH[tensorMeta.dtype]
    if (!tsTorchDtype) {
      throw new Error(`Unknown safetensors dtype: ${tensorMeta.dtype}`)
    }

    const [startOffset, endOffset] = tensorMeta.data_offsets
    const rawBytes = buffer.slice(dataStart + startOffset, dataStart + endOffset)

    // Create properly typed array from raw bytes
    const data = createTypedArrayFromBytes(tsTorchDtype, rawBytes)

    state[key] = {
      data,
      shape: tensorMeta.shape,
      dtype: tsTorchDtype,
    }
  }

  return { tensors: state, metadata }
}

/**
 * Load a safetensors file from disk.
 *
 * @param path - Path to .safetensors file
 * @returns Object with tensors (StateDict) and metadata
 */
export async function loadSafetensors(path: string): Promise<{ tensors: StateDict; metadata: Record<string, string> }> {
  const buffer = await readFile(path)
  return decodeSafetensors(new Uint8Array(buffer))
}

// ==================== Encoder ====================

/**
 * Encode a StateDict into safetensors binary format.
 *
 * @param tensors - StateDict to encode
 * @param metadata - Optional string metadata to include
 * @returns Uint8Array of the safetensors file
 */
export function encodeSafetensors(
  tensors: StateDict,
  metadata?: Record<string, string>,
): Uint8Array {
  // Sort keys for deterministic output
  const keys = Object.keys(tensors).sort()

  // Build header and compute data offsets
  const header: Record<string, unknown> = {}
  let currentOffset = 0

  // Pre-compute all tensor byte representations
  const tensorBytes: Uint8Array[] = []

  for (const key of keys) {
    const tensor = tensors[key]!
    const sfDtype = TSTORCH_TO_SF[tensor.dtype]
    if (!sfDtype) {
      throw new Error(`Cannot encode dtype "${tensor.dtype}" to safetensors`)
    }

    const rawBytes = new Uint8Array(
      tensor.data.buffer,
      tensor.data.byteOffset,
      tensor.data.byteLength,
    )
    tensorBytes.push(rawBytes)

    const startOffset = currentOffset
    currentOffset += rawBytes.byteLength

    header[key] = {
      dtype: sfDtype,
      shape: tensor.shape,
      data_offsets: [startOffset, currentOffset],
    }
  }

  // Add metadata if provided
  if (metadata && Object.keys(metadata).length > 0) {
    header.__metadata__ = metadata
  }

  // Encode header as JSON
  const headerJson = JSON.stringify(header)
  const headerBytes = new TextEncoder().encode(headerJson)

  // Build final buffer: [8 bytes header_size][header JSON][tensor data...]
  const totalSize = 8 + headerBytes.byteLength + currentOffset
  const result = new Uint8Array(totalSize)
  const resultView = new DataView(result.buffer)

  // Write header size as uint64 LE
  resultView.setBigUint64(0, BigInt(headerBytes.byteLength), true)

  // Write header JSON
  result.set(headerBytes, 8)

  // Write tensor data
  let dataOffset = 8 + headerBytes.byteLength
  for (const bytes of tensorBytes) {
    result.set(bytes, dataOffset)
    dataOffset += bytes.byteLength
  }

  return result
}

/**
 * Save a StateDict to a safetensors file.
 *
 * @param path - Output file path
 * @param tensors - StateDict to save
 * @param metadata - Optional metadata
 */
export async function saveSafetensors(
  path: string,
  tensors: StateDict,
  metadata?: Record<string, string>,
): Promise<void> {
  const buffer = encodeSafetensors(tensors, metadata)
  await writeFile(path, buffer)
}

// ==================== Metadata Helpers ====================

/**
 * Serialize user metadata for safetensors __metadata__ header.
 * Always adds `framework: "ts-torch"`. JSON-stringifies non-string values.
 *
 * @param meta - User metadata (epoch, loss, etc.)
 * @returns String-valued record suitable for safetensors metadata
 */
export function serializeMetadata(meta?: Record<string, unknown>): Record<string, string> {
  const result: Record<string, string> = { framework: 'ts-torch' }
  if (!meta) return result

  for (const [key, value] of Object.entries(meta)) {
    if (key === 'framework') continue // don't let user override framework tag
    result[key] = typeof value === 'string' ? value : JSON.stringify(value)
  }

  return result
}

/**
 * Deserialize safetensors metadata back to user-friendly types.
 * JSON-parses values that look like JSON, passes through plain strings.
 * Strips the internal `framework` key.
 *
 * @param meta - Raw string metadata from safetensors header
 * @returns Parsed metadata record
 */
export function deserializeMetadata(meta: Record<string, string>): Record<string, unknown> {
  const result: Record<string, unknown> = {}

  for (const [key, value] of Object.entries(meta)) {
    if (key === 'framework') continue // internal tag, don't expose

    // Try to JSON-parse values that look like JSON
    try {
      const parsed = JSON.parse(value)
      // Only use parsed result if it's not a plain string (those go through as-is)
      if (typeof parsed !== 'string') {
        result[key] = parsed
        continue
      }
    } catch {
      // Not JSON, fall through
    }

    result[key] = value
  }

  return result
}

// ==================== Utilities ====================

function createTypedArrayFromBytes(
  dtype: string,
  bytes: Uint8Array,
): TensorData['data'] {
  // Copy to aligned buffer
  const aligned = new ArrayBuffer(bytes.byteLength)
  new Uint8Array(aligned).set(bytes)

  switch (dtype) {
    case 'float32':
      return new Float32Array(aligned)
    case 'float64':
      return new Float64Array(aligned)
    case 'int32':
      return new Int32Array(aligned)
    case 'int64':
      return new BigInt64Array(aligned)
    case 'float16':
    case 'bfloat16':
      return new Uint16Array(aligned)
    case 'bool':
    case 'uint8':
      return new Uint8Array(aligned)
    default:
      throw new Error(`Unknown dtype: ${dtype}`)
  }
}
