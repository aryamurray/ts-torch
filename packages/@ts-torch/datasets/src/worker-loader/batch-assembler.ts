/**
 * Batch assembler - converts raw buffer data to tensors
 */

import { device, int64, float32, type Tensor } from '@ts-torch/core'
import type { BatchMetadata, DType, ReadySlot, TensorPair } from './types.js'

const cpu = device.cpu()

/**
 * Assembles batch data from a SharedArrayBuffer slot into tensors
 */
export class BatchAssembler {
  /**
   * Assemble a batch from a ready slot
   */
  assemble(slot: ReadySlot): TensorPair {
    const { buffer, metadata } = slot

    const data = this.readData(buffer, metadata)
    const label = this.readLabel(buffer, metadata)

    return { data, label }
  }

  /**
   * Read data tensor from buffer
   */
  private readData(buffer: SharedArrayBuffer, metadata: BatchMetadata): Tensor {
    const { dataByteOffset, dataShape, dataDtype } = metadata

    const elements = dataShape.reduce((a, b) => a * b, 1)

    // Create typed array view
    const typedArray = this.createTypedArray(buffer, dataByteOffset, elements, dataDtype)

    // Copy to regular array (tensor creation may take ownership)
    const data = this.typedArrayToNumberArray(typedArray, dataDtype)

    // Create tensor - use type assertion since we handle dtype dynamically
    return cpu.tensor(data, dataShape as readonly number[], this.dtypeToTorchDtype(dataDtype)) as Tensor
  }

  /**
   * Read label tensor from buffer
   */
  private readLabel(buffer: SharedArrayBuffer, metadata: BatchMetadata): Tensor {
    const { labelByteOffset, labelShape, labelDtype } = metadata

    const elements = labelShape.reduce((a, b) => a * b, 1)

    // Create typed array view
    const typedArray = this.createTypedArray(buffer, labelByteOffset, elements, labelDtype)

    // Copy to regular array
    const data = this.typedArrayToNumberArray(typedArray, labelDtype)

    // Create tensor - use type assertion since we handle dtype dynamically
    return cpu.tensor(data, labelShape as readonly number[], this.dtypeToTorchDtype(labelDtype)) as Tensor
  }

  /**
   * Create a typed array view into the buffer
   */
  private createTypedArray(
    buffer: SharedArrayBuffer,
    byteOffset: number,
    elements: number,
    dtype: DType,
  ): Float32Array | Float64Array | Int32Array | BigInt64Array | Uint8Array {
    switch (dtype) {
      case 'float32':
        return new Float32Array(buffer, byteOffset, elements)
      case 'float64':
        return new Float64Array(buffer, byteOffset, elements)
      case 'int32':
        return new Int32Array(buffer, byteOffset, elements)
      case 'int64':
        return new BigInt64Array(buffer, byteOffset, elements)
      case 'uint8':
        return new Uint8Array(buffer, byteOffset, elements)
      default:
        throw new Error(`Unsupported dtype: ${dtype}`)
    }
  }

  /**
   * Convert typed array to number array (copying data)
   */
  private typedArrayToNumberArray(
    typedArray: Float32Array | Float64Array | Int32Array | BigInt64Array | Uint8Array,
    dtype: DType,
  ): number[] {
    if (dtype === 'int64') {
      // BigInt64Array needs special handling
      return Array.from(typedArray as BigInt64Array).map(Number)
    }
    return Array.from(typedArray as Float32Array | Float64Array | Int32Array | Uint8Array)
  }

  /**
   * Convert our DType to torch dtype
   */
  private dtypeToTorchDtype(dtype: DType): typeof float32 | typeof int64 {
    switch (dtype) {
      case 'float32':
        return float32
      case 'float64':
        // For now, use float32 - could add float64 support
        return float32
      case 'int32':
      case 'int64':
        return int64
      case 'uint8':
        // uint8 data typically gets cast to float32
        return float32
      default:
        return float32
    }
  }
}

/**
 * Create a batch assembler instance
 */
export function createBatchAssembler(): BatchAssembler {
  return new BatchAssembler()
}
