/**
 * MNIST dataset loader
 *
 * Loads MNIST handwritten digit images from IDX binary files.
 */

import { readFileSync } from 'fs'
import { resolve, relative } from 'path'
import { device, int64, type Tensor, type DType } from '@ts-torch/core'

// CPU device for tensor creation
const cpu = device.cpu()

/** Maximum number of samples to prevent memory exhaustion */
const MAX_SAMPLES = 1_000_000

/** Minimum header sizes for IDX files */
const IDX3_HEADER_SIZE = 16 // magic(4) + numImages(4) + rows(4) + cols(4)
const IDX1_HEADER_SIZE = 8 // magic(4) + numLabels(4)

/**
 * Validate that a path is within the allowed root directory
 * @throws Error if path traversal is detected
 */
function validatePath(root: string, filename: string): string {
  const resolvedRoot = resolve(root)
  const resolvedPath = resolve(root, filename)
  const relativePath = relative(resolvedRoot, resolvedPath)

  // Check for path traversal (path goes outside root)
  if (relativePath.startsWith('..') || resolve(relativePath) === relativePath) {
    throw new Error(`Path traversal detected: ${filename}`)
  }

  return resolvedPath
}

/**
 * MNIST sample containing image tensor and label
 */
export interface MNISTSample {
  /** 28x28 grayscale image as flattened 784-dim tensor, normalized to [0,1] */
  image: Tensor<readonly [784], DType<'float32'>>
  /** Digit label 0-9 */
  label: number
}

/**
 * MNIST handwritten digits dataset
 *
 * The MNIST database consists of 70,000 28x28 grayscale images of handwritten digits.
 * - Training set: 60,000 images
 * - Test set: 10,000 images
 *
 * @example
 * ```ts
 * const mnist = new MNIST('./data/mnist', true);
 * await mnist.load();
 *
 * // Get a single sample
 * const sample = mnist.get(0);
 * console.log(sample.image.shape); // [784]
 * console.log(sample.label); // 5 (or whatever digit)
 *
 * // Iterate over batches
 * for (const batch of mnist.batches(64)) {
 *   // batch.images: [64, 784]
 *   // batch.labels: number[]
 * }
 * ```
 */
export class MNIST {
  private images: Float32Array | null = null
  private labels: Uint8Array | null = null
  private numSamples = 0

  constructor(
    private root: string,
    private train: boolean = true,
  ) {}

  /**
   * Load MNIST data from IDX files
   */
  async load(): Promise<void> {
    const prefix = this.train ? 'train' : 't10k'

    // Validate paths to prevent path traversal attacks
    const imagesPath = validatePath(this.root, `${prefix}-images.idx3-ubyte`)
    const labelsPath = validatePath(this.root, `${prefix}-labels.idx1-ubyte`)

    // Load and parse images in a block scope to allow buffer release
    {
      let imagesBuffer: Buffer
      try {
        imagesBuffer = readFileSync(imagesPath)
      } catch (error) {
        if (error && typeof error === 'object' && 'code' in error) {
          const code = (error as { code: string }).code
          if (code === 'ENOENT') {
            throw new Error(
              `MNIST images file not found: ${imagesPath}\n` +
              `Please download the MNIST dataset from http://yann.lecun.com/exdb/mnist/\n` +
              `Expected file: ${prefix}-images.idx3-ubyte`
            )
          } else if (code === 'EACCES') {
            throw new Error(
              `Permission denied reading MNIST images file: ${imagesPath}\n` +
              `Please check file permissions.`
            )
          }
        }
        throw new Error(
          `Failed to read MNIST images file: ${imagesPath}\n` +
          `Error: ${error instanceof Error ? error.message : String(error)}`
        )
      }

      // Validate buffer size before creating DataView
      if (imagesBuffer.length < IDX3_HEADER_SIZE) {
        throw new Error(
          `Images file too small: ${imagesBuffer.length} bytes, minimum ${IDX3_HEADER_SIZE} required\n` +
          `File: ${imagesPath}`,
        )
      }

      const imagesView = new DataView(imagesBuffer.buffer, imagesBuffer.byteOffset, imagesBuffer.length)

      // Parse IDX3 header: magic(4), numImages(4), rows(4), cols(4)
      const magic = imagesView.getUint32(0, false)
      if (magic !== 0x00000803) {
        throw new Error(
          `Invalid MNIST images magic number: 0x${magic.toString(16).padStart(8, '0')} (expected 0x00000803)\n` +
          `File may be corrupted or is not a valid MNIST images file: ${imagesPath}`
        )
      }

      this.numSamples = imagesView.getUint32(4, false)

      // Validate numSamples to prevent memory exhaustion
      if (this.numSamples <= 0) {
        throw new Error(`Invalid number of samples: ${this.numSamples}`)
      }
      if (this.numSamples > MAX_SAMPLES) {
        throw new Error(
          `Number of samples ${this.numSamples} exceeds maximum ${MAX_SAMPLES}`,
        )
      }

      const rows = imagesView.getUint32(8, false)
      const cols = imagesView.getUint32(12, false)

      if (rows !== 28 || cols !== 28) {
        throw new Error(
          `Unexpected MNIST image size: ${rows}x${cols} (expected 28x28)\n` +
          `File: ${imagesPath}`
        )
      }

      // Validate file size matches expected data size
      const expectedImageDataSize = this.numSamples * rows * cols
      const actualImageDataSize = imagesBuffer.length - IDX3_HEADER_SIZE
      if (actualImageDataSize < expectedImageDataSize) {
        throw new Error(
          `Images file size mismatch: expected ${expectedImageDataSize} bytes of pixel data, ` +
            `but file only contains ${actualImageDataSize} bytes after header\n` +
            `File may be truncated or corrupted: ${imagesPath}`,
        )
      }

      // Convert pixel data to normalized Float32 [0, 1]
      // Create a copy of the data we need, then let imagesBuffer be garbage collected
      const pixelData = new Uint8Array(
        imagesBuffer.buffer,
        imagesBuffer.byteOffset + IDX3_HEADER_SIZE,
        expectedImageDataSize,
      )
      this.images = new Float32Array(expectedImageDataSize)
      for (let i = 0; i < expectedImageDataSize; i++) {
        this.images[i] = pixelData[i]! / 255.0
      }
      // imagesBuffer goes out of scope here and can be garbage collected
    }

    // Load and parse labels in a block scope to allow buffer release
    {
      let labelsBuffer: Buffer
      try {
        labelsBuffer = readFileSync(labelsPath)
      } catch (error) {
        if (error && typeof error === 'object' && 'code' in error) {
          const code = (error as { code: string }).code
          if (code === 'ENOENT') {
            throw new Error(
              `MNIST labels file not found: ${labelsPath}\n` +
              `Please download the MNIST dataset from http://yann.lecun.com/exdb/mnist/\n` +
              `Expected file: ${prefix}-labels.idx1-ubyte`
            )
          } else if (code === 'EACCES') {
            throw new Error(
              `Permission denied reading MNIST labels file: ${labelsPath}\n` +
              `Please check file permissions.`
            )
          }
        }
        throw new Error(
          `Failed to read MNIST labels file: ${labelsPath}\n` +
          `Error: ${error instanceof Error ? error.message : String(error)}`
        )
      }

      // Validate buffer size before creating DataView
      if (labelsBuffer.length < IDX1_HEADER_SIZE) {
        throw new Error(
          `Labels file too small: ${labelsBuffer.length} bytes, minimum ${IDX1_HEADER_SIZE} required\n` +
          `File: ${labelsPath}`,
        )
      }

      const labelsView = new DataView(labelsBuffer.buffer, labelsBuffer.byteOffset, labelsBuffer.length)

      // Parse IDX1 header: magic(4), numLabels(4)
      const labelsMagic = labelsView.getUint32(0, false)
      if (labelsMagic !== 0x00000801) {
        throw new Error(
          `Invalid MNIST labels magic number: 0x${labelsMagic.toString(16).padStart(8, '0')} (expected 0x00000801)\n` +
          `File may be corrupted or is not a valid MNIST labels file: ${labelsPath}`
        )
      }

      const numLabels = labelsView.getUint32(4, false)

      // Validate labels count matches images count
      if (numLabels !== this.numSamples) {
        throw new Error(
          `Label count ${numLabels} does not match image count ${this.numSamples}`,
        )
      }

      // Validate file size matches expected data size
      const expectedLabelsDataSize = this.numSamples
      const actualLabelsDataSize = labelsBuffer.length - IDX1_HEADER_SIZE
      if (actualLabelsDataSize < expectedLabelsDataSize) {
        throw new Error(
          `Labels file size mismatch: expected ${expectedLabelsDataSize} bytes of label data, ` +
            `but file only contains ${actualLabelsDataSize} bytes after header\n` +
            `File may be truncated or corrupted: ${labelsPath}`,
        )
      }

      // Create a copy of the labels data (not a view) to allow labelsBuffer to be released
      this.labels = new Uint8Array(expectedLabelsDataSize)
      this.labels.set(
        new Uint8Array(
          labelsBuffer.buffer,
          labelsBuffer.byteOffset + IDX1_HEADER_SIZE,
          expectedLabelsDataSize,
        ),
      )
      // labelsBuffer goes out of scope here and can be garbage collected
    }

    console.log(`Loaded MNIST ${this.train ? 'train' : 'test'}: ${this.numSamples} samples`)
  }

  /**
   * Get number of samples
   */
  get length(): number {
    return this.numSamples
  }

  /**
   * Get a single sample by index
   */
  get(index: number): MNISTSample {
    if (!this.images || !this.labels) {
      throw new Error('MNIST not loaded. Call load() first.')
    }

    if (index < 0 || index >= this.numSamples) {
      throw new Error(`Index ${index} out of bounds [0, ${this.numSamples})`)
    }

    const start = index * 784
    const imageData = this.images.slice(start, start + 784)
    const image = cpu.tensor(imageData, [784] as const) as Tensor<readonly [784], DType<'float32'>>

    const label = this.labels[index]
    if (label === undefined) {
      throw new Error(`Label at index ${index} is undefined. Data corruption detected.`)
    }

    return {
      image,
      label,
    }
  }

  /**
   * Get a batch of samples
   */
  getBatch(
    startIndex: number,
    batchSize: number,
  ): { images: Tensor<readonly [number, 784], DType<'float32'>>; labels: number[] } {
    if (!this.images || !this.labels) {
      throw new Error('MNIST not loaded. Call load() first.')
    }

    const actualBatchSize = Math.min(batchSize, this.numSamples - startIndex)
    const batchData = new Float32Array(actualBatchSize * 784)
    const batchLabels: number[] = []

    for (let i = 0; i < actualBatchSize; i++) {
      const srcStart = (startIndex + i) * 784
      batchData.set(this.images.slice(srcStart, srcStart + 784), i * 784)
      const label = this.labels[startIndex + i]
      if (label === undefined) {
        throw new Error(`Label at index ${startIndex + i} is undefined. Data corruption detected.`)
      }
      batchLabels.push(label)
    }

    const images = cpu.tensor(batchData, [actualBatchSize, 784] as const) as Tensor<
      readonly [number, 784],
      DType<'float32'>
    >

    return { images, labels: batchLabels }
  }

  /**
   * Iterate over the dataset in batches
   */
  *batches(
    batchSize: number,
    shuffle = false,
  ): Generator<{
    images: Tensor<readonly [number, 784], DType<'float32'>>
    labels: number[]
    labelsTensor: Tensor<readonly [number], DType<'int64'>>
  }> {
    if (!this.images || !this.labels) {
      throw new Error('MNIST not loaded. Call load() first.')
    }

    // Create index array
    const indices = Array.from({ length: this.numSamples }, (_, i) => i)

    // Shuffle if requested
    if (shuffle) {
      for (let i = indices.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1))
        const idxI = indices[i]
        const idxJ = indices[j]
        if (idxI === undefined || idxJ === undefined) {
          throw new Error(`Indices array corruption detected during shuffle at positions ${i}, ${j}`)
        }
        ;[indices[i], indices[j]] = [idxJ, idxI]
      }
    }

    // Yield batches
    for (let start = 0; start < this.numSamples; start += batchSize) {
      const actualBatchSize = Math.min(batchSize, this.numSamples - start)
      const batchData = new Float32Array(actualBatchSize * 784)
      const batchLabels: number[] = []

      for (let i = 0; i < actualBatchSize; i++) {
        const idx = indices[start + i]
        if (idx === undefined) {
          throw new Error(`Index at position ${start + i} is undefined. Data corruption detected.`)
        }
        const srcStart = idx * 784
        batchData.set(this.images.slice(srcStart, srcStart + 784), i * 784)
        const label = this.labels[idx]
        if (label === undefined) {
          throw new Error(`Label at index ${idx} is undefined. Data corruption detected.`)
        }
        batchLabels.push(label)
      }

      const images = cpu.tensor(batchData, [actualBatchSize, 784] as const) as Tensor<
        readonly [number, 784],
        DType<'float32'>
      >

      // Create labels tensor as int64 for cross_entropy_loss
      const labelsTensor = cpu.tensor(batchLabels, [actualBatchSize] as const, int64) as Tensor<
        readonly [number],
        DType<'int64'>
      >

      yield { images, labels: batchLabels, labelsTensor }
    }
  }

  /**
   * Get a batch of samples by arbitrary indices (for shuffled batching)
   *
   * This is more efficient than calling get() N times because it:
   * - Creates a single Float32Array for all images
   * - Creates a single int64 tensor for all labels
   * - Avoids N tensor allocations
   *
   * @param indices - Array of sample indices to include in batch
   */
  getBatchByIndices(indices: number[]): {
    images: Tensor<readonly [number, 784], DType<'float32'>>
    labels: number[]
    labelsTensor: Tensor<readonly [number], DType<'int64'>>
  } {
    if (!this.images || !this.labels) {
      throw new Error('MNIST not loaded. Call load() first.')
    }

    const batchSize = indices.length
    const batchData = new Float32Array(batchSize * 784)
    const batchLabels: number[] = []

    for (let i = 0; i < batchSize; i++) {
      const idx = indices[i]
      if (idx === undefined || idx < 0 || idx >= this.numSamples) {
        throw new Error(`Index ${idx} out of bounds [0, ${this.numSamples})`)
      }
      const srcStart = idx * 784
      batchData.set(this.images.slice(srcStart, srcStart + 784), i * 784)
      const label = this.labels[idx]
      if (label === undefined) {
        throw new Error(`Label at index ${idx} is undefined. Data corruption detected.`)
      }
      batchLabels.push(label)
    }

    const images = cpu.tensor(batchData, [batchSize, 784] as const) as Tensor<
      readonly [number, 784],
      DType<'float32'>
    >

    // Create labels tensor as int64 for cross_entropy_loss
    const labelsTensor = cpu.tensor(batchLabels, [batchSize] as const, int64) as Tensor<
      readonly [number],
      DType<'int64'>
    >

    return { images, labels: batchLabels, labelsTensor }
  }

  /**
   * Class labels
   */
  get classes(): string[] {
    return ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
  }
}
