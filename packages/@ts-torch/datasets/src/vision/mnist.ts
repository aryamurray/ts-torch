/**
 * MNIST dataset loader
 *
 * Loads MNIST handwritten digit images from IDX binary files.
 */

import { readFileSync } from "fs";
import { join } from "path";
import { torch, type Tensor, type DType } from "@ts-torch/core";

/**
 * MNIST sample containing image tensor and label
 */
export interface MNISTSample {
  /** 28x28 grayscale image as flattened 784-dim tensor, normalized to [0,1] */
  image: Tensor<readonly [784], DType<"float32">>;
  /** Digit label 0-9 */
  label: number;
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
  private images: Float32Array | null = null;
  private labels: Uint8Array | null = null;
  private numSamples = 0;

  constructor(
    private root: string,
    private train: boolean = true,
  ) {}

  /**
   * Load MNIST data from IDX files
   */
  async load(): Promise<void> {
    const prefix = this.train ? "train" : "t10k";

    const imagesPath = join(this.root, `${prefix}-images.idx3-ubyte`);
    const labelsPath = join(this.root, `${prefix}-labels.idx1-ubyte`);

    // Load images
    const imagesBuffer = readFileSync(imagesPath);
    const imagesView = new DataView(imagesBuffer.buffer, imagesBuffer.byteOffset);

    // Parse IDX3 header: magic(4), numImages(4), rows(4), cols(4)
    const magic = imagesView.getUint32(0, false);
    if (magic !== 0x00000803) {
      throw new Error(`Invalid MNIST images magic number: ${magic}`);
    }

    this.numSamples = imagesView.getUint32(4, false);
    const rows = imagesView.getUint32(8, false);
    const cols = imagesView.getUint32(12, false);

    if (rows !== 28 || cols !== 28) {
      throw new Error(`Unexpected MNIST image size: ${rows}x${cols}`);
    }

    // Convert pixel data to normalized Float32 [0, 1]
    const pixelData = new Uint8Array(imagesBuffer.buffer, imagesBuffer.byteOffset + 16);
    this.images = new Float32Array(pixelData.length);
    for (let i = 0; i < pixelData.length; i++) {
      this.images[i] = pixelData[i]! / 255.0;
    }

    // Load labels
    const labelsBuffer = readFileSync(labelsPath);
    const labelsView = new DataView(labelsBuffer.buffer, labelsBuffer.byteOffset);

    // Parse IDX1 header: magic(4), numLabels(4)
    const labelsMagic = labelsView.getUint32(0, false);
    if (labelsMagic !== 0x00000801) {
      throw new Error(`Invalid MNIST labels magic number: ${labelsMagic}`);
    }

    this.labels = new Uint8Array(labelsBuffer.buffer, labelsBuffer.byteOffset + 8);

    console.log(`Loaded MNIST ${this.train ? "train" : "test"}: ${this.numSamples} samples`);
  }

  /**
   * Get number of samples
   */
  get length(): number {
    return this.numSamples;
  }

  /**
   * Get a single sample by index
   */
  get(index: number): MNISTSample {
    if (!this.images || !this.labels) {
      throw new Error("MNIST not loaded. Call load() first.");
    }

    if (index < 0 || index >= this.numSamples) {
      throw new Error(`Index ${index} out of bounds [0, ${this.numSamples})`);
    }

    const start = index * 784;
    const imageData = this.images.slice(start, start + 784);
    const image = torch.tensor(
      Array.from(imageData),
      [784] as const,
    ) as Tensor<readonly [784], DType<"float32">>;

    return {
      image,
      label: this.labels[index]!,
    };
  }

  /**
   * Get a batch of samples
   */
  getBatch(
    startIndex: number,
    batchSize: number,
  ): { images: Tensor<readonly [number, 784], DType<"float32">>; labels: number[] } {
    if (!this.images || !this.labels) {
      throw new Error("MNIST not loaded. Call load() first.");
    }

    const actualBatchSize = Math.min(batchSize, this.numSamples - startIndex);
    const batchData = new Float32Array(actualBatchSize * 784);
    const batchLabels: number[] = [];

    for (let i = 0; i < actualBatchSize; i++) {
      const srcStart = (startIndex + i) * 784;
      batchData.set(this.images.slice(srcStart, srcStart + 784), i * 784);
      batchLabels.push(this.labels[startIndex + i]!);
    }

    const images = torch.tensor(
      Array.from(batchData),
      [actualBatchSize, 784] as const,
    ) as Tensor<readonly [number, 784], DType<"float32">>;

    return { images, labels: batchLabels };
  }

  /**
   * Iterate over the dataset in batches
   */
  *batches(
    batchSize: number,
    shuffle = false,
  ): Generator<{ images: Tensor<readonly [number, 784], DType<"float32">>; labels: number[] }> {
    if (!this.images || !this.labels) {
      throw new Error("MNIST not loaded. Call load() first.");
    }

    // Create index array
    const indices = Array.from({ length: this.numSamples }, (_, i) => i);

    // Shuffle if requested
    if (shuffle) {
      for (let i = indices.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [indices[i], indices[j]] = [indices[j]!, indices[i]!];
      }
    }

    // Yield batches
    for (let start = 0; start < this.numSamples; start += batchSize) {
      const actualBatchSize = Math.min(batchSize, this.numSamples - start);
      const batchData = new Float32Array(actualBatchSize * 784);
      const batchLabels: number[] = [];

      for (let i = 0; i < actualBatchSize; i++) {
        const idx = indices[start + i]!;
        const srcStart = idx * 784;
        batchData.set(this.images!.slice(srcStart, srcStart + 784), i * 784);
        batchLabels.push(this.labels![idx]!);
      }

      const images = torch.tensor(
        Array.from(batchData),
        [actualBatchSize, 784] as const,
      ) as Tensor<readonly [number, 784], DType<"float32">>;

      yield { images, labels: batchLabels };
    }
  }

  /**
   * Class labels
   */
  get classes(): string[] {
    return ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];
  }
}
