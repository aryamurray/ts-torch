/**
 * CIFAR-10 and CIFAR-100 datasets
 */

import { device, type Tensor } from '@ts-torch/core'
import { readFileSync } from 'fs'
import { join, resolve } from 'path'
import { BaseDataset } from '../dataset.js'
import type { Transform } from '../transforms.js'

const cpu = device.cpu()

const CIFAR_WIDTH = 32
const CIFAR_HEIGHT = 32
const CIFAR_CHANNELS = 3
const CIFAR_RECORD_SIZE = 1 + CIFAR_WIDTH * CIFAR_HEIGHT * CIFAR_CHANNELS
const CIFAR100_RECORD_SIZE = 2 + CIFAR_WIDTH * CIFAR_HEIGHT * CIFAR_CHANNELS

/**
 * CIFAR-10 dataset
 *
 * The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes.
 * - Training set: 50,000 images
 * - Test set: 10,000 images
 */
export class CIFAR10 extends BaseDataset<[Tensor, number]> {
  private data: Float32Array | null = null
  private labels: Uint8Array | null = null
  private root: string
  private transform: Transform<Tensor, Tensor> | undefined
  private download: boolean

  constructor(
    root: string,
    private train: boolean = true,
    transform?: Transform<Tensor, Tensor>,
    download: boolean = false,
  ) {
    super()
    this.root = resolve(root)
    this.transform = transform
    this.download = download
  }

  async init(): Promise<void> {
    const baseDir = join(this.root, 'cifar-10-batches-bin')
    const files = this.train
      ? ['data_batch_1.bin', 'data_batch_2.bin', 'data_batch_3.bin', 'data_batch_4.bin', 'data_batch_5.bin']
      : ['test_batch.bin']

    const buffers: Buffer[] = []
    try {
      for (const file of files) {
        buffers.push(readFileSync(join(baseDir, file)))
      }
    } catch (error) {
      if (this.download) {
        throw new Error('CIFAR-10 download is not implemented. Please download the dataset manually.')
      }
      throw new Error(`Failed to read CIFAR-10 data files in ${baseDir}: ${String(error)}`)
    }

    const totalRecords = buffers.reduce((acc, buffer) => acc + Math.floor(buffer.length / CIFAR_RECORD_SIZE), 0)
    this.data = new Float32Array(totalRecords * CIFAR_CHANNELS * CIFAR_WIDTH * CIFAR_HEIGHT)
    this.labels = new Uint8Array(totalRecords)

    let offset = 0
    let labelOffset = 0
    for (const buffer of buffers) {
      const records = Math.floor(buffer.length / CIFAR_RECORD_SIZE)
      for (let i = 0; i < records; i++) {
        const recordStart = i * CIFAR_RECORD_SIZE
        const label = buffer[recordStart] ?? 0
        this.labels[labelOffset++] = label
        for (let j = 0; j < CIFAR_WIDTH * CIFAR_HEIGHT * CIFAR_CHANNELS; j++) {
          const value = buffer[recordStart + 1 + j] ?? 0
          this.data[offset++] = value / 255
        }
      }
    }
  }

  getItem(index: number): [Tensor, number] {
    if (!this.data || !this.labels) {
      throw new Error('CIFAR10 dataset not initialized. Call init() first.')
    }

    if (index < 0 || index >= this.length) {
      throw new Error(`Index ${index} out of bounds`)
    }

    const imageSize = CIFAR_WIDTH * CIFAR_HEIGHT * CIFAR_CHANNELS
    const start = index * imageSize
    const imageData = this.data.slice(start, start + imageSize)
    let image = cpu.tensor(imageData, [CIFAR_CHANNELS, CIFAR_HEIGHT, CIFAR_WIDTH] as const) as Tensor
    if (this.transform) {
      image = this.transform.apply(image) as Tensor
    }
    const label = this.labels[index] ?? 0
    return [image, label]
  }

  get length(): number {
    if (!this.data) {
      return this.train ? 50000 : 10000
    }
    return Math.floor(this.data.length / (CIFAR_WIDTH * CIFAR_HEIGHT * CIFAR_CHANNELS))
  }

  get classes(): string[] {
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
  }
}

/**
 * CIFAR-100 dataset
 *
 * The CIFAR-100 dataset is similar to CIFAR-10 but has 100 classes.
 * Each class contains 600 images (500 training, 100 test).
 */
export class CIFAR100 extends BaseDataset<[Tensor, number]> {
  private fineLabels: boolean
  private data: Float32Array | null = null
  private labels: Uint8Array | null = null
  private root: string
  private transform: Transform<Tensor, Tensor> | undefined
  private download: boolean

  constructor(
    root: string,
    private train: boolean = true,
    transform?: Transform<Tensor, Tensor>,
    download: boolean = false,
    fineLabels: boolean = true,
  ) {
    super()
    this.fineLabels = fineLabels
    this.root = resolve(root)
    this.transform = transform
    this.download = download
  }

  async init(): Promise<void> {
    const baseDir = join(this.root, 'cifar-100-binary')
    const file = this.train ? 'train.bin' : 'test.bin'

    let buffer: Buffer
    try {
      buffer = readFileSync(join(baseDir, file))
    } catch (error) {
      if (this.download) {
        throw new Error('CIFAR-100 download is not implemented. Please download the dataset manually.')
      }
      throw new Error(`Failed to read CIFAR-100 data file in ${baseDir}: ${String(error)}`)
    }

    const records = Math.floor(buffer.length / CIFAR100_RECORD_SIZE)
    this.data = new Float32Array(records * CIFAR_CHANNELS * CIFAR_WIDTH * CIFAR_HEIGHT)
    this.labels = new Uint8Array(records)

    let offset = 0
    for (let i = 0; i < records; i++) {
      const recordStart = i * CIFAR100_RECORD_SIZE
      const coarse = buffer[recordStart] ?? 0
      const fine = buffer[recordStart + 1] ?? 0
      this.labels[i] = this.fineLabels ? fine : coarse
      for (let j = 0; j < CIFAR_WIDTH * CIFAR_HEIGHT * CIFAR_CHANNELS; j++) {
        const value = buffer[recordStart + 2 + j] ?? 0
        this.data[offset++] = value / 255
      }
    }
  }

  getItem(index: number): [Tensor, number] {
    if (!this.data || !this.labels) {
      throw new Error('CIFAR100 dataset not initialized. Call init() first.')
    }
    if (index < 0 || index >= this.length) {
      throw new Error(`Index ${index} out of bounds`)
    }
    const imageSize = CIFAR_WIDTH * CIFAR_HEIGHT * CIFAR_CHANNELS
    const start = index * imageSize
    const imageData = this.data.slice(start, start + imageSize)
    let image = cpu.tensor(imageData, [CIFAR_CHANNELS, CIFAR_HEIGHT, CIFAR_WIDTH] as const) as Tensor
    if (this.transform) {
      image = this.transform.apply(image) as Tensor
    }
    const label = this.labels[index] ?? 0
    return [image, label]
  }

  get length(): number {
    if (!this.data) {
      return this.train ? 50000 : 10000
    }
    return Math.floor(this.data.length / (CIFAR_WIDTH * CIFAR_HEIGHT * CIFAR_CHANNELS))
  }

  get numClasses(): number {
    return this.fineLabels ? 100 : 20
  }
}
