/**
 * ImageFolder dataset for loading images from a directory structure
 */

import { device, type Tensor } from '@ts-torch/core'
import { readdir, readFile } from 'fs/promises'
import { join, extname, resolve } from 'path'
import jpeg from 'jpeg-js'
import { PNG } from 'pngjs'
import { BaseDataset } from '../dataset.js'
import type { Transform } from '../transforms.js'

const cpu = device.cpu()

/**
 * ImageFolder dataset
 *
 * Loads images from a directory where subdirectories represent classes.
 *
 * Expected structure:
 * ```
 * root/
 *   dog/
 *     xxx.png
 *     xxy.png
 *   cat/
 *     123.png
 *     nsdf3.png
 * ```
 */
export class ImageFolder extends BaseDataset<[Tensor, number]> {
  private samples: Array<[string, number]> = []
  private classToIdx: Map<string, number> = new Map()
  private classes: string[] = []
  private root: string
  private transform: Transform<Tensor, Tensor> | undefined
  private extensions: string[]

  constructor(
    root: string,
    transform?: Transform<Tensor, Tensor>,
    extensions: string[] = ['.jpg', '.jpeg', '.png'],
  ) {
    super()
    this.root = resolve(root)
    this.transform = transform
    this.extensions = extensions.map((ext) => ext.toLowerCase())
  }

  async init(): Promise<void> {
    const entries = await readdir(this.root, { withFileTypes: true })
    const classDirs = entries.filter((entry) => entry.isDirectory()).map((entry) => entry.name).sort()

    this.samples = []
    this.classToIdx = new Map()
    this.classes = []

    for (const [idx, className] of classDirs.entries()) {
      this.classToIdx.set(className, idx)
      this.classes.push(className)

      const classDir = join(this.root, className)
      const files = await readdir(classDir, { withFileTypes: true })
      for (const file of files) {
        if (!file.isFile()) {
          continue
        }
        const ext = extname(file.name).toLowerCase()
        if (this.extensions.includes(ext)) {
          this.samples.push([join(classDir, file.name), idx])
        }
      }
    }
  }

  async getItem(index: number): Promise<[Tensor, number]> {
    if (index < 0 || index >= this.length) {
      throw new Error(`Index ${index} out of bounds`)
    }

    const [path, label] = this.samples[index]!
    let image = await loadImageTensor(path)
    if (this.transform) {
      image = await Promise.resolve(this.transform.apply(image))
    }
    return [image, label]
  }

  get length(): number {
    return this.samples.length
  }

  getClasses(): string[] {
    return this.classes
  }

  getClassToIdx(): Map<string, number> {
    return this.classToIdx
  }
}

async function loadImageTensor(path: string): Promise<Tensor> {
  const buffer = await readFile(path)
  const ext = extname(path).toLowerCase()
  if (ext === '.png') {
    const png = PNG.sync.read(buffer)
    return imageDataToTensor(png.data, png.width, png.height)
  }
  if (ext === '.jpg' || ext === '.jpeg') {
    const decoded = jpeg.decode(buffer, { useTArray: true })
    if (!decoded.data) {
      throw new Error(`Failed to decode JPEG image: ${path}`)
    }
    return imageDataToTensor(decoded.data, decoded.width, decoded.height)
  }
  throw new Error(`Unsupported image extension: ${ext}`)
}

function imageDataToTensor(data: Uint8Array, width: number, height: number): Tensor {
  const channels = 3
  const out = new Float32Array(width * height * channels)
  for (let i = 0; i < width * height; i++) {
    const base = i * 4
    const r = data[base] ?? 0
    const g = data[base + 1] ?? 0
    const b = data[base + 2] ?? 0
    const outIdx = i * channels
    out[outIdx] = r / 255
    out[outIdx + 1] = g / 255
    out[outIdx + 2] = b / 255
  }
  return cpu.tensor(out, [channels, height, width] as const)
}
