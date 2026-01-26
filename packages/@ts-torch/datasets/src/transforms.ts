/**
 * Data transformation utilities for image and tensor preprocessing
 */

import { device, type Tensor, type DType } from '@ts-torch/core'

const cpu = device.cpu()

/**
 * Base transform interface
 */
export interface Transform<TInput = Tensor, TOutput = Tensor> {
  apply(input: TInput): TOutput | Promise<TOutput>
}

/**
 * Compose multiple transforms into a pipeline
 *
 * @example
 * ```ts
 * const transform = new Compose([
 *   new Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
 *   new RandomHorizontalFlip(0.5),
 * ])
 *
 * const transformed = await transform.apply(imageTensor)
 * ```
 */
export class Compose<T = Tensor> implements Transform<T, T> {
  constructor(private transforms: Transform<T, T>[]) {}

  async apply(input: T): Promise<T> {
    let result = input
    for (const transform of this.transforms) {
      result = await Promise.resolve(transform.apply(result))
    }
    return result
  }
}

/**
 * Normalize a tensor with mean and standard deviation
 *
 * For each channel: output = (input - mean) / std
 *
 * @example
 * ```ts
 * // ImageNet normalization
 * const normalize = new Normalize(
 *   [0.485, 0.456, 0.406],  // mean per channel
 *   [0.229, 0.224, 0.225]   // std per channel
 * )
 *
 * // Single value for grayscale
 * const normalizeGray = new Normalize(0.5, 0.5)
 * ```
 */
export class Normalize implements Transform<Tensor, Tensor> {
  private mean: number[]
  private std: number[]

  constructor(mean: number | number[], std: number | number[]) {
    this.mean = Array.isArray(mean) ? mean : [mean]
    this.std = Array.isArray(std) ? std : [std]

    // Validate std is not zero
    for (const s of this.std) {
      if (s === 0) {
        throw new Error('Standard deviation cannot be zero')
      }
    }
  }

  apply(tensor: Tensor): Tensor {
    if (this.mean.length === 1 && this.std.length === 1) {
      const centered = (tensor as any).subScalar(this.mean[0])
      return (centered as any).divScalar(this.std[0])
    }

    const shape = tensor.shape as readonly number[]
    const { batch, channels, height, width } = parseImageShape(shape)

    if (channels !== this.mean.length || channels !== this.std.length) {
      const avgMean = this.mean.reduce((a, b) => a + b, 0) / this.mean.length
      const avgStd = this.std.reduce((a, b) => a + b, 0) / this.std.length
      const centered = (tensor as any).subScalar(avgMean)
      return (centered as any).divScalar(avgStd)
    }

    const data = toNumberArray(tensor)
    const out = new Float32Array(data.length)

    for (let n = 0; n < batch; n++) {
      for (let c = 0; c < channels; c++) {
        const mean = this.mean[c]!
        const std = this.std[c]!
        for (let h = 0; h < height; h++) {
          for (let w = 0; w < width; w++) {
            const idx = (((n * channels + c) * height + h) * width + w)
            out[idx] = (data[idx]! - mean) / std
          }
        }
      }
    }

    return cpu.tensor(out, shape as unknown as readonly number[], tensor.dtype as DType<string>) as Tensor
  }
}

/**
 * Lambda transform - apply an arbitrary function to the input
 *
 * @example
 * ```ts
 * const addNoise = new Lambda((t) => t.add(noise))
 * const transformed = addNoise.apply(tensor)
 * ```
 */
export class Lambda<TInput = Tensor, TOutput = Tensor> implements Transform<TInput, TOutput> {
  constructor(private fn: (input: TInput) => TOutput | Promise<TOutput>) {}

  apply(input: TInput): TOutput | Promise<TOutput> {
    return this.fn(input)
  }
}

/**
 * Randomly flip the tensor horizontally with a given probability
 *
 * Assumes tensor shape is [C, H, W] or [N, C, H, W] where W is the width dimension.
 *
 * @example
 * ```ts
 * const flip = new RandomHorizontalFlip(0.5)  // 50% chance to flip
 * const transformed = flip.apply(imageTensor)
 * ```
 */
export class RandomHorizontalFlip implements Transform<Tensor, Tensor> {
  constructor(private p: number = 0.5) {
    if (p < 0 || p > 1) {
      throw new Error('Probability must be between 0 and 1')
    }
  }

  apply(tensor: Tensor): Tensor {
    if (Math.random() < this.p) {
      const shape = tensor.shape as readonly number[]
      const { batch, channels, height, width } = parseImageShape(shape)
      const data = toNumberArray(tensor)
      const out = new Float32Array(data.length)

      for (let n = 0; n < batch; n++) {
        for (let c = 0; c < channels; c++) {
          for (let h = 0; h < height; h++) {
            for (let w = 0; w < width; w++) {
              const src = (((n * channels + c) * height + h) * width + w)
              const dst = (((n * channels + c) * height + h) * width + (width - 1 - w))
              out[dst] = data[src]!
            }
          }
        }
      }

      return cpu.tensor(out, shape as unknown as readonly number[], tensor.dtype as DType<string>) as Tensor
    }
    return tensor
  }
}

/**
 * Randomly crop the tensor to a specified size
 *
 * Assumes tensor shape is [C, H, W] or [N, C, H, W].
 *
 * @example
 * ```ts
 * const crop = new RandomCrop([224, 224])
 * const cropped = crop.apply(imageTensor)  // Random 224x224 crop
 * ```
 */
export class RandomCrop implements Transform<Tensor, Tensor> {
  private height: number
  private width: number

  constructor(size: number | [number, number]) {
    if (typeof size === 'number') {
      this.height = size
      this.width = size
    } else {
      this.height = size[0]
      this.width = size[1]
    }
  }

  apply(tensor: Tensor): Tensor {
    const shape = tensor.shape as readonly number[]
    const ndim = shape.length

    if (ndim < 2) {
      throw new Error('RandomCrop requires at least 2D tensor')
    }

    // Get H and W dimensions (last two)
    const h = shape[ndim - 2]!
    const w = shape[ndim - 1]!

    if (h < this.height || w < this.width) {
      throw new Error(`Crop size (${this.height}, ${this.width}) is larger than tensor size (${h}, ${w})`)
    }

    // Random start positions
    const startH = Math.floor(Math.random() * (h - this.height + 1))
    const startW = Math.floor(Math.random() * (w - this.width + 1))

    // Use narrow to crop
    // First narrow along H dimension, then along W dimension
    const hDim = ndim - 2
    const wDim = ndim - 1

    let result = (tensor as any).narrow(hDim, startH, this.height)
    result = (result as any).narrow(wDim, startW, this.width)

    return result
  }
}

/**
 * Resize the tensor to a specified size
 *
 * Note: Resize requires interpolation which is not yet implemented.
 * This transform currently returns the original tensor.
 *
 * @example
 * ```ts
 * const resize = new Resize([256, 256])
 * const resized = resize.apply(imageTensor)
 * ```
 */
export class Resize implements Transform<Tensor, Tensor> {
  private config: { height: number; width: number; interpolation: 'nearest' | 'bilinear' }

  constructor(size: number | [number, number], interpolation: 'nearest' | 'bilinear' = 'bilinear') {
    const [height, width] = typeof size === 'number' ? [size, size] : size
    this.config = { height, width, interpolation }
  }

  apply(tensor: Tensor): Tensor {
    const { height: targetH, width: targetW } = this.config
    const shape = tensor.shape as readonly number[]
    const { batch, channels, height, width } = parseImageShape(shape)

    const data = toNumberArray(tensor)
    const out = new Float32Array(batch * channels * targetH * targetW)

    for (let n = 0; n < batch; n++) {
      for (let c = 0; c < channels; c++) {
        for (let y = 0; y < targetH; y++) {
          const srcY = Math.min(height - 1, Math.round((y / targetH) * height))
          for (let x = 0; x < targetW; x++) {
            const srcX = Math.min(width - 1, Math.round((x / targetW) * width))
            const src = (((n * channels + c) * height + srcY) * width + srcX)
            const dst = (((n * channels + c) * targetH + y) * targetW + x)
            out[dst] = data[src]!
          }
        }
      }
    }

    const newShape = shape.length === 4 ? [batch, channels, targetH, targetW] : [channels, targetH, targetW]
    return cpu.tensor(out, newShape as unknown as readonly number[], tensor.dtype as DType<string>) as Tensor
  }
}

/**
 * Crop the tensor from the center to a specified size
 *
 * Assumes tensor shape is [C, H, W] or [N, C, H, W].
 *
 * @example
 * ```ts
 * const crop = new CenterCrop([224, 224])
 * const cropped = crop.apply(imageTensor)  // Center 224x224 crop
 * ```
 */
export class CenterCrop implements Transform<Tensor, Tensor> {
  private height: number
  private width: number

  constructor(size: number | [number, number]) {
    if (typeof size === 'number') {
      this.height = size
      this.width = size
    } else {
      this.height = size[0]
      this.width = size[1]
    }
  }

  apply(tensor: Tensor): Tensor {
    const shape = tensor.shape as readonly number[]
    const ndim = shape.length

    if (ndim < 2) {
      throw new Error('CenterCrop requires at least 2D tensor')
    }

    // Get H and W dimensions (last two)
    const h = shape[ndim - 2]!
    const w = shape[ndim - 1]!

    if (h < this.height || w < this.width) {
      throw new Error(`Crop size (${this.height}, ${this.width}) is larger than tensor size (${h}, ${w})`)
    }

    // Center start positions
    const startH = Math.floor((h - this.height) / 2)
    const startW = Math.floor((w - this.width) / 2)

    // Use narrow to crop
    const hDim = ndim - 2
    const wDim = ndim - 1

    let result = (tensor as any).narrow(hDim, startH, this.height)
    result = (result as any).narrow(wDim, startW, this.width)

    return result
  }
}

/**
 * Convert a JavaScript array to a Tensor
 *
 * @example
 * ```ts
 * const toTensor = new ToTensor()
 * const tensor = toTensor.apply([[1, 2, 3], [4, 5, 6]])
 * ```
 */
export class ToTensor implements Transform<number[] | number[][] | number[][][], Tensor> {
  apply(_data: number[] | number[][] | number[][][]): Tensor {
    const { flat, shape } = flattenNested(_data)
    return cpu.tensor(flat, shape as unknown as readonly number[]) as Tensor
  }
}

/**
 * Pad a tensor with a constant value
 *
 * @example
 * ```ts
 * const pad = new Pad([1, 1, 2, 2])  // left, right, top, bottom
 * const padded = pad.apply(imageTensor)
 * ```
 */
export class Pad implements Transform<Tensor, Tensor> {
  private config: { padding: [number, number, number, number]; value: number }

  constructor(padding: number | [number, number] | [number, number, number, number], value: number = 0) {
    let normalizedPadding: [number, number, number, number]
    if (typeof padding === 'number') {
      normalizedPadding = [padding, padding, padding, padding]
    } else if (padding.length === 2) {
      // [horizontal, vertical]
      normalizedPadding = [padding[0], padding[0], padding[1], padding[1]]
    } else {
      normalizedPadding = padding
    }
    this.config = { padding: normalizedPadding, value }
  }

  apply(tensor: Tensor): Tensor {
    const shape = tensor.shape as readonly number[]
    const { padding, value } = this.config
    const [padLeft, padRight, padTop, padBottom] = padding
    const { batch, channels, height, width } = parseImageShape(shape)

    const outH = height + padTop + padBottom
    const outW = width + padLeft + padRight
    const out = new Float32Array(batch * channels * outH * outW)
    out.fill(value)

    const data = toNumberArray(tensor)
    for (let n = 0; n < batch; n++) {
      for (let c = 0; c < channels; c++) {
        for (let h = 0; h < height; h++) {
          for (let w = 0; w < width; w++) {
            const src = (((n * channels + c) * height + h) * width + w)
            const dst = (((n * channels + c) * outH + (h + padTop)) * outW + (w + padLeft))
            out[dst] = data[src]!
          }
        }
      }
    }

    const newShape = shape.length === 4 ? [batch, channels, outH, outW] : [channels, outH, outW]
    return cpu.tensor(out, newShape as unknown as readonly number[], tensor.dtype as DType<string>) as Tensor
  }
}

/**
 * Random erasing augmentation
 *
 * Randomly erases a rectangular region in the tensor
 *
 * @example
 * ```ts
 * const erase = new RandomErasing(0.5, [0.02, 0.33], [0.3, 3.3])
 * const augmented = erase.apply(imageTensor)
 * ```
 */
export class RandomErasing implements Transform<Tensor, Tensor> {
  private config: {
    p: number
    scaleRange: [number, number]
    ratioRange: [number, number]
    value: number
  }

  constructor(
    p: number = 0.5,
    scaleRange: [number, number] = [0.02, 0.33],
    ratioRange: [number, number] = [0.3, 3.3],
    value: number = 0,
  ) {
    if (p < 0 || p > 1) {
      throw new Error('Probability must be between 0 and 1')
    }
    this.config = { p, scaleRange, ratioRange, value }
  }

  apply(tensor: Tensor): Tensor {
    if (Math.random() >= this.config.p) {
      return tensor
    }

    const shape = tensor.shape as readonly number[]
    const { batch, channels, height, width } = parseImageShape(shape)
    const data = toNumberArray(tensor)
    const out = new Float32Array(data)

    for (let n = 0; n < batch; n++) {
      const area = height * width
      const targetArea = randomInRange(this.config.scaleRange[0], this.config.scaleRange[1]) * area
      const aspect = randomInRange(this.config.ratioRange[0], this.config.ratioRange[1])

      const eraseH = Math.max(1, Math.round(Math.sqrt(targetArea * aspect)))
      const eraseW = Math.max(1, Math.round(Math.sqrt(targetArea / aspect)))

      if (eraseH > height || eraseW > width) {
        continue
      }

      const top = Math.floor(Math.random() * (height - eraseH + 1))
      const left = Math.floor(Math.random() * (width - eraseW + 1))

      for (let c = 0; c < channels; c++) {
        for (let h = 0; h < eraseH; h++) {
          for (let w = 0; w < eraseW; w++) {
            const idx = (((n * channels + c) * height + (top + h)) * width + (left + w))
            out[idx] = this.config.value
          }
        }
      }
    }

    return cpu.tensor(out, shape as unknown as readonly number[], tensor.dtype as DType<string>) as Tensor
  }
}

/**
 * Clamp tensor values to a range
 *
 * @example
 * ```ts
 * const clamp = new Clamp(0, 1)  // Clamp to [0, 1]
 * const clamped = clamp.apply(tensor)
 * ```
 */
export class Clamp implements Transform<Tensor, Tensor> {
  constructor(
    private min: number,
    private max: number,
  ) {
    if (min > max) {
      throw new Error('min must be less than or equal to max')
    }
  }

  apply(tensor: Tensor): Tensor {
    // Use clamp if available, otherwise return original
    if ('clamp' in tensor && typeof (tensor as any).clamp === 'function') {
      return (tensor as any).clamp(this.min, this.max)
    }
    // Fallback: return original tensor
    return tensor
  }
}

function toNumberArray(tensor: Tensor): Float32Array {
  const data = tensor.toArray() as Iterable<number | bigint>
  return Float32Array.from(data, (value) => Number(value))
}

function parseImageShape(shape: readonly number[]): {
  batch: number
  channels: number
  height: number
  width: number
} {
  if (shape.length === 2) {
    return { batch: 1, channels: 1, height: shape[0]!, width: shape[1]! }
  }
  if (shape.length === 3) {
    return { batch: 1, channels: shape[0]!, height: shape[1]!, width: shape[2]! }
  }
  if (shape.length === 4) {
    return { batch: shape[0]!, channels: shape[1]!, height: shape[2]!, width: shape[3]! }
  }
  throw new Error(`Expected tensor with shape [H,W], [C,H,W], or [N,C,H,W], got [${shape.join(', ')}]`)
}

function flattenNested(data: number[] | number[][] | number[][][]): { flat: number[]; shape: number[] } {
  if (Array.isArray(data) && typeof data[0] === 'number') {
    return { flat: data as number[], shape: [data.length] }
  }

  if (Array.isArray(data) && Array.isArray(data[0])) {
    const outer = data as number[][]
    const rows = outer.length
    const cols = outer[0]?.length ?? 0
    const flat: number[] = []
    for (const row of outer) {
      flat.push(...row)
    }
    return { flat, shape: [rows, cols] }
  }

  const outer = data as number[][][]
  const depth = outer.length
  const rows = outer[0]?.length ?? 0
  const cols = outer[0]?.[0]?.length ?? 0
  const flat: number[] = []
  for (const matrix of outer) {
    for (const row of matrix) {
      flat.push(...row)
    }
  }
  return { flat, shape: [depth, rows, cols] }
}

function randomInRange(min: number, max: number): number {
  return min + Math.random() * (max - min)
}
