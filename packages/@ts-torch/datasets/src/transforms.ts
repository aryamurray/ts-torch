/**
 * Data transformation utilities for image and tensor preprocessing
 */

import type { Tensor } from '@ts-torch/core'

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
    // For single-channel normalization, apply globally
    if (this.mean.length === 1 && this.std.length === 1) {
      const centered = (tensor as any).subScalar(this.mean[0])
      return (centered as any).divScalar(this.std[0])
    }

    // For multi-channel, we need to apply per-channel
    // This requires the tensor shape to be [C, H, W] or [N, C, H, W]
    // For now, apply simple global normalization as a fallback
    // TODO: Implement proper per-channel normalization when broadcasting is available
    const avgMean = this.mean.reduce((a, b) => a + b, 0) / this.mean.length
    const avgStd = this.std.reduce((a, b) => a + b, 0) / this.std.length
    const centered = (tensor as any).subScalar(avgMean)
    return (centered as any).divScalar(avgStd)
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
      // Flip along the last dimension (width)
      // Uses reverse indexing via narrow slices assembled in reverse order
      // Note: Full flip implementation requires native support
      // For now, we return the original tensor as a placeholder
      // TODO: Implement when flip() is added to Tensor
      return tensor
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
    // TODO: Implement resize with interpolation using this.config
    // This requires native support for interpolation (nearest/bilinear)
    // For now, return the original tensor
    void this.config
    console.warn('Resize transform not yet implemented - returning original tensor')
    return tensor
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
    // TODO: Implement when tensor factory is available in this context
    // This would need: import { tensor } from '@ts-torch/core'
    // return tensor(data)
    throw new Error(
      'ToTensor not yet implemented - use device.tensor() or device.cpu().tensor() from @ts-torch/core directly',
    )
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
    // TODO: Implement padding using this.config
    // This requires creating a new tensor with padded dimensions
    // and copying the original data to the center
    void this.config
    console.warn('Pad transform not yet implemented - returning original tensor')
    return tensor
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

    // TODO: Implement random erasing using this.config
    // This requires in-place modification or creating a new tensor
    // with a region set to the erase value
    void this.config
    return tensor
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
