/**
 * Transform registry for creating transforms from serializable configs
 *
 * Since functions cannot be passed to workers via postMessage,
 * transforms are specified as serializable config objects and
 * instantiated within each worker.
 */

import type { TransformConfig, TransformType } from './types.js'

/**
 * A transform function that can be applied to data
 */
export type TransformFn<T = unknown> = (input: T) => T | Promise<T>

/**
 * Parameters for different transform types
 */
export interface ResizeParams {
  size: [number, number]
  interpolation?: 'nearest' | 'bilinear'
}

export interface NormalizeParams {
  mean: number | number[]
  std: number | number[]
}

export interface CropParams {
  size: [number, number]
}

export interface FlipParams {
  p?: number
}

export interface RotationParams {
  degrees: number
}

export interface ColorJitterParams {
  brightness?: number
  contrast?: number
  saturation?: number
  hue?: number
}

/**
 * Registry of transform factories
 */
type TransformFactory = (params: Record<string, unknown>) => TransformFn

const transformFactories: Partial<Record<TransformType, TransformFactory>> = {
  resize: (params) => {
    const { size: _size } = params as unknown as ResizeParams
    return (input: unknown) => {
      // Placeholder - real implementation would resize image data
      // This would typically use a library like sharp or jimp
      return input
    }
  },

  normalize: (params) => {
    const { mean, std } = params as unknown as NormalizeParams
    const meanArr = Array.isArray(mean) ? mean : [mean]
    const stdArr = Array.isArray(std) ? std : [std]

    return (input: unknown) => {
      if (!isNumberArray(input)) {
        return input
      }

      // Simple normalization for flat arrays
      const result = new Array(input.length)
      for (let i = 0; i < input.length; i++) {
        const m = meanArr[i % meanArr.length]!
        const s = stdArr[i % stdArr.length]!
        result[i] = (input[i]! - m) / s
      }
      return result
    }
  },

  randomCrop: (_params) => {
    // const { size } = params as CropParams
    return (input: unknown) => {
      // Placeholder - real implementation would crop image data
      return input
    }
  },

  centerCrop: (_params) => {
    // const { size } = params as CropParams
    return (input: unknown) => {
      // Placeholder - real implementation would center crop image data
      return input
    }
  },

  randomHorizontalFlip: (params) => {
    const p = (params as FlipParams).p ?? 0.5
    return (input: unknown) => {
      if (Math.random() >= p) {
        return input
      }
      // Placeholder - real implementation would flip image data horizontally
      return input
    }
  },

  randomVerticalFlip: (params) => {
    const p = (params as FlipParams).p ?? 0.5
    return (input: unknown) => {
      if (Math.random() >= p) {
        return input
      }
      // Placeholder - real implementation would flip image data vertically
      return input
    }
  },

  randomRotation: (_params) => {
    // const { degrees } = params as RotationParams
    return (input: unknown) => {
      // Placeholder - real implementation would rotate image data
      return input
    }
  },

  colorJitter: (_params) => {
    // const jitterParams = params as ColorJitterParams
    return (input: unknown) => {
      // Placeholder - real implementation would apply color jittering
      return input
    }
  },

  toTensor: () => {
    return (input: unknown) => {
      // Pass through - data is already in tensor-compatible format
      return input
    }
  },
}

/**
 * Check if value is a number array
 */
function isNumberArray(value: unknown): value is number[] {
  return (
    Array.isArray(value) && value.length > 0 && typeof value[0] === 'number'
  )
}

/**
 * Create a transform function from a config
 */
export function createTransform(config: TransformConfig): TransformFn | null {
  const factory = transformFactories[config.type]
  if (!factory) {
    console.warn(`Unknown transform type: ${config.type}`)
    return null
  }
  return factory(config.params as never)
}

/**
 * Create multiple transform functions from configs
 */
export function createTransforms(configs: TransformConfig[]): TransformFn[] {
  const transforms: TransformFn[] = []
  for (const config of configs) {
    const transform = createTransform(config)
    if (transform) {
      transforms.push(transform)
    }
  }
  return transforms
}

/**
 * Compose multiple transforms into a single transform
 */
export function composeTransforms(transforms: TransformFn[]): TransformFn {
  return async (input: unknown) => {
    let result = input
    for (const transform of transforms) {
      result = await Promise.resolve(transform(result))
    }
    return result
  }
}

/**
 * Helper functions for creating transform configs
 */
export const Transforms = {
  resize: (size: [number, number], interpolation?: 'nearest' | 'bilinear'): TransformConfig => ({
    type: 'resize',
    params: { size, interpolation },
  }),

  normalize: (mean: number | number[], std: number | number[]): TransformConfig => ({
    type: 'normalize',
    params: { mean, std },
  }),

  randomCrop: (size: [number, number]): TransformConfig => ({
    type: 'randomCrop',
    params: { size },
  }),

  centerCrop: (size: [number, number]): TransformConfig => ({
    type: 'centerCrop',
    params: { size },
  }),

  randomHorizontalFlip: (p = 0.5): TransformConfig => ({
    type: 'randomHorizontalFlip',
    params: { p },
  }),

  randomVerticalFlip: (p = 0.5): TransformConfig => ({
    type: 'randomVerticalFlip',
    params: { p },
  }),

  randomRotation: (degrees: number): TransformConfig => ({
    type: 'randomRotation',
    params: { degrees },
  }),

  colorJitter: (options: ColorJitterParams): TransformConfig => ({
    type: 'colorJitter',
    params: options as unknown as Record<string, unknown>,
  }),

  toTensor: (): TransformConfig => ({
    type: 'toTensor',
    params: {},
  }),

  custom: (params: Record<string, unknown>): TransformConfig => ({
    type: 'custom',
    params,
  }),
}
