/**
 * Data transformation utilities
 */

import type { Tensor } from "@ts-torch/core";

/**
 * Base transform interface
 */
export interface Transform<TInput = Tensor, TOutput = Tensor> {
  apply(input: TInput): TOutput | Promise<TOutput>;
}

/**
 * Compose multiple transforms
 */
export class Compose<T = Tensor> implements Transform<T, T> {
  constructor(private transforms: Transform<T, T>[]) {}

  async apply(input: T): Promise<T> {
    let result = input;
    for (const transform of this.transforms) {
      result = await Promise.resolve(transform.apply(result));
    }
    return result;
  }
}

/**
 * Normalize transform
 */
export class Normalize implements Transform<Tensor, Tensor> {
  constructor(_mean: number | number[], _std: number | number[]) {}

  apply(tensor: Tensor): Tensor {
    // TODO: Implement normalization
    // return (tensor - mean) / std
    return tensor;
  }
}

/**
 * Lambda transform - apply arbitrary function
 */
export class Lambda<TInput = Tensor, TOutput = Tensor> implements Transform<TInput, TOutput> {
  constructor(private fn: (input: TInput) => TOutput | Promise<TOutput>) {}

  apply(input: TInput): TOutput | Promise<TOutput> {
    return this.fn(input);
  }
}

/**
 * Random horizontal flip transform
 */
export class RandomHorizontalFlip implements Transform<Tensor, Tensor> {
  constructor(private p: number = 0.5) {
    if (p < 0 || p > 1) {
      throw new Error("Probability must be between 0 and 1");
    }
  }

  apply(tensor: Tensor): Tensor {
    if (Math.random() < this.p) {
      // TODO: Implement horizontal flip
      // return tensor.flip(-1)
    }
    return tensor;
  }
}

/**
 * Random crop transform
 */
export class RandomCrop implements Transform<Tensor, Tensor> {
  constructor(_size: number | [number, number]) {}

  apply(tensor: Tensor): Tensor {
    // TODO: Implement random crop
    return tensor;
  }
}

/**
 * Resize transform
 */
export class Resize implements Transform<Tensor, Tensor> {
  constructor(
    _size: number | [number, number],
    _interpolation: "nearest" | "bilinear" = "bilinear",
  ) {}

  apply(tensor: Tensor): Tensor {
    // TODO: Implement resize
    return tensor;
  }
}

/**
 * Center crop transform
 */
export class CenterCrop implements Transform<Tensor, Tensor> {
  constructor(_size: number | [number, number]) {}

  apply(tensor: Tensor): Tensor {
    // TODO: Implement center crop
    return tensor;
  }
}

/**
 * Convert to tensor transform
 */
export class ToTensor implements Transform<number[] | number[][], Tensor> {
  apply(_data: number[] | number[][]): Tensor {
    // TODO: Convert array to tensor
    // return tensor(data)
    throw new Error("ToTensor not yet implemented");
  }
}
