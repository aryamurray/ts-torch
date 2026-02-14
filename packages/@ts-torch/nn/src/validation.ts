/**
 * State Dict Validation
 *
 * Validates that a state dict is compatible with a model's parameters
 * before loading. Used by both loadStateDict() and config.load().
 */

import type { Parameter } from './module.js'
import type { StateDict } from './safetensors.js'

/** Structural type for anything with namedParameters() */
interface HasNamedParameters {
  namedParameters(): Map<string, Parameter<any, any, any>>
}

export class MissingKeyError extends Error {
  constructor(key: string) {
    super(`Missing key in state dict: "${key}"`)
    this.name = 'MissingKeyError'
  }
}

export class UnexpectedKeyError extends Error {
  constructor(key: string) {
    super(`Unexpected key in state dict: "${key}"`)
    this.name = 'UnexpectedKeyError'
  }
}

export class ShapeMismatchError extends Error {
  constructor(key: string, expected: number[], actual: number[]) {
    super(
      `Shape mismatch for "${key}": model expects [${expected.join(', ')}] but state dict has [${actual.join(', ')}]`,
    )
    this.name = 'ShapeMismatchError'
  }
}

export class DTypeMismatchError extends Error {
  constructor(key: string, expected: string, actual: string) {
    super(
      `DType mismatch for "${key}": model expects ${expected} but state dict has ${actual}`,
    )
    this.name = 'DTypeMismatchError'
  }
}

export class DataLengthMismatchError extends Error {
  constructor(key: string, expected: number, actual: number, shape: number[]) {
    super(
      `Data length mismatch for "${key}": shape [${shape.join(', ')}] expects ${expected} elements but got ${actual}`,
    )
    this.name = 'DataLengthMismatchError'
  }
}

/**
 * Validate that a state dict is compatible with a model's parameters.
 *
 * @param model - The model whose parameters to validate against
 * @param state - The state dict to validate
 * @param strict - If true (default), require exact key match. If false, skip missing/unexpected key checks.
 *
 * @throws MissingKeyError - model has parameter not in state dict (strict mode only)
 * @throws UnexpectedKeyError - state dict has key not in model (strict mode only)
 * @throws ShapeMismatchError - shape mismatch (always, regardless of strict)
 * @throws DTypeMismatchError - dtype mismatch (always, regardless of strict)
 */
export function validateStateDict(
  model: HasNamedParameters,
  state: StateDict,
  strict: boolean = true,
): void {
  const currentParams = model.namedParameters()
  const stateKeys = new Set(Object.keys(state))
  const paramKeys = new Set(currentParams.keys())

  if (strict) {
    // Check for missing keys (model has param, state doesn't)
    for (const key of paramKeys) {
      if (!stateKeys.has(key)) {
        throw new MissingKeyError(key)
      }
    }
    // Check for unexpected keys (state has key, model doesn't)
    for (const key of stateKeys) {
      if (!paramKeys.has(key)) {
        throw new UnexpectedKeyError(key)
      }
    }
  }

  // Shape and dtype checks for all matching keys (always enforced)
  for (const [name, param] of currentParams) {
    const tensorData = state[name]
    if (!tensorData) continue

    const tensor = param.data as any

    // Shape check
    const modelShape: number[] = Array.isArray(tensor.shape) ? tensor.shape : [tensor.shape]
    const stateShape = tensorData.shape
    if (
      modelShape.length !== stateShape.length ||
      !modelShape.every((dim: number, i: number) => dim === stateShape[i])
    ) {
      throw new ShapeMismatchError(name, modelShape, stateShape)
    }

    // Data length check
    const expectedLength = stateShape.reduce((a: number, d: number) => a * d, 1)
    if (tensorData.data.length !== expectedLength) {
      throw new DataLengthMismatchError(name, expectedLength, tensorData.data.length, stateShape)
    }

    // DType check
    const modelDtype: string = tensor.dtype?.name ?? 'float32'
    const stateDtype = tensorData.dtype
    if (modelDtype !== stateDtype) {
      throw new DTypeMismatchError(name, modelDtype, stateDtype)
    }
  }
}
