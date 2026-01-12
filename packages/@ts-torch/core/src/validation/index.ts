/**
 * Input validation utilities for public APIs
 * Provides comprehensive runtime validation with clear error messages
 */

// Re-export existing validation functions from ffi/error for backwards compatibility

// Re-export existing validation functions
export { validateShape, validateDtype, checkNull } from '../ffi/error.js'

/**
 * Validation error for input validation failures
 * Extends Error with additional context for debugging
 */
export class ValidationError extends Error {
  public readonly parameter: string
  public readonly value: unknown
  public readonly constraint: string

  constructor(parameter: string, value: unknown, constraint: string) {
    const message = `Invalid ${parameter}: expected ${constraint}, got ${formatValue(value)}`
    super(message)
    this.name = 'ValidationError'
    this.parameter = parameter
    this.value = value
    this.constraint = constraint

    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, ValidationError)
    }
  }
}

/**
 * Format a value for error messages
 */
function formatValue(value: unknown): string {
  if (value === null) return 'null'
  if (value === undefined) return 'undefined'
  if (typeof value === 'number') {
    if (Number.isNaN(value)) return 'NaN'
    if (!Number.isFinite(value)) return value > 0 ? 'Infinity' : '-Infinity'
    return String(value)
  }
  if (Array.isArray(value)) {
    if (value.length > 5) {
      return `[${value.slice(0, 5).join(', ')}, ... (${value.length} items)]`
    }
    return `[${value.join(', ')}]`
  }
  if (typeof value === 'object') {
    return Object.prototype.toString.call(value)
  }
  return String(value)
}

// ==================== Number Validators ====================

/**
 * Validate that a value is a finite number (not NaN or Infinity)
 */
export function validateFinite(value: number, parameter: string): void {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new ValidationError(parameter, value, 'a finite number')
  }
}

/**
 * Validate that a value is a positive number (> 0)
 */
export function validatePositive(value: number, parameter: string): void {
  validateFinite(value, parameter)
  if (value <= 0) {
    throw new ValidationError(parameter, value, 'a positive number (> 0)')
  }
}

/**
 * Validate that a value is a non-negative number (>= 0)
 */
export function validateNonNegative(value: number, parameter: string): void {
  validateFinite(value, parameter)
  if (value < 0) {
    throw new ValidationError(parameter, value, 'a non-negative number (>= 0)')
  }
}

/**
 * Validate that a value is a positive integer (> 0)
 */
export function validatePositiveInt(value: number, parameter: string): void {
  if (typeof value !== 'number' || !Number.isInteger(value) || value <= 0) {
    throw new ValidationError(parameter, value, 'a positive integer (> 0)')
  }
}

/**
 * Validate that a value is a non-negative integer (>= 0)
 */
export function validateNonNegativeInt(value: number, parameter: string): void {
  if (typeof value !== 'number' || !Number.isInteger(value) || value < 0) {
    throw new ValidationError(parameter, value, 'a non-negative integer (>= 0)')
  }
}

/**
 * Validate that a value is within a range [min, max]
 */
export function validateRange(
  value: number,
  min: number,
  max: number,
  parameter: string,
  options: { minInclusive?: boolean; maxInclusive?: boolean } = {},
): void {
  const { minInclusive = true, maxInclusive = true } = options
  validateFinite(value, parameter)

  const minOk = minInclusive ? value >= min : value > min
  const maxOk = maxInclusive ? value <= max : value < max

  if (!minOk || !maxOk) {
    const minBracket = minInclusive ? '[' : '('
    const maxBracket = maxInclusive ? ']' : ')'
    throw new ValidationError(parameter, value, `a number in range ${minBracket}${min}, ${max}${maxBracket}`)
  }
}

/**
 * Validate that a value is a probability [0, 1]
 */
export function validateProbability(value: number, parameter: string): void {
  validateRange(value, 0, 1, parameter)
}

// ==================== Shape Validators ====================

/**
 * Validate that shapes are compatible for element-wise operations
 * Shapes are compatible if they're equal or broadcastable
 */
export function validateShapesCompatible(
  shape1: readonly number[],
  shape2: readonly number[],
  operation: string,
): void {
  // Check if shapes are exactly equal
  if (shape1.length === shape2.length && shape1.every((dim, i) => dim === shape2[i])) {
    return
  }

  // Check if shapes are broadcastable
  const maxLen = Math.max(shape1.length, shape2.length)
  const padded1 = [...Array(maxLen - shape1.length).fill(1), ...shape1]
  const padded2 = [...Array(maxLen - shape2.length).fill(1), ...shape2]

  for (let i = 0; i < maxLen; i++) {
    const d1 = padded1[i]!
    const d2 = padded2[i]!
    if (d1 !== d2 && d1 !== 1 && d2 !== 1) {
      throw new ValidationError(
        'shapes',
        `${formatValue(shape1)} and ${formatValue(shape2)}`,
        `compatible shapes for ${operation} (broadcastable)`,
      )
    }
  }
}

/**
 * Validate shapes for matrix multiplication
 * For matmul: [..., M, K] @ [..., K, N] -> [..., M, N]
 */
export function validateMatmulShapes(shape1: readonly number[], shape2: readonly number[]): void {
  if (shape1.length < 1 || shape2.length < 1) {
    throw new ValidationError(
      'tensor',
      shape1.length < 1 ? `${shape1.length}D tensor` : `${shape2.length}D tensor`,
      'at least 1D tensor for matmul',
    )
  }

  // Get the contracting dimensions
  const k1 = shape1[shape1.length - 1]!
  const k2 = shape2.length === 1 ? shape2[0]! : shape2[shape2.length - 2]!

  if (k1 !== k2) {
    throw new ValidationError(
      'matmul shapes',
      `${formatValue(shape1)} @ ${formatValue(shape2)} (contracting dims: ${k1} vs ${k2})`,
      'matching contracting dimensions',
    )
  }
}

/**
 * Validate dimension index is in bounds
 */
export function validateDimension(dim: number, ndim: number, parameter: string = 'dim'): void {
  const normalizedDim = dim < 0 ? dim + ndim : dim
  if (!Number.isInteger(dim) || normalizedDim < 0 || normalizedDim >= ndim) {
    throw new ValidationError(
      parameter,
      dim,
      `a valid dimension index for ${ndim}D tensor (range: ${-ndim} to ${ndim - 1})`,
    )
  }
}

/**
 * Validate that reshape preserves total element count
 */
export function validateReshape(oldShape: readonly number[], newShape: readonly number[]): void {
  const oldNumel = oldShape.reduce((a, b) => a * b, 1)
  const newNumel = newShape.reduce((a, b) => a * b, 1)

  if (oldNumel !== newNumel) {
    throw new ValidationError(
      'newShape',
      `${formatValue(newShape)} (${newNumel} elements)`,
      `a shape with ${oldNumel} elements (same as original ${formatValue(oldShape)})`,
    )
  }
}

// ==================== Module Parameter Validators ====================

/**
 * Validate Conv2d parameters
 */
export function validateConv2dParams(params: {
  inChannels: number
  outChannels: number
  kernelSize: [number, number]
  stride: [number, number]
  padding: [number, number]
  dilation: [number, number]
  groups: number
}): void {
  validatePositiveInt(params.inChannels, 'inChannels')
  validatePositiveInt(params.outChannels, 'outChannels')
  validatePositiveInt(params.kernelSize[0], 'kernelSize[0]')
  validatePositiveInt(params.kernelSize[1], 'kernelSize[1]')
  validatePositiveInt(params.stride[0], 'stride[0]')
  validatePositiveInt(params.stride[1], 'stride[1]')
  validateNonNegativeInt(params.padding[0], 'padding[0]')
  validateNonNegativeInt(params.padding[1], 'padding[1]')
  validatePositiveInt(params.dilation[0], 'dilation[0]')
  validatePositiveInt(params.dilation[1], 'dilation[1]')
  validatePositiveInt(params.groups, 'groups')

  if (params.inChannels % params.groups !== 0) {
    throw new ValidationError(
      'inChannels',
      params.inChannels,
      `divisible by groups (${params.groups})`,
    )
  }
  if (params.outChannels % params.groups !== 0) {
    throw new ValidationError(
      'outChannels',
      params.outChannels,
      `divisible by groups (${params.groups})`,
    )
  }
}

/**
 * Validate Linear layer parameters
 */
export function validateLinearParams(inFeatures: number, outFeatures: number): void {
  validatePositiveInt(inFeatures, 'inFeatures')
  validatePositiveInt(outFeatures, 'outFeatures')
}

/**
 * Validate pooling parameters
 */
export function validatePoolingParams(params: {
  kernelSize: [number, number]
  stride: [number, number]
  padding: [number, number]
}): void {
  validatePositiveInt(params.kernelSize[0], 'kernelSize[0]')
  validatePositiveInt(params.kernelSize[1], 'kernelSize[1]')
  validatePositiveInt(params.stride[0], 'stride[0]')
  validatePositiveInt(params.stride[1], 'stride[1]')
  validateNonNegativeInt(params.padding[0], 'padding[0]')
  validateNonNegativeInt(params.padding[1], 'padding[1]')
}

/**
 * Validate normalization layer parameters
 */
export function validateNormParams(params: {
  numFeatures?: number
  normalizedShape?: readonly number[]
  eps: number
  momentum?: number
}): void {
  if (params.numFeatures !== undefined) {
    validatePositiveInt(params.numFeatures, 'numFeatures')
  }
  if (params.normalizedShape !== undefined) {
    for (let i = 0; i < params.normalizedShape.length; i++) {
      validatePositiveInt(params.normalizedShape[i]!, `normalizedShape[${i}]`)
    }
  }
  validatePositive(params.eps, 'eps')
  if (params.momentum !== undefined) {
    validateRange(params.momentum, 0, 1, 'momentum')
  }
}

// ==================== Optimizer Parameter Validators ====================

/**
 * Validate SGD optimizer parameters
 */
export function validateSGDParams(params: {
  lr: number
  momentum?: number
  weightDecay?: number
}): void {
  validatePositive(params.lr, 'lr (learning rate)')
  if (params.momentum !== undefined) {
    validateRange(params.momentum, 0, 1, 'momentum', { maxInclusive: false })
  }
  if (params.weightDecay !== undefined) {
    validateNonNegative(params.weightDecay, 'weightDecay')
  }
}

/**
 * Validate Adam optimizer parameters
 */
export function validateAdamParams(params: {
  lr: number
  betas?: [number, number]
  eps?: number
  weightDecay?: number
}): void {
  validatePositive(params.lr, 'lr (learning rate)')
  if (params.betas !== undefined) {
    validateRange(params.betas[0], 0, 1, 'betas[0]', { maxInclusive: false })
    validateRange(params.betas[1], 0, 1, 'betas[1]', { maxInclusive: false })
  }
  if (params.eps !== undefined) {
    validatePositive(params.eps, 'eps')
  }
  if (params.weightDecay !== undefined) {
    validateNonNegative(params.weightDecay, 'weightDecay')
  }
}

/**
 * Validate RMSprop optimizer parameters
 */
export function validateRMSpropParams(params: {
  lr: number
  alpha?: number
  eps?: number
  weightDecay?: number
  momentum?: number
}): void {
  validatePositive(params.lr, 'lr (learning rate)')
  if (params.alpha !== undefined) {
    validateRange(params.alpha, 0, 1, 'alpha')
  }
  if (params.eps !== undefined) {
    validatePositive(params.eps, 'eps')
  }
  if (params.weightDecay !== undefined) {
    validateNonNegative(params.weightDecay, 'weightDecay')
  }
  if (params.momentum !== undefined) {
    validateRange(params.momentum, 0, 1, 'momentum', { maxInclusive: false })
  }
}

// ==================== DataLoader Validators ====================

/**
 * Validate DataLoader parameters
 */
export function validateDataLoaderParams(params: {
  batchSize: number
  shuffle?: boolean
  dropLast?: boolean
}): void {
  validatePositiveInt(params.batchSize, 'batchSize')
}

/**
 * Validate that a dataset is not empty
 */
export function validateDatasetNotEmpty(length: number): void {
  if (length === 0) {
    throw new ValidationError('dataset', 'empty dataset', 'a dataset with at least one sample')
  }
}

// ==================== Scalar Value Validators ====================

/**
 * Validate a scalar for tensor operations (must be finite - not NaN or Infinity)
 */
export function validateScalar(value: number, parameter: string = 'scalar'): void {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new ValidationError(parameter, value, 'a finite number (not NaN or Infinity)')
  }
}

/**
 * Validate divisor is not zero
 */
export function validateNonZero(value: number, parameter: string): void {
  validateFinite(value, parameter)
  if (value === 0) {
    throw new ValidationError(parameter, value, 'a non-zero number')
  }
}
