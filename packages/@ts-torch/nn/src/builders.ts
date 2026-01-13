/**
 * Declarative Model Builders
 *
 * Provides factory functions for creating neural network models declaratively.
 * All builders accept a device context and create models directly on that device.
 *
 * @example
 * ```ts
 * import { nn } from '@ts-torch/nn'
 * import { device } from '@ts-torch/core'
 *
 * const cuda = device.cuda(0)
 *
 * // Declarative model creation
 * const model = nn.sequence(cuda, [
 *   nn.linear(784, 128),
 *   nn.relu(),
 *   nn.dropout(0.2),
 *   nn.linear(128, 10)
 * ])
 *
 * // Or use the MLP convenience
 * const mlp = nn.mlp({
 *   device: cuda,
 *   layers: [784, 128, 64, 10]
 * })
 * ```
 */

import { Sequential } from './modules/container.js'
import { Linear, type LinearOptions } from './modules/linear.js'
import { ReLU, Sigmoid, Tanh, Softmax, LeakyReLU, GELU } from './modules/activation.js'
import { Dropout } from './modules/dropout.js'
import { Module, type float32 } from './module.js'
import type { DType, Shape } from '@ts-torch/core'

/**
 * Device context type - matches the DeviceContext from @ts-torch/core
 */
interface DeviceContext {
  readonly type: 'cpu' | 'cuda' | 'mps'
  readonly index: number
}

// ==================== Model Builders ====================

/**
 * Create a Sequential model from an array of layers
 *
 * @param device - Device to place model on
 * @param layers - Array of modules to chain together
 * @returns Sequential model on the specified device
 *
 * @example
 * ```ts
 * const model = nn.sequence(cuda, [
 *   nn.linear(784, 128),
 *   nn.relu(),
 *   nn.linear(128, 10)
 * ])
 * ```
 */
export function sequence<In extends Shape = Shape, Out extends Shape = Shape, D extends DType<string> = float32>(
  device: DeviceContext,
  layers: Module<any, any, D>[],
): Sequential<In, Out, D> {
  const model = new Sequential<In, Out, D>(...layers)
  model.to(device.type)
  return model
}

/**
 * MLP configuration
 *
 * @example
 * ```ts
 * const model = nn.mlp({
 *   device: cuda,
 *   layers: [784, 128, 64, 10],
 *   activation: 'gelu',
 *   dropout: 0.2
 * })
 * ```
 */
export interface MLPConfig {
  /** Device to place model on */
  device: DeviceContext
  /** Array of layer sizes [input, hidden1, hidden2, ..., output] */
  layers: number[]
  /** Activation function between layers (default: 'relu') */
  activation?: 'relu' | 'gelu' | 'tanh' | 'sigmoid' | 'leaky_relu'
  /** Dropout probability between layers (default: 0) */
  dropout?: number
  /** Apply activation after the last layer (default: false) */
  outputActivation?: boolean
}

/**
 * Create a Multi-Layer Perceptron (MLP)
 *
 * @param config - MLP configuration
 * @returns Sequential model representing the MLP
 *
 * @example
 * ```ts
 * const model = nn.mlp({
 *   device: cuda,
 *   layers: [784, 128, 64, 10],
 *   activation: 'gelu',
 *   dropout: 0.2
 * })
 * ```
 */
export function mlp(config: MLPConfig): Sequential {
  const { device, layers: layerSizes, activation = 'relu', dropout = 0, outputActivation = false } = config

  if (layerSizes.length < 2) {
    throw new Error('MLP requires at least 2 layer sizes (input and output)')
  }

  const layers: Module<any, any, any>[] = []

  for (let i = 0; i < layerSizes.length - 1; i++) {
    const inSize = layerSizes[i]!
    const outSize = layerSizes[i + 1]!
    const isLastLayer = i === layerSizes.length - 2

    // Add linear layer
    layers.push(new Linear(inSize, outSize))

    // Add activation (skip for last layer unless outputActivation is true)
    if (!isLastLayer || outputActivation) {
      layers.push(createActivation(activation))
    }

    // Add dropout (skip for last layer)
    if (dropout > 0 && !isLastLayer) {
      layers.push(new Dropout(dropout))
    }
  }

  const model = new Sequential(...layers)
  model.to(device.type)
  return model
}

/**
 * Create activation module from name
 * @internal
 */
function createActivation(name: string): Module<any, any, any> {
  switch (name) {
    case 'relu':
      return new ReLU()
    case 'gelu':
      return new GELU()
    case 'tanh':
      return new Tanh()
    case 'sigmoid':
      return new Sigmoid()
    case 'leaky_relu':
      return new LeakyReLU()
    default:
      throw new Error(`Unknown activation: ${name}`)
  }
}

// ==================== Layer Factories ====================

/**
 * Create a Linear layer
 *
 * @param inFeatures - Number of input features
 * @param outFeatures - Number of output features
 * @param options - Linear layer options
 * @returns Linear module
 *
 * @example
 * ```ts
 * const layer = nn.linear(784, 128)
 * const layer = nn.linear(784, 128, { bias: false })
 * ```
 */
export function linear<In extends number, Out extends number>(
  inFeatures: In,
  outFeatures: Out,
  options?: LinearOptions,
): Linear<In, Out> {
  return new Linear(inFeatures, outFeatures, options)
}

/**
 * Create a ReLU activation
 *
 * @param inplace - Whether to modify input in-place
 * @returns ReLU module
 *
 * @example
 * ```ts
 * const relu = nn.relu()
 * ```
 */
export function relu(inplace = false): ReLU {
  return new ReLU(inplace)
}

/**
 * Create a GELU activation
 *
 * @returns GELU module
 *
 * @example
 * ```ts
 * const gelu = nn.gelu()
 * ```
 */
export function gelu(): GELU {
  return new GELU()
}

/**
 * Create a Sigmoid activation
 *
 * @returns Sigmoid module
 *
 * @example
 * ```ts
 * const sigmoid = nn.sigmoid()
 * ```
 */
export function sigmoid(): Sigmoid {
  return new Sigmoid()
}

/**
 * Create a Tanh activation
 *
 * @returns Tanh module
 *
 * @example
 * ```ts
 * const tanh = nn.tanh()
 * ```
 */
export function tanh(): Tanh {
  return new Tanh()
}

/**
 * Create a Softmax activation
 *
 * @param dim - Dimension along which to apply softmax (default: -1)
 * @returns Softmax module
 *
 * @example
 * ```ts
 * const softmax = nn.softmax(-1)
 * ```
 */
export function softmax(dim = -1): Softmax {
  return new Softmax(dim)
}

/**
 * Create a Leaky ReLU activation
 *
 * @param negativeSlope - Slope for negative values (default: 0.01)
 * @param inplace - Whether to modify input in-place
 * @returns LeakyReLU module
 *
 * @example
 * ```ts
 * const leakyRelu = nn.leakyRelu(0.01)
 * ```
 */
export function leakyRelu(negativeSlope = 0.01, inplace = false): LeakyReLU {
  return new LeakyReLU(negativeSlope, inplace)
}

/**
 * Create a Dropout layer
 *
 * @param p - Dropout probability (default: 0.5)
 * @param options - Dropout options
 * @returns Dropout module
 *
 * @example
 * ```ts
 * const dropout = nn.dropout(0.2)
 * ```
 */
export function dropout(p = 0.5, options?: { inplace?: boolean }): Dropout {
  return new Dropout(p, options)
}

// ==================== nn Namespace ====================

/**
 * Neural network namespace - declarative model building
 *
 * @example
 * ```ts
 * import { nn } from '@ts-torch/nn'
 *
 * // Model builders
 * const model = nn.sequence(cuda, [nn.linear(784, 10)])
 * const mlp = nn.mlp({
 *   device: cuda,
 *   layers: [784, 128, 10]
 * })
 *
 * // Layer factories
 * const layer = nn.linear(784, 128)
 * const act = nn.relu()
 * const drop = nn.dropout(0.2)
 * ```
 */
export const nn = {
  // Model builders
  sequence,
  mlp,

  // Layer factories
  linear,
  relu,
  gelu,
  sigmoid,
  tanh,
  softmax,
  leakyRelu,
  dropout,
}
