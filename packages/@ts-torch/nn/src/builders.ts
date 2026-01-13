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
import type { DType, Shape, DeviceType } from '@ts-torch/core'
import type { DeviceContext as DeviceContextType } from '@ts-torch/core'

/**
 * Device context type - matches the DeviceContext from @ts-torch/core
 */
type DeviceContext<Dev extends DeviceType = DeviceType> = DeviceContextType<Dev>

// ==================== Model Builders ====================

/**
 * Create a Sequential model from an array of layers
 *
 * Layers are created on CPU by default (via nn.linear(), etc.) and moved to
 * the target device. The returned Sequential has the correct device type.
 *
 * @template In - Input shape
 * @template Out - Output shape
 * @template D - Data type
 * @template Dev - Target device type (inferred from device parameter)
 * @param device - Device to place model on
 * @param layers - Array of modules to chain together (typically created on CPU)
 * @returns Sequential model on the specified device
 *
 * @example
 * ```ts
 * const cuda = device.cuda(0)
 *
 * // Layers are created on CPU and moved to CUDA
 * const model = nn.sequence(cuda, [
 *   nn.linear(784, 128),  // Created on CPU
 *   nn.relu(),
 *   nn.linear(128, 10)
 * ])
 * // model type: Sequential<..., 'cuda'>
 * ```
 */
export function sequence<
  In extends Shape = Shape,
  Out extends Shape = Shape,
  D extends DType<string> = float32,
  Dev extends DeviceType = DeviceType,
>(
  device: DeviceContext<Dev>,
  // Accept layers with any device type - they'll be moved to target device
  layers: Module<any, any, D, DeviceType>[],
): Sequential<In, Out, D, Dev> {
  // Create Sequential (inherits CPU device from layers)
  const cpuModel = new Sequential<In, Out, D, DeviceType>(...layers)
  // Move to target device and return with correct device type
  return cpuModel.to(device.type) as Sequential<In, Out, D, Dev>
}

/**
 * MLP configuration
 *
 * @template Dev - Target device type
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
export interface MLPConfig<Dev extends DeviceType = DeviceType> {
  /** Device to place model on */
  device: DeviceContext<Dev>
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
 * Layers are created on CPU and moved to the target device.
 *
 * @template Dev - Target device type (inferred from config.device)
 * @param config - MLP configuration
 * @returns Sequential model representing the MLP on the target device
 *
 * @example
 * ```ts
 * const cuda = device.cuda(0)
 *
 * const model = nn.mlp({
 *   device: cuda,
 *   layers: [784, 128, 64, 10],
 *   activation: 'gelu',
 *   dropout: 0.2
 * })
 * // model type: Sequential<..., 'cuda'>
 * ```
 */
export function mlp<Dev extends DeviceType>(config: MLPConfig<Dev>): Sequential<Shape, Shape, float32, Dev> {
  const { device, layers: layerSizes, activation = 'relu', dropout = 0, outputActivation = false } = config

  if (layerSizes.length < 2) {
    throw new Error('MLP requires at least 2 layer sizes (input and output)')
  }

  const layers: Module<any, any, float32, 'cpu'>[] = []

  for (let i = 0; i < layerSizes.length - 1; i++) {
    const inSize = layerSizes[i]!
    const outSize = layerSizes[i + 1]!
    const isLastLayer = i === layerSizes.length - 2

    // Add linear layer (created on CPU)
    layers.push(new Linear(inSize, outSize))

    // Add activation (skip for last layer unless outputActivation is true)
    if (!isLastLayer || outputActivation) {
      layers.push(createActivation(activation) as Module<any, any, float32, 'cpu'>)
    }

    // Add dropout (skip for last layer)
    if (dropout > 0 && !isLastLayer) {
      layers.push(new Dropout(dropout) as Module<any, any, float32, 'cpu'>)
    }
  }

  // Create on CPU and move to target device
  const cpuModel = new Sequential<Shape, Shape, float32, 'cpu'>(...layers)
  // Note: .to() returns Module, cast to Sequential since we know the concrete type
  return cpuModel.to(device.type) as unknown as Sequential<Shape, Shape, float32, Dev>
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
