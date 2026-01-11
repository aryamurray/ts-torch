/**
 * Base Optimizer class
 */

import type { Tensor } from '@ts-torch/core'

/**
 * Parameter group for optimizer
 */
export interface ParameterGroup {
  params: Tensor[]
  lr?: number
  [key: string]: unknown
}

/**
 * Optimizer configuration options
 */
export interface OptimizerOptions {
  lr: number
  [key: string]: unknown
}

/**
 * Base class for all optimizers
 *
 * Optimizers are responsible for updating model parameters based on gradients.
 * All optimization algorithms should inherit from this class.
 */
export abstract class Optimizer {
  protected paramGroups: ParameterGroup[]
  protected defaults: OptimizerOptions
  protected state: Map<Tensor, Record<string, unknown>> = new Map()

  constructor(params: Tensor[] | ParameterGroup[], defaults: OptimizerOptions) {
    this.defaults = defaults

    // Convert params to parameter groups
    if (Array.isArray(params) && params.length > 0 && this.isTensor(params[0])) {
      this.paramGroups = [{ params: params as Tensor[], ...defaults }]
    } else {
      this.paramGroups = params as ParameterGroup[]
    }

    // Validate parameter groups
    this.validateParamGroups()
  }

  private isTensor(obj: unknown): obj is Tensor {
    return obj instanceof Object && 'shape' in obj && 'dtype' in obj
  }

  private validateParamGroups(): void {
    for (const group of this.paramGroups) {
      if (!group.params || !Array.isArray(group.params)) {
        throw new Error('Parameter group must have a params array')
      }
    }
  }

  /**
   * Perform a single optimization step
   */
  abstract step(): void

  /**
   * Zero all gradients of optimized parameters
   */
  zeroGrad(): void {
    for (const group of this.paramGroups) {
      for (const param of group.params) {
        if ('grad' in param && param.grad !== null) {
          ;(param as { grad: unknown }).grad = null
        }
      }
    }
  }

  /**
   * Get the current learning rate
   */
  get learningRate(): number {
    return this.defaults.lr
  }

  /**
   * Set the learning rate for all parameter groups
   */
  set learningRate(lr: number) {
    this.defaults.lr = lr
    for (const group of this.paramGroups) {
      group.lr = lr
    }
  }

  /**
   * Get optimizer state
   */
  getState(): Map<Tensor, Record<string, unknown>> {
    return this.state
  }

  /**
   * Load optimizer state
   */
  loadState(state: Map<Tensor, Record<string, unknown>>): void {
    this.state = state
  }

  /**
   * Get state for a specific parameter
   */
  getParamState(param: Tensor): Record<string, unknown> | undefined {
    return this.state.get(param)
  }

  /**
   * Set state for a specific parameter
   */
  setParamState(param: Tensor, state: Record<string, unknown>): void {
    this.state.set(param, state)
  }

  /**
   * Check if optimizer has state for a parameter
   */
  hasParamState(param: Tensor): boolean {
    return this.state.has(param)
  }

  /**
   * Delete optimizer state for a specific parameter
   */
  deleteParamState(param: Tensor): boolean {
    return this.state.delete(param)
  }

  /**
   * Clear all optimizer state
   */
  clearState(): void {
    this.state.clear()
  }

  /**
   * Add a parameter group
   */
  addParamGroup(paramGroup: ParameterGroup): void {
    this.paramGroups.push({
      ...this.defaults,
      ...paramGroup,
    })
  }

  /**
   * Get all parameters
   */
  getAllParams(): Tensor[] {
    return this.paramGroups.flatMap((group) => group.params)
  }

  /**
   * Get string representation
   */
  toString(): string {
    const defaultsStr = Object.entries(this.defaults)
      .map(([key, value]) => `${key}: ${value}`)
      .join(', ')
    return `${this.constructor.name}(${defaultsStr})`
  }
}
