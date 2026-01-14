/**
 * Categorical Distribution for Discrete Actions
 *
 * Used by policies with discrete action spaces (e.g., CartPole, Atari).
 * Takes logits as input and provides sampling via softmax probabilities.
 *
 * @example
 * ```ts
 * const logits = policy.forward(obs)  // [batch, nActions]
 * const dist = new CategoricalDistribution(logits)
 * const actions = dist.sample()       // [batch] action indices
 * const logProbs = dist.logProb(actions)
 * const entropy = dist.entropy()
 * ```
 */

import type { Tensor } from '@ts-torch/core'
import type { Distribution } from './base.js'

/**
 * Categorical distribution for discrete action spaces
 *
 * Parameterized by unnormalized log probabilities (logits).
 * Internally converts to probabilities using softmax.
 */
export class CategoricalDistribution implements Distribution<readonly [number]> {
  private readonly logits_: Tensor<readonly [number, number]>
  private readonly batchSize: number
  private readonly nActions: number
  private probs_: Float32Array | null = null
  private logProbs_: Float32Array | null = null

  /**
   * Create a categorical distribution from logits
   *
   * @param logits - Unnormalized log probabilities [batch, nActions]
   */
  constructor(logits: Tensor<readonly [number, number]>) {
    this.logits_ = logits
    this.batchSize = logits.shape[0]
    this.nActions = logits.shape[1]
  }

  /**
   * Get probabilities (computed lazily and cached)
   */
  private getProbs(): Float32Array {
    if (this.probs_ === null) {
      // Compute softmax probabilities
      const softmaxTensor = this.logits_.softmax(1)
      this.probs_ = softmaxTensor.toArray() as Float32Array
    }
    return this.probs_
  }

  /**
   * Get log probabilities (computed lazily and cached)
   */
  private getLogProbs(): Float32Array {
    if (this.logProbs_ === null) {
      // Compute log-softmax for numerical stability
      const logSoftmaxTensor = this.logits_.logSoftmax(1)
      this.logProbs_ = logSoftmaxTensor.toArray() as Float32Array
    }
    return this.logProbs_
  }

  /**
   * Sample actions from the distribution
   *
   * Uses inverse transform sampling with softmax probabilities.
   * @returns Tensor of action indices [batch]
   */
  sample(): Tensor<readonly [number]> {
    const probs = this.getProbs()
    const actions = new Int32Array(this.batchSize)

    for (let b = 0; b < this.batchSize; b++) {
      const offset = b * this.nActions
      const u = Math.random()
      let cumSum = 0

      for (let a = 0; a < this.nActions; a++) {
        cumSum += probs[offset + a]!
        if (u <= cumSum) {
          actions[b] = a
          break
        }
      }
      // Handle numerical edge case
      if (actions[b] === undefined) {
        actions[b] = this.nActions - 1
      }
    }

    // Create tensor from actions
    // Note: We need to get the device context from logits
    return this.createActionTensor(actions)
  }

  /**
   * Get deterministic actions (argmax)
   * @returns Tensor of action indices [batch]
   */
  mode(): Tensor<readonly [number]> {
    const probs = this.getProbs()
    const actions = new Int32Array(this.batchSize)

    for (let b = 0; b < this.batchSize; b++) {
      const offset = b * this.nActions
      let maxProb = -Infinity
      let maxAction = 0

      for (let a = 0; a < this.nActions; a++) {
        if (probs[offset + a]! > maxProb) {
          maxProb = probs[offset + a]!
          maxAction = a
        }
      }
      actions[b] = maxAction
    }

    return this.createActionTensor(actions)
  }

  /**
   * Compute log probability of given actions
   *
   * @param actions - Action indices [batch]
   * @returns Log probabilities [batch]
   */
  logProb(actions: Tensor<readonly [number]>): Tensor<readonly [number]> {
    const logProbs = this.getLogProbs()
    const actionArray = actions.toArray() as Int32Array | Float32Array
    const result = new Float32Array(this.batchSize)

    for (let b = 0; b < this.batchSize; b++) {
      const action = Math.round(actionArray[b]!)
      result[b] = logProbs[b * this.nActions + action]!
    }

    return this.createTensor(result)
  }

  /**
   * Compute entropy of the distribution
   *
   * H = -sum(p * log(p))
   * @returns Entropy values [batch]
   */
  entropy(): Tensor<readonly [number]> {
    const probs = this.getProbs()
    const logProbs = this.getLogProbs()
    const result = new Float32Array(this.batchSize)

    for (let b = 0; b < this.batchSize; b++) {
      const offset = b * this.nActions
      let entropy = 0

      for (let a = 0; a < this.nActions; a++) {
        const p = probs[offset + a]!
        const logP = logProbs[offset + a]!
        // Only add if p > 0 to avoid NaN
        if (p > 1e-8) {
          entropy -= p * logP
        }
      }
      result[b] = entropy
    }

    return this.createTensor(result)
  }

  /**
   * Get the logits tensor
   */
  get logits(): Tensor<readonly [number, number]> {
    return this.logits_
  }

  // ==================== Tensor Methods (for training with autograd) ====================

  /**
   * Get log probabilities tensor - maintains computational graph
   * 
   * @returns Log softmax tensor [batch, nActions] connected to logits
   */
  logProbsTensor(): Tensor<readonly [number, number]> {
    return this.logits_.logSoftmax(1)
  }

  /**
   * Get probabilities tensor - maintains computational graph
   * 
   * @returns Softmax tensor [batch, nActions] connected to logits
   */
  probsTensor(): Tensor<readonly [number, number]> {
    return this.logits_.softmax(1)
  }

  /**
   * Compute mean entropy as scalar tensor - maintains computational graph
   * 
   * H = -sum(p * log(p)) averaged over batch
   * Uses: -(probs * log_probs).sum() / batchSize (as mean over batch, sum over actions)
   * 
   * @returns Scalar entropy tensor connected to logits
   */
  meanEntropyTensor(): any {
    const probs = this.probsTensor()
    const logProbs = this.logProbsTensor()
    // -sum(p * log_p) for all elements, then divide by batch size
    // This gives mean entropy per sample
    const pLogP = (probs as any).mul(logProbs)
    const total = (pLogP as any).sum()
    const meanEntropy = (total as any).neg().divScalar(this.batchSize)
    return meanEntropy
  }

  /**
   * Compute policy gradient loss component for given actions
   * 
   * Uses one-hot encoding to select log probs for taken actions, then multiplies
   * by advantages. Returns scalar loss ready for backward().
   * 
   * loss = -mean(log_prob(action) * advantage)
   * 
   * @param actions - Action indices as Float32Array [batch]
   * @param advantages - Advantage values as Float32Array [batch]
   * @returns Scalar loss tensor connected to logits
   */
  policyGradientLoss(actions: Float32Array, advantages: Float32Array): any {
    const logProbs = this.logProbsTensor()  // [batch, nActions]
    
    // Create one-hot encoding of actions [batch, nActions]
    const oneHot = new Float32Array(this.batchSize * this.nActions)
    for (let b = 0; b < this.batchSize; b++) {
      const action = Math.round(actions[b]!)
      oneHot[b * this.nActions + action] = 1.0
    }
    
    // Create tensors for one-hot and advantages
    const { fromArray } = require('@ts-torch/core')
    const oneHotTensor = fromArray(oneHot, [this.batchSize, this.nActions] as const)
    
    // Expand advantages to [batch, nActions] for element-wise multiply
    const advantagesExpanded = new Float32Array(this.batchSize * this.nActions)
    for (let b = 0; b < this.batchSize; b++) {
      for (let a = 0; a < this.nActions; a++) {
        advantagesExpanded[b * this.nActions + a] = advantages[b]!
      }
    }
    const advantagesTensor = fromArray(advantagesExpanded, [this.batchSize, this.nActions] as const)
    
    // selected_log_probs = log_probs * one_hot [batch, nActions]
    // policy_gradient = selected_log_probs * advantages [batch, nActions]
    // loss = -mean(policy_gradient) * nActions (correction for mean vs sum over actions)
    const selectedLogProbs = (logProbs as any).mul(oneHotTensor)
    const policyGradient = (selectedLogProbs as any).mul(advantagesTensor)
    
    // Since we want mean over batch but sum over actions, and mean() does both:
    // mean = sum / (batch * nActions), we want sum / batch = mean * nActions
    const loss = (policyGradient as any).mean().neg().mulScalar(this.nActions)
    
    return loss
  }

  /**
   * Get number of actions
   */
  get numActions(): number {
    return this.nActions
  }

  /**
   * Get batch size
   */
  get size(): number {
    return this.batchSize
  }

  // ==================== Array Methods (for inference) ====================

  /**
   * Create a Float32 tensor from data
   */
  private createTensor(data: Float32Array): Tensor<readonly [number]> {
    // Import fromArray dynamically to avoid circular deps
    const { fromArray } = require('@ts-torch/core')
    return fromArray(data, [this.batchSize] as const)
  }

  /**
   * Create an action tensor from Int32Array
   */
  private createActionTensor(data: Int32Array): Tensor<readonly [number]> {
    // Convert to Float32 for tensor creation (actions stored as floats)
    const floatData = new Float32Array(data)
    const { fromArray } = require('@ts-torch/core')
    return fromArray(floatData, [this.batchSize] as const)
  }
}

/**
 * Create a categorical distribution from logits
 *
 * @param logits - Unnormalized log probabilities [batch, nActions]
 * @returns Categorical distribution
 */
export function categorical(logits: Tensor<readonly [number, number]>): CategoricalDistribution {
  return new CategoricalDistribution(logits)
}
