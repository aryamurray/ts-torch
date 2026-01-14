/**
 * Diagonal Gaussian Distribution for Continuous Actions
 *
 * Used by policies with continuous action spaces (e.g., Pendulum, MuJoCo).
 * Parameterized by mean and log standard deviation.
 *
 * @example
 * ```ts
 * const { mean, logStd } = policy.forward(obs)
 * const dist = new DiagGaussianDistribution(mean, logStd)
 * const actions = dist.sample()  // [batch, actionDim]
 * const logProbs = dist.logProb(actions)
 * const entropy = dist.entropy()
 * ```
 */

import type { Tensor } from '@ts-torch/core'
import type { Distribution } from './base.js'

// Constants
const LOG_2PI = Math.log(2 * Math.PI)
const LOG_STD_MIN = -20
const LOG_STD_MAX = 2

/**
 * Diagonal Gaussian distribution for continuous action spaces
 *
 * Each action dimension is modeled as an independent Gaussian.
 * Log probability is the sum of log probabilities across dimensions.
 */
export class DiagGaussianDistribution implements Distribution<readonly [number, number]> {
  private readonly mean_: Tensor<readonly [number, number]>
  private readonly logStd_: Tensor<readonly [number, number]> | Tensor<readonly [number]>
  private readonly batchSize: number
  private readonly actionDim: number
  private meanArray_: Float32Array | null = null
  private logStdArray_: Float32Array | null = null

  /**
   * Create a diagonal Gaussian distribution
   *
   * @param mean - Mean of the distribution [batch, actionDim]
   * @param logStd - Log standard deviation [batch, actionDim] or [actionDim] (shared)
   */
  constructor(
    mean: Tensor<readonly [number, number]>,
    logStd: Tensor<readonly [number, number]> | Tensor<readonly [number]>,
  ) {
    this.mean_ = mean
    this.logStd_ = logStd
    this.batchSize = mean.shape[0]
    this.actionDim = mean.shape[1]
  }

  /**
   * Get mean array (cached)
   */
  private getMeanArray(): Float32Array {
    if (this.meanArray_ === null) {
      this.meanArray_ = this.mean_.toArray() as Float32Array
    }
    return this.meanArray_
  }

  /**
   * Get log std array (cached), expanded to [batch, actionDim] if needed
   */
  private getLogStdArray(): Float32Array {
    if (this.logStdArray_ === null) {
      const raw = this.logStd_.toArray() as Float32Array

      // If logStd is [actionDim], expand to [batch, actionDim]
      if (this.logStd_.shape.length === 1) {
        this.logStdArray_ = new Float32Array(this.batchSize * this.actionDim)
        for (let b = 0; b < this.batchSize; b++) {
          for (let a = 0; a < this.actionDim; a++) {
            // Clamp log_std to prevent numerical issues
            const clampedLogStd = Math.max(LOG_STD_MIN, Math.min(LOG_STD_MAX, raw[a]!))
            this.logStdArray_[b * this.actionDim + a] = clampedLogStd
          }
        }
      } else {
        // Clamp the values
        this.logStdArray_ = new Float32Array(raw.length)
        for (let i = 0; i < raw.length; i++) {
          this.logStdArray_[i] = Math.max(LOG_STD_MIN, Math.min(LOG_STD_MAX, raw[i]!))
        }
      }
    }
    return this.logStdArray_
  }

  /**
   * Sample actions from the distribution
   *
   * Uses reparameterization trick: action = mean + std * noise
   * @returns Sampled actions [batch, actionDim]
   */
  sample(): Tensor<readonly [number, number]> {
    const mean = this.getMeanArray()
    const logStd = this.getLogStdArray()
    const result = new Float32Array(this.batchSize * this.actionDim)

    for (let i = 0; i < result.length; i++) {
      const std = Math.exp(logStd[i]!)
      const noise = this.sampleNormal()
      result[i] = mean[i]! + std * noise
    }

    return this.createTensor(result, [this.batchSize, this.actionDim])
  }

  /**
   * Get deterministic actions (mean)
   * @returns Mean actions [batch, actionDim]
   */
  mode(): Tensor<readonly [number, number]> {
    // Clone mean tensor
    const mean = this.getMeanArray()
    return this.createTensor(new Float32Array(mean), [this.batchSize, this.actionDim])
  }

  /**
   * Compute log probability of given actions
   *
   * log p(a) = sum_i(-0.5 * log(2*pi) - log_std_i - 0.5 * ((a_i - mean_i) / std_i)^2)
   *
   * @param actions - Actions [batch, actionDim]
   * @returns Log probabilities [batch]
   */
  logProb(actions: Tensor<readonly [number, number]>): Tensor<readonly [number]> {
    const mean = this.getMeanArray()
    const logStd = this.getLogStdArray()
    const actionArray = actions.toArray() as Float32Array
    const result = new Float32Array(this.batchSize)

    for (let b = 0; b < this.batchSize; b++) {
      let logProbSum = 0
      const offset = b * this.actionDim

      for (let a = 0; a < this.actionDim; a++) {
        const idx = offset + a
        const diff = actionArray[idx]! - mean[idx]!
        const std = Math.exp(logStd[idx]!)

        // Log probability of Gaussian
        // -0.5 * log(2*pi) - log_std - 0.5 * ((x - mean) / std)^2
        logProbSum += -0.5 * LOG_2PI - logStd[idx]! - 0.5 * (diff / std) ** 2
      }

      result[b] = logProbSum
    }

    return this.createTensor(result, [this.batchSize])
  }

  /**
   * Compute entropy of the distribution
   *
   * H = sum_i(0.5 * log(2*pi*e) + log_std_i) = sum_i(0.5 + 0.5*log(2*pi) + log_std_i)
   *
   * @returns Entropy values [batch]
   */
  entropy(): Tensor<readonly [number]> {
    const logStd = this.getLogStdArray()
    const result = new Float32Array(this.batchSize)

    // Entropy of diagonal Gaussian: 0.5 * d * (1 + log(2*pi)) + sum(log_std)
    const perDimEntropy = 0.5 * (1 + LOG_2PI)

    for (let b = 0; b < this.batchSize; b++) {
      let entropy = this.actionDim * perDimEntropy
      const offset = b * this.actionDim

      for (let a = 0; a < this.actionDim; a++) {
        entropy += logStd[offset + a]!
      }

      result[b] = entropy
    }

    return this.createTensor(result, [this.batchSize])
  }

  /**
   * Get the mean tensor
   */
  get mean(): Tensor<readonly [number, number]> {
    return this.mean_
  }

  /**
   * Get the log std tensor
   */
  get logStd(): Tensor<readonly [number, number]> | Tensor<readonly [number]> {
    return this.logStd_
  }

  // ==================== Tensor Methods (for training with autograd) ====================

  /**
   * Get log probabilities tensor - for continuous this returns the mean tensor
   * as a placeholder (actual log_prob computation needs action values)
   * 
   * Note: For continuous actions, use logProbTensor(actions) instead.
   * This method exists for API compatibility with CategoricalDistribution.
   * 
   * @returns Mean tensor [batch, actionDim] (NOT actual log probs)
   */
  logProbsTensor(): any {
    // For continuous distributions, we can't return log_probs without knowing actions
    // Return mean as placeholder - caller should use logProbTensor(actions) for actual computation
    return this.mean_
  }

  /**
   * Compute log probability tensor for given actions
   * 
   * @param actions - Actions to evaluate [batch, actionDim]
   * @returns Log probability tensor [batch] (summed over action dims)
   */
  logProbTensor(actions: Float32Array): any {
    const { fromArray } = require('@ts-torch/core')

    // Create action tensor [batch, actionDim]
    const actionTensor = fromArray(actions, [this.batchSize, this.actionDim] as const)

    // diff = action - mean [batch, actionDim]
    const diff = (actionTensor as any).sub(this.mean_)

    // diff_squared = diff * diff [batch, actionDim]
    const diffSquared = (diff as any).mul(diff)

    // Get log_std as tensor, expand if needed
    let logStdTensor: any
    if (this.logStd_.shape.length === 1) {
      const logStdArray = this.logStd_.toArray() as Float32Array
      const expanded = new Float32Array(this.batchSize * this.actionDim)
      for (let b = 0; b < this.batchSize; b++) {
        for (let a = 0; a < this.actionDim; a++) {
          expanded[b * this.actionDim + a] = Math.max(LOG_STD_MIN, Math.min(LOG_STD_MAX, logStdArray[a]!))
        }
      }
      logStdTensor = fromArray(expanded, [this.batchSize, this.actionDim] as const)
    } else {
      logStdTensor = this.logStd_
    }

    // variance = exp(log_std)^2
    const std = (logStdTensor as any).exp()
    const variance = (std as any).mul(std)

    // normalized = diff_squared / variance [batch, actionDim]
    const normalized = (diffSquared as any).div(variance)

    // log_prob_per_dim = -0.5 * log(2π) - log_std - 0.5 * normalized
    const negLogStd = (logStdTensor as any).neg()
    const negHalfNorm = (normalized as any).mulScalar(-0.5)
    const logProbElements = (negLogStd as any).add(negHalfNorm).addScalar(-0.5 * LOG_2PI)

    // Sum over action dimension to get per-sample log prob [batch]
    const logProbs = (logProbElements as any).sumDim(1)

    return logProbs
  }

  /**
   * Compute policy gradient loss for continuous actions
   * 
   * Uses tensor operations to maintain computational graph.
   * log_prob = sum(-0.5 * log(2π) - log_std - 0.5 * ((a - μ) / σ)²)
   * loss = -mean(log_prob * advantage)
   * 
   * @param actions - Actions taken [batch, actionDim]
   * @param advantages - Advantage values [batch]
   * @returns Scalar loss tensor connected to mean (and logStd if learnable)
   */
  policyGradientLoss(actions: Float32Array, advantages: Float32Array): any {
    const { fromArray } = require('@ts-torch/core')
    
    // Create action tensor [batch, actionDim]
    const actionTensor = fromArray(actions, [this.batchSize, this.actionDim] as const)
    
    // diff = action - mean [batch, actionDim]
    const diff = (actionTensor as any).sub(this.mean_)
    
    // diff_squared = diff * diff [batch, actionDim]
    const diffSquared = (diff as any).mul(diff)
    
    // Get log_std as tensor, expand if needed
    let logStdTensor: any
    if (this.logStd_.shape.length === 1) {
      // Shared log_std [actionDim] - need to expand to [batch, actionDim]
      const logStdArray = this.logStd_.toArray() as Float32Array
      const expanded = new Float32Array(this.batchSize * this.actionDim)
      for (let b = 0; b < this.batchSize; b++) {
        for (let a = 0; a < this.actionDim; a++) {
          expanded[b * this.actionDim + a] = Math.max(LOG_STD_MIN, Math.min(LOG_STD_MAX, logStdArray[a]!))
        }
      }
      logStdTensor = fromArray(expanded, [this.batchSize, this.actionDim] as const)
    } else {
      logStdTensor = this.logStd_
    }
    
    // variance = exp(2 * log_std) = exp(log_std)^2
    const std = (logStdTensor as any).exp()
    const variance = (std as any).mul(std)
    
    // normalized = diff_squared / variance [batch, actionDim]
    const normalized = (diffSquared as any).div(variance)
    
    // log_prob per element = -0.5 * log(2π) - log_std - 0.5 * normalized
    // We compute: -0.5 * (log(2π) + normalized) - log_std
    // Total sum = mean * (batch * actionDim), we want sum over actionDim, mean over batch
    
    // Compute each component and combine
    // component1 = -0.5 * log(2π) per element = constant
    // component2 = -log_std per element
    // component3 = -0.5 * normalized per element
    
    const negLogStd = (logStdTensor as any).neg()  // [batch, actionDim]
    const negHalfNorm = (normalized as any).mulScalar(-0.5)  // [batch, actionDim]
    
    // log_prob_elements = -0.5*log(2π) - log_std - 0.5*normalized [batch, actionDim]
    const logProbElements = (negLogStd as any).add(negHalfNorm).addScalar(-0.5 * LOG_2PI)
    
    // Multiply by advantages and reduce
    // We want: sum_batch(sum_actions(log_prob_element) * advantage) / batch
    // = sum_batch_actions(log_prob_element * advantage_expanded) / batch
    // Using mean() which gives sum / (batch * actionDim), then multiply by actionDim
    // Create advantage tensor expanded to [batch, actionDim]
    const advantagesExpanded = new Float32Array(this.batchSize * this.actionDim)
    for (let b = 0; b < this.batchSize; b++) {
      for (let a = 0; a < this.actionDim; a++) {
        advantagesExpanded[b * this.actionDim + a] = advantages[b]! / this.actionDim  // Divide by actionDim to normalize
      }
    }
    const advantagesTensor = fromArray(advantagesExpanded, [this.batchSize, this.actionDim] as const)
    
    // weighted = log_prob_elements * advantages [batch, actionDim]
    const weighted = (logProbElements as any).mul(advantagesTensor)
    
    // loss = -mean(weighted) * actionDim (to get sum over actions, mean over batch)
    const loss = (weighted as any).mean().neg().mulScalar(this.actionDim)
    
    return loss
  }

  /**
   * Compute mean entropy as scalar tensor
   * 
   * H = 0.5 * actionDim * (1 + log(2π)) + sum(log_std)
   * 
   * @returns Scalar entropy tensor
   */
  meanEntropyTensor(): any {
    const { fromArray } = require('@ts-torch/core')
    
    // Get log_std as tensor
    let logStdTensor: any
    if (this.logStd_.shape.length === 1) {
      const logStdArray = this.logStd_.toArray() as Float32Array
      const expanded = new Float32Array(this.batchSize * this.actionDim)
      for (let b = 0; b < this.batchSize; b++) {
        for (let a = 0; a < this.actionDim; a++) {
          expanded[b * this.actionDim + a] = Math.max(LOG_STD_MIN, Math.min(LOG_STD_MAX, logStdArray[a]!))
        }
      }
      logStdTensor = fromArray(expanded, [this.batchSize, this.actionDim] as const)
    } else {
      logStdTensor = this.logStd_
    }
    
    // Per-dim constant: 0.5 * (1 + log(2π))
    const perDimConst = 0.5 * (1 + LOG_2PI)
    
    // entropy_per_element = perDimConst + log_std [batch, actionDim]
    const entropyElements = (logStdTensor as any).addScalar(perDimConst)
    
    // Mean over batch, sum over actionDim: mean() * actionDim
    const meanEntropy = (entropyElements as any).mean().mulScalar(this.actionDim)
    
    return meanEntropy
  }

  /**
   * Get action dimension
   */
  get actionDimension(): number {
    return this.actionDim
  }

  /**
   * Get batch size
   */
  get size(): number {
    return this.batchSize
  }

  // ==================== Array Methods (for inference) ====================

  /**
   * Sample from standard normal distribution (Box-Muller transform)
   */
  private sampleNormal(): number {
    const u1 = Math.random()
    const u2 = Math.random()
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
  }

  /**
   * Create a tensor from data
   */
  private createTensor<S extends readonly number[]>(
    data: Float32Array,
    shape: S,
  ): Tensor<S> {
    const { fromArray } = require('@ts-torch/core')
    return fromArray(data, shape)
  }
}

/**
 * Create a diagonal Gaussian distribution
 *
 * @param mean - Mean of the distribution [batch, actionDim]
 * @param logStd - Log standard deviation [batch, actionDim] or [actionDim]
 * @returns Diagonal Gaussian distribution
 */
export function diagGaussian(
  mean: Tensor<readonly [number, number]>,
  logStd: Tensor<readonly [number, number]> | Tensor<readonly [number]>,
): DiagGaussianDistribution {
  return new DiagGaussianDistribution(mean, logStd)
}
