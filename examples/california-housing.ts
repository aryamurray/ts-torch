/**
 * California Housing Regression Example
 *
 * This example demonstrates:
 * - Regression with MSE loss (not classification)
 * - Feature normalization/standardization
 * - Adam optimizer (not just SGD)
 * - Deeper MLP architecture
 * - Train/validation split
 * - R² score evaluation
 * - Handling missing values and categorical features
 *
 * Dataset: California Housing from Kaggle
 * - 8 numeric features + 1 categorical (ocean_proximity, dropped)
 * - Target: median_house_value
 * - ~20,640 samples
 *
 * Download from: https://www.kaggle.com/datasets/camnugent/california-housing-prices
 */

import { torch, fromArray, DType } from '@ts-torch/core'
import { Linear, ReLU } from '@ts-torch/nn'
import { Adam, SGD, mseLoss } from '@ts-torch/optim'
import { readFileSync, existsSync } from 'node:fs'

// ==================== Configuration ====================
const CONFIG = {
  // Model architecture
  hiddenSizes: [64, 32, 16],

  // Training hyperparameters
  learningRate: 0.001,
  epochs: 50,
  batchSize: 64,
  validationSplit: 0.2,

  // Data
  dataPath: './data/california_housing/housing.csv',
}

console.log('=== California Housing Regression ===\n')

// ==================== Data Loading ====================

// Numeric feature columns (excluding ocean_proximity which is categorical)
const NUMERIC_FEATURES = [
  'longitude',
  'latitude',
  'housing_median_age',
  'total_rooms',
  'total_bedrooms',
  'population',
  'households',
  'median_income',
]

const TARGET_COLUMN = 'median_house_value'

interface Dataset {
  features: Float32Array
  targets: Float32Array
  numSamples: number
  numFeatures: number
  featureNames: string[]
}

function loadCSV(path: string): Dataset {
  if (!existsSync(path)) {
    console.error(`Dataset not found at: ${path}`)
    console.error('\nDownload from: https://www.kaggle.com/datasets/camnugent/california-housing-prices')
    process.exit(1)
  }

  const content = readFileSync(path, 'utf-8')
  const lines = content.trim().split('\n')
  const header = lines[0]!.split(',')

  // Find column indices
  const featureIndices = NUMERIC_FEATURES.map(name => {
    const idx = header.indexOf(name)
    if (idx === -1) throw new Error(`Column not found: ${name}`)
    return idx
  })
  const targetIdx = header.indexOf(TARGET_COLUMN)
  if (targetIdx === -1) throw new Error(`Target column not found: ${TARGET_COLUMN}`)

  const numFeatures = NUMERIC_FEATURES.length

  // First pass: count valid samples (no missing values)
  const validRows: number[] = []
  for (let i = 1; i < lines.length; i++) {
    const values = lines[i]!.split(',')
    let valid = true

    for (const idx of featureIndices) {
      const val = values[idx]
      if (val === '' || val === 'NA' || val === 'NaN' || isNaN(Number(val))) {
        valid = false
        break
      }
    }

    const targetVal = values[targetIdx]
    if (targetVal === '' || targetVal === 'NA' || isNaN(Number(targetVal))) {
      valid = false
    }

    if (valid) validRows.push(i)
  }

  const numSamples = validRows.length
  console.log(`Found ${numSamples} valid samples (${lines.length - 1 - numSamples} dropped due to missing values)`)

  const features = new Float32Array(numSamples * numFeatures)
  const targets = new Float32Array(numSamples)

  for (let i = 0; i < validRows.length; i++) {
    const values = lines[validRows[i]!]!.split(',')

    for (let j = 0; j < numFeatures; j++) {
      features[i * numFeatures + j] = Number(values[featureIndices[j]!])
    }

    // Scale target to $100k for easier learning
    targets[i] = Number(values[targetIdx]) / 100000
  }

  return { features, targets, numSamples, numFeatures, featureNames: NUMERIC_FEATURES }
}

// ==================== Feature Normalization ====================

interface NormParams {
  mean: Float32Array
  std: Float32Array
}

function computeNormParams(features: Float32Array, numSamples: number, numFeatures: number): NormParams {
  const mean = new Float32Array(numFeatures)
  const std = new Float32Array(numFeatures)

  for (let i = 0; i < numSamples; i++) {
    for (let j = 0; j < numFeatures; j++) {
      mean[j] += features[i * numFeatures + j]!
    }
  }
  for (let j = 0; j < numFeatures; j++) {
    mean[j] /= numSamples
  }

  for (let i = 0; i < numSamples; i++) {
    for (let j = 0; j < numFeatures; j++) {
      const diff = features[i * numFeatures + j]! - mean[j]!
      std[j] += diff * diff
    }
  }
  for (let j = 0; j < numFeatures; j++) {
    std[j] = Math.sqrt(std[j]! / numSamples) || 1
  }

  return { mean, std }
}

function normalizeFeatures(
  features: Float32Array,
  numSamples: number,
  numFeatures: number,
  params: NormParams
): Float32Array {
  const normalized = new Float32Array(features.length)

  for (let i = 0; i < numSamples; i++) {
    for (let j = 0; j < numFeatures; j++) {
      const idx = i * numFeatures + j
      normalized[idx] = (features[idx]! - params.mean[j]!) / params.std[j]!
    }
  }

  return normalized
}

// ==================== Train/Validation Split ====================

interface SplitData {
  trainFeatures: Float32Array
  trainTargets: Float32Array
  valFeatures: Float32Array
  valTargets: Float32Array
  trainSize: number
  valSize: number
}

function trainValSplit(
  features: Float32Array,
  targets: Float32Array,
  numSamples: number,
  numFeatures: number,
  valRatio: number
): SplitData {
  const indices = Array.from({ length: numSamples }, (_, i) => i)
  for (let i = indices.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[indices[i], indices[j]] = [indices[j]!, indices[i]!]
  }

  const valSize = Math.floor(numSamples * valRatio)
  const trainSize = numSamples - valSize

  const trainFeatures = new Float32Array(trainSize * numFeatures)
  const trainTargets = new Float32Array(trainSize)
  const valFeatures = new Float32Array(valSize * numFeatures)
  const valTargets = new Float32Array(valSize)

  for (let i = 0; i < trainSize; i++) {
    const srcIdx = indices[i]!
    for (let j = 0; j < numFeatures; j++) {
      trainFeatures[i * numFeatures + j] = features[srcIdx * numFeatures + j]!
    }
    trainTargets[i] = targets[srcIdx]!
  }

  for (let i = 0; i < valSize; i++) {
    const srcIdx = indices[trainSize + i]!
    for (let j = 0; j < numFeatures; j++) {
      valFeatures[i * numFeatures + j] = features[srcIdx * numFeatures + j]!
    }
    valTargets[i] = targets[srcIdx]!
  }

  return { trainFeatures, trainTargets, valFeatures, valTargets, trainSize, valSize }
}

// ==================== Metrics ====================

function computeR2(predictions: Float32Array, targets: Float32Array): number {
  const n = predictions.length
  let meanTarget = 0
  for (let i = 0; i < n; i++) meanTarget += targets[i]!
  meanTarget /= n

  let ssRes = 0, ssTot = 0
  for (let i = 0; i < n; i++) {
    ssRes += (targets[i]! - predictions[i]!) ** 2
    ssTot += (targets[i]! - meanTarget) ** 2
  }

  return 1 - ssRes / ssTot
}

function computeRMSE(predictions: Float32Array, targets: Float32Array): number {
  const n = predictions.length
  let mse = 0
  for (let i = 0; i < n; i++) {
    mse += (predictions[i]! - targets[i]!) ** 2
  }
  return Math.sqrt(mse / n)
}

// ==================== Model Definition ====================

class MLPRegressor {
  private layers: Linear<number, number>[] = []
  private relu = new ReLU()

  constructor(inputSize: number, hiddenSizes: number[]) {
    let prevSize = inputSize

    for (const hiddenSize of hiddenSizes) {
      this.layers.push(new Linear(prevSize, hiddenSize))
      prevSize = hiddenSize
    }

    this.layers.push(new Linear(prevSize, 1))
  }

  forward(x: any): any {
    let h = x
    for (let i = 0; i < this.layers.length - 1; i++) {
      h = this.layers[i]!.forward(h)
      h = this.relu.forward(h)
    }
    h = this.layers[this.layers.length - 1]!.forward(h)
    return h
  }

  parameters(): any[] {
    const params: any[] = []
    for (const layer of this.layers) {
      params.push(...layer.parameters().map(p => p.data))
    }
    return params
  }

  toString(): string {
    return `MLP(${this.layers.map(l => l.outFeatures).join('->')})`
  }
}

// ==================== Batch Generator ====================

function* batchGenerator(
  features: Float32Array,
  targets: Float32Array,
  numSamples: number,
  numFeatures: number,
  batchSize: number,
  shuffle: boolean = true
): Generator<{ X: any; y: any; size: number }> {
  const indices = Array.from({ length: numSamples }, (_, i) => i)

  if (shuffle) {
    for (let i = indices.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1))
      ;[indices[i], indices[j]] = [indices[j]!, indices[i]!]
    }
  }

  for (let start = 0; start < numSamples; start += batchSize) {
    const size = Math.min(batchSize, numSamples - start)
    const batchFeatures = new Float32Array(size * numFeatures)
    const batchTargets = new Float32Array(size)

    for (let i = 0; i < size; i++) {
      const idx = indices[start + i]!
      for (let j = 0; j < numFeatures; j++) {
        batchFeatures[i * numFeatures + j] = features[idx * numFeatures + j]!
      }
      batchTargets[i] = targets[idx]!
    }

    const X = torch.tensor(Array.from(batchFeatures), [size, numFeatures] as const)
    const y = torch.tensor(Array.from(batchTargets), [size, 1] as const)

    yield { X, y, size }
  }
}

// ==================== Main ====================

async function main() {
  console.log('Loading dataset...')
  const dataset = loadCSV(CONFIG.dataPath)
  console.log(`Features: ${dataset.featureNames.join(', ')}\n`)

  console.log('Normalizing features...')
  const normParams = computeNormParams(dataset.features, dataset.numSamples, dataset.numFeatures)
  const normalizedFeatures = normalizeFeatures(
    dataset.features, dataset.numSamples, dataset.numFeatures, normParams
  )

  console.log('Splitting data...')
  const split = trainValSplit(
    normalizedFeatures, dataset.targets,
    dataset.numSamples, dataset.numFeatures,
    CONFIG.validationSplit
  )
  console.log(`Train: ${split.trainSize}, Val: ${split.valSize}\n`)

  const model = new MLPRegressor(dataset.numFeatures, CONFIG.hiddenSizes)
  console.log(`Model: ${model.toString()}`)

  const optimizer = new Adam(model.parameters(), { lr: CONFIG.learningRate })
  console.log(`Optimizer: Adam(lr=${CONFIG.learningRate})\n`)

  console.log(`Training for ${CONFIG.epochs} epochs...\n`)

  let bestR2 = -Infinity

  for (let epoch = 0; epoch < CONFIG.epochs; epoch++) {
    const t0 = Date.now()
    let trainLoss = 0, nBatches = 0

    for (const batch of batchGenerator(
      split.trainFeatures, split.trainTargets,
      split.trainSize, dataset.numFeatures,
      CONFIG.batchSize, true
    )) {
      torch.run(() => {
        optimizer.zeroGrad()
        const pred = model.forward(batch.X)
        const loss = mseLoss(pred, batch.y)
        loss.backward()
        optimizer.step()
        trainLoss += (loss.toArray() as Float32Array)[0]!
        nBatches++
      })
    }
    trainLoss /= nBatches

    // Validation
    const allPreds: number[] = []
    const allTargets: number[] = []

    for (const batch of batchGenerator(
      split.valFeatures, split.valTargets,
      split.valSize, dataset.numFeatures,
      CONFIG.batchSize, false
    )) {
      torch.run(() => {
        const pred = model.forward(batch.X)
        const pArr = pred.toArray() as Float32Array
        const tArr = batch.y.toArray() as Float32Array
        for (let i = 0; i < batch.size; i++) {
          allPreds.push(pArr[i]!)
          allTargets.push(tArr[i]!)
        }
      })
    }

    const r2 = computeR2(Float32Array.from(allPreds), Float32Array.from(allTargets))
    const rmse = computeRMSE(Float32Array.from(allPreds), Float32Array.from(allTargets))
    bestR2 = Math.max(bestR2, r2)

    if ((epoch + 1) % 5 === 0 || epoch === 0 || epoch === CONFIG.epochs - 1) {
      console.log(
        `Epoch ${String(epoch + 1).padStart(2)}/${CONFIG.epochs} | ` +
        `Loss: ${trainLoss.toFixed(4)} | ` +
        `R²: ${r2.toFixed(4)} | ` +
        `RMSE: $${(rmse * 100000).toFixed(0)} | ` +
        `${((Date.now() - t0) / 1000).toFixed(1)}s`
      )
    }
  }

  console.log(`\n=== Best R²: ${bestR2.toFixed(4)} ===`)

  // Sample predictions
  console.log('\nSample predictions (in $):')
  const sampleBatch = batchGenerator(
    split.valFeatures, split.valTargets,
    5, dataset.numFeatures, 5, false
  ).next().value!

  torch.run(() => {
    const preds = model.forward(sampleBatch.X)
    const pArr = preds.toArray() as Float32Array
    const tArr = sampleBatch.y.toArray() as Float32Array
    for (let i = 0; i < 5; i++) {
      const pred = (pArr[i]! * 100000).toFixed(0)
      const actual = (tArr[i]! * 100000).toFixed(0)
      console.log(`  Pred: $${pred.padStart(7)} | Actual: $${actual.padStart(7)}`)
    }
  })

  console.log('\n=== Done ===')
}

main().catch(console.error)
