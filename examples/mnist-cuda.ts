/**
 * MNIST Classification with ts-torch (CUDA) - Zero-Copy Batching
 *
 * This version uses narrow() for TRUE zero-copy GPU batching:
 * - Entire dataset transferred to GPU once
 * - narrow() returns views - NO memory allocation per batch
 * - NO CPU-GPU transfers during training
 * - Loss read only once per epoch (single sync point)
 */

import { torch } from '@ts-torch/core'
import { Linear, ReLU } from '@ts-torch/nn'
import { MNIST } from '@ts-torch/datasets'
import { SGD, crossEntropyLoss } from '@ts-torch/optim'

console.log('=== MNIST with CUDA (Zero-Copy Batching) ===\n')

// ==================== Check CUDA ====================
const cudaAvailable = torch.cuda.isAvailable()
console.log(`CUDA available: ${cudaAvailable}`)
if (!cudaAvailable) {
  console.error('CUDA not available')
  process.exit(1)
}
console.log(`CUDA devices: ${torch.cuda.deviceCount()}\n`)

// ==================== Load Dataset ====================
console.log('Loading MNIST...')
const trainData = new MNIST('./data/mnist', true)
const testData = new MNIST('./data/mnist', false)
await trainData.load()
await testData.load()
console.log(`Train: ${trainData.length}, Test: ${testData.length}\n`)

// ==================== ONE-TIME GPU Transfer ====================
console.log('Transferring to GPU (one-time)...')
const t0 = Date.now()

// Collect all data into flat arrays
const trainImages = new Float32Array(trainData.length * 784)
const trainLabels = new BigInt64Array(trainData.length)
for (let i = 0; i < trainData.length; i++) {
  const s = trainData.get(i)
  trainImages.set(s.image.toArray() as Float32Array, i * 784)
  trainLabels[i] = BigInt(s.label)
}

const testImages = new Float32Array(testData.length * 784)
const testLabels = new BigInt64Array(testData.length)
for (let i = 0; i < testData.length; i++) {
  const s = testData.get(i)
  testImages.set(s.image.toArray() as Float32Array, i * 784)
  testLabels[i] = BigInt(s.label)
}

// Create GPU tensors - these stay on GPU for entire training
let gpuTrainX: any, gpuTrainY: any, gpuTestX: any, gpuTestY: any

torch.run(() => {
  const cpuTrainX = torch.tensor(Array.from(trainImages), [trainData.length, 784] as const)
  const cpuTrainY = torch.tensor(Array.from(trainLabels).map(Number), [trainData.length] as const, torch.int64)
  gpuTrainX = cpuTrainX.cuda().escape()
  gpuTrainY = cpuTrainY.cuda().escape()

  const cpuTestX = torch.tensor(Array.from(testImages), [testData.length, 784] as const)
  const cpuTestY = torch.tensor(Array.from(testLabels).map(Number), [testData.length] as const, torch.int64)
  gpuTestX = cpuTestX.cuda().escape()
  gpuTestY = cpuTestY.cuda().escape()
})

console.log(`  Done in ${((Date.now() - t0) / 1000).toFixed(1)}s\n`)

// ==================== Model ====================
console.log('Model: 784 -> 128 -> 64 -> 10')
const fc1 = new Linear(784, 128)
const fc2 = new Linear(128, 64)
const fc3 = new Linear(64, 10)
const relu = new ReLU()

fc1.to('cuda')
fc2.to('cuda')
fc3.to('cuda')

function forward(x: any): any {
  return fc3.forward(relu.forward(fc2.forward(relu.forward(fc1.forward(x)))))
}

// ==================== Optimizer ====================
const params = [
  ...fc1.parameters().map(p => p.data),
  ...fc2.parameters().map(p => p.data),
  ...fc3.parameters().map(p => p.data),
]
const optimizer = new SGD(params, { lr: 0.01 })
console.log(`Optimizer: ${optimizer.toString()}\n`)

// ==================== Training ====================
const EPOCHS = 3
const BATCH_SIZE = 512
const numBatches = Math.ceil(trainData.length / BATCH_SIZE)

console.log(`Training: ${EPOCHS} epochs, batch ${BATCH_SIZE}, ${numBatches} batches/epoch`)
console.log('Using narrow() for zero-copy batching\n')

for (let epoch = 0; epoch < EPOCHS; epoch++) {
  const epochStart = Date.now()
  let lastLoss = 0

  // Single torch.run() for entire epoch - minimal scope overhead
  torch.run(() => {
    for (let b = 0; b < numBatches; b++) {
      const start = b * BATCH_SIZE
      const size = Math.min(BATCH_SIZE, trainData.length - start)

      // narrow() returns a VIEW - zero copy, zero allocation!
      const batchX = gpuTrainX.narrow(0, start, size)
      const batchY = gpuTrainY.narrow(0, start, size)

      optimizer.zeroGrad()
      const logits = forward(batchX)
      const loss = crossEntropyLoss(logits, batchY)
      loss.backward()
      optimizer.step()

      // Only sync at end of epoch
      if (b === numBatches - 1) {
        lastLoss = loss.item()
      }
    }
  })

  const elapsed = ((Date.now() - epochStart) / 1000).toFixed(3)
  console.log(`Epoch ${epoch + 1}/${EPOCHS} | Loss: ${lastLoss.toFixed(4)} | Time: ${elapsed}s`)
}

console.log('')

// ==================== Evaluation ====================
console.log('Evaluating...')
const evalStart = Date.now()

let correct = 0
let total = 0
const evalBatch = 1000
const numEvalBatches = Math.ceil(testData.length / evalBatch)

torch.run(() => {
  for (let b = 0; b < numEvalBatches; b++) {
    const start = b * evalBatch
    const size = Math.min(evalBatch, testData.length - start)

    const batchX = gpuTestX.narrow(0, start, size)
    const batchY = gpuTestY.narrow(0, start, size)

    const logits = forward(batchX)

    // CPU evaluation for simplicity (test set is small)
    const logitsArr = logits.cpu().toArray() as Float32Array
    const labelsArr = batchY.cpu().toArray() as BigInt64Array

    for (let i = 0; i < size; i++) {
      let maxVal = -Infinity, pred = 0
      for (let c = 0; c < 10; c++) {
        const v = logitsArr[i * 10 + c]!
        if (v > maxVal) { maxVal = v; pred = c }
      }
      if (pred === Number(labelsArr[i])) correct++
      total++
    }
  }
})

const evalTime = ((Date.now() - evalStart) / 1000).toFixed(2)
console.log(`\nAccuracy: ${((correct / total) * 100).toFixed(2)}% (${correct}/${total})`)
console.log(`Eval time: ${evalTime}s`)
console.log('\n=== Done ===')
