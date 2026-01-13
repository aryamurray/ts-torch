/**
 * MNIST Classification with ts-torch (CUDA)
 *
 * Trains a simple MLP on the MNIST handwritten digits dataset
 * using proper autograd backpropagation and SGD optimizer.
 *
 * This version runs on GPU using CUDA.
 */

import { torch } from '@ts-torch/core'
import { Linear, ReLU } from '@ts-torch/nn'
import { MNIST } from '@ts-torch/datasets'
import { SGD, crossEntropyLoss } from '@ts-torch/optim'

console.log('=== MNIST Classification with CUDA ===\n')

// ==================== Check CUDA Availability ====================
const cudaAvailable = torch.cuda.isAvailable()
console.log(`CUDA available: ${cudaAvailable}`)

if (!cudaAvailable) {
  console.error('CUDA is not available. Please ensure:')
  console.error('  1. You have an NVIDIA GPU')
  console.error('  2. CUDA toolkit is installed')
  console.error('  3. You built with: bun run build:native:cuda')
  process.exit(1)
}

const deviceCount = torch.cuda.deviceCount()
console.log(`CUDA devices: ${deviceCount}`)
console.log('')

// ==================== Load Dataset ====================
console.log('Loading MNIST dataset...')

const trainData = new MNIST('./data/mnist', true)
const testData = new MNIST('./data/mnist', false)

await trainData.load()
await testData.load()

console.log(`Training samples: ${trainData.length}`)
console.log(`Test samples: ${testData.length}`)
console.log('')

// ==================== Define Model ====================
console.log('Creating model: 784 -> 128 -> 64 -> 10')

// Create layers (weights automatically have requires_grad=true via Parameter class)
const fc1 = new Linear(784, 128)
const fc2 = new Linear(128, 64)
const fc3 = new Linear(64, 10)
const relu = new ReLU()

// Move model to CUDA
console.log('Moving model to CUDA...')
fc1.to('cuda')
fc2.to('cuda')
fc3.to('cuda')

console.log('  ' + fc1.toString())
console.log('  ' + fc2.toString())
console.log('  ' + fc3.toString())
console.log('')

// Forward pass function
function forward(x: any): any {
  let h: any = fc1.forward(x)
  h = relu.forward(h)
  h = fc2.forward(h)
  h = relu.forward(h)
  h = fc3.forward(h)
  return h
}

// ==================== Setup Optimizer ====================
// Collect all parameter tensors from the model
const allParams = [
  ...fc1.parameters().map((p) => p.data),
  ...fc2.parameters().map((p) => p.data),
  ...fc3.parameters().map((p) => p.data),
]

const LEARNING_RATE = 0.01
const optimizer = new SGD(allParams, { lr: LEARNING_RATE })

console.log(`Optimizer: ${optimizer.toString()}`)
console.log('')

// ==================== Training ====================
const EPOCHS = 3
const BATCH_SIZE = 64

console.log(`Training for ${EPOCHS} epochs, batch size ${BATCH_SIZE}`)
console.log('')

for (let epoch = 0; epoch < EPOCHS; epoch++) {
  let totalLoss = 0
  let numBatches = 0
  let correct = 0
  let total = 0

  const startTime = Date.now()
  console.log(`Starting epoch ${epoch + 1}...`)

  for (const batch of trainData.batches(BATCH_SIZE, true)) {
    torch.run(() => {
      // Move batch data to CUDA
      const images = batch.images.cuda()
      const labels = batch.labelsTensor.cuda()

      // Zero gradients
      optimizer.zeroGrad()

      // Forward pass
      const logits = forward(images)

      // Compute cross-entropy loss
      const loss = crossEntropyLoss(logits as any, labels as any)

      // Backward pass - compute gradients
      loss.backward()

      // Update weights
      optimizer.step()

      // Track loss (move to CPU for reading)
      const lossArray = loss.cpu().toArray() as Float32Array
      totalLoss += lossArray[0] ?? 0
      numBatches++

      // Compute accuracy (move to CPU for reading)
      const probs = logits.softmax(1)
      const probsArray = probs.cpu().toArray() as Float32Array

      for (let i = 0; i < batch.labels.length; i++) {
        const label = batch.labels[i]!

        // Argmax prediction
        let maxProb = -1
        let pred = 0
        for (let c = 0; c < 10; c++) {
          const p = probsArray[i * 10 + c]!
          if (p > maxProb) {
            maxProb = p
            pred = c
          }
        }
        if (pred === label) correct++
        total++
      }
    })

    // Progress indicator every 100 batches
    if (numBatches % 100 === 0) {
      process.stdout.write(`\r  Batch ${numBatches}/${Math.ceil(trainData.length / BATCH_SIZE)}`)
    }
  }
  console.log() // New line after progress

  const elapsed = ((Date.now() - startTime) / 1000).toFixed(1)
  const avgLoss = (totalLoss / numBatches).toFixed(4)
  const accuracy = ((correct / total) * 100).toFixed(2)

  console.log(`Epoch ${epoch + 1}/${EPOCHS} | Loss: ${avgLoss} | Train Acc: ${accuracy}% | Time: ${elapsed}s`)
}

console.log('')

// ==================== Evaluation ====================
console.log('Evaluating on test set...')

let testCorrect = 0
let testTotal = 0

for (const batch of testData.batches(1000)) {
  torch.run(() => {
    // Move batch to CUDA
    const images = batch.images.cuda()

    const logits = forward(images)
    const probs = logits.softmax(1)
    const probsArray = probs.cpu().toArray() as Float32Array

    for (let i = 0; i < batch.labels.length; i++) {
      const label = batch.labels[i]!

      // Argmax prediction
      let maxProb = -1
      let pred = 0
      for (let c = 0; c < 10; c++) {
        const p = probsArray[i * 10 + c]!
        if (p > maxProb) {
          maxProb = p
          pred = c
        }
      }

      if (pred === label) testCorrect++
      testTotal++
    }
  })
}

const testAccuracy = ((testCorrect / testTotal) * 100).toFixed(2)
console.log(`\nTest Accuracy: ${testAccuracy}%`)
console.log(`Correct: ${testCorrect} / ${testTotal}`)

console.log('\n=== Done ===')
