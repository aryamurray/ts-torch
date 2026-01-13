/**
 * MNIST Classification - Declarative API
 *
 * This example demonstrates the new declarative API for ts-torch.
 * Compare to mnist-cuda.ts to see the difference in verbosity.
 *
 * ~50 lines vs ~180 lines - same functionality.
 */

import { device, Data, int64 } from '@ts-torch/core'
import type { Dataset, TensorPair } from '@ts-torch/core'
import { nn } from '@ts-torch/nn'
import { MNIST } from '@ts-torch/datasets'
import { Trainer, Adam, type EpochContext } from '@ts-torch/train'

// ==================== Device ====================
const cuda = device.cuda(0)
const cpu = device.cpu()
console.log('=== MNIST with Declarative API ===\n')

// ==================== Data ====================
console.log('Loading MNIST...')
const mnistTrain = new MNIST('./data/mnist', true)
const mnistTest = new MNIST('./data/mnist', false)
await mnistTrain.load()
await mnistTest.load()
console.log(`Train: ${mnistTrain.length}, Test: ${mnistTest.length}\n`)

// Adapt MNIST to Dataset interface
function wrapMNIST(mnist: MNIST): Dataset<TensorPair> {
  return {
    getItem(index: number): TensorPair {
      const sample = mnist.get(index)
      // Reshape image to [1, 784] - model expects batch dimension
      const imageBatched = sample.image.reshape([1, 784] as const)
      // Create label tensor as [1] shape int64 for cross_entropy_loss
      const labelTensor = cpu.tensor([sample.label], [1] as const, int64)
      // Return as TensorPair - tensors have different dtypes (float32 vs int64)
      const pair: TensorPair = {
        data: imageBatched,
        label: labelTensor,
      }
      return pair
    },
    get length() {
      return mnist.length
    },
  }
}

const trainData = wrapMNIST(mnistTrain)
const testData = wrapMNIST(mnistTest)

// Create data pipelines - each item is already batched (batch size 1)
// Note: True batching with tensor stacking will be added to pipeline in future
const trainLoader = Data.pipeline(trainData).shuffle().to(cuda)
const testLoader = Data.pipeline(testData).to(cuda)

// ==================== Model ====================
const model = nn.mlp({ device: cuda, layers: [784, 128, 64, 10] })
console.log('Model: 784 -> 128 -> 64 -> 10\n')

// ==================== Training ====================
const trainer = new Trainer(model)

console.log('Training...\n')
const history = await trainer.fit(trainLoader, {
  epochs: 3,
  optimizer: Adam({ lr: 1e-3 }),
  loss: 'crossEntropy',
  metrics: { loss: true, accuracy: true },
  validateOn: testLoader,
  onEpochEnd: ({ epoch, metrics, valMetrics }) => {
    console.log(
      `Epoch ${epoch} | Loss: ${metrics.loss.toFixed(4)} | ` +
        `Accuracy: ${metrics.accuracy?.toFixed(2) ?? 'N/A'}% | ` +
        `Val Accuracy: ${valMetrics?.accuracy?.toFixed(2) ?? 'N/A'}%`,
    )
  },
})

// ==================== Final Evaluation ====================
console.log('\nFinal Evaluation:')
const finalMetrics = await trainer.evaluate(testLoader)
console.log(`Accuracy: ${finalMetrics.accuracy?.toFixed(2) ?? 'N/A'}%`)

console.log('\n=== Done ===')
