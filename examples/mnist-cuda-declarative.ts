/**
 * MNIST Training Example - Declarative API with CUDA
 *
 * Demonstrates GPU-accelerated training using the declarative API.
 * This is ~3x shorter than the imperative mnist-cuda.ts example.
 *
 * Key features:
 * - Automatic CUDA availability check with graceful fallback
 * - Pipeline-based data loading with .to(cuda) for GPU transfer
 * - Config-based model creation with device parameter
 * - Declarative trainer with automatic metrics tracking
 */

import { device, cuda, Data, int64 } from '@ts-torch/core'
import type { BatchableDataset, TensorPair } from '@ts-torch/core'
import { nn } from '@ts-torch/nn'
import { MNIST } from '@ts-torch/datasets'
import { Trainer, Adam } from '@ts-torch/train'

async function main() {
  console.log('=== MNIST with Declarative API + CUDA ===\n')

  // ==================== Check CUDA ====================
  if (!cuda.isAvailable()) {
    console.log('CUDA not available, please use mnist-declarative.ts for CPU training')
    return
  }

  const gpu = device.cuda(0)
  const cpu = device.cpu()
  console.log(`Using CUDA device ${gpu.index}`)
  console.log(`CUDA devices: ${cuda.deviceCount()}\n`)

  // ==================== Load Data ====================
  console.log('Loading MNIST...')
  const mnistTrain = new MNIST('./data/mnist', true)
  const mnistTest = new MNIST('./data/mnist', false)
  await mnistTrain.load()
  await mnistTest.load()
  console.log(`Train: ${mnistTrain.length}, Test: ${mnistTest.length}\n`)

  // Adapt MNIST to BatchableDataset interface for efficient batch loading
  function wrapMNIST(mnist: MNIST): BatchableDataset<TensorPair> {
    return {
      // Single item access (slow path, used for non-batched iteration)
      getItem(index: number): TensorPair {
        const sample = mnist.get(index)
        const imageBatched = sample.image.reshape([1, 784] as const)
        const labelTensor = cpu.tensor([sample.label], [1] as const, int64)
        return { data: imageBatched, label: labelTensor }
      },

      // Batch access (fast path, avoids N tensor allocations)
      getBatch(indices: number[]): TensorPair {
        const batch = mnist.getBatchByIndices(indices)
        return { data: batch.images, label: batch.labelsTensor }
      },

      get length() {
        return mnist.length
      },
    }
  }

  const trainData = wrapMNIST(mnistTrain)
  const testData = wrapMNIST(mnistTest)

  // Create data pipelines with GPU transfer and batching
  const trainLoader = Data.pipeline(trainData).shuffle().batch(64).to(gpu)
  const testLoader = Data.pipeline(testData).batch(64).to(gpu)

  // ==================== Model on GPU ====================
  const model = nn.mlp({ device: gpu, layers: [784, 128, 64, 10] })
  console.log('Model: 784 -> 128 -> 64 -> 10\n')

  // ==================== Training ====================
  const trainer = new Trainer(model)

  console.log('Training...\n')
  const t0 = Date.now()

  await trainer.fit(trainLoader, {
    epochs: 3,
    optimizer: Adam({ lr: 1e-3 }),
    loss: 'crossEntropy',
    metrics: { loss: true, accuracy: true },
    validateOn: testLoader,
    onEpochEnd: ({ epoch, metrics, valMetrics }) => {
      const elapsed = ((Date.now() - t0) / 1000).toFixed(1)
      console.log(
        `Epoch ${epoch} [${elapsed}s] | Loss: ${metrics.loss.toFixed(4)} | ` +
          `Accuracy: ${metrics.accuracy?.toFixed(2) ?? 'N/A'}% | ` +
          `Val Accuracy: ${valMetrics?.accuracy?.toFixed(2) ?? 'N/A'}%`,
      )
    },
  })

  const totalTime = ((Date.now() - t0) / 1000).toFixed(1)
  console.log(`\nTotal time: ${totalTime}s`)

  // ==================== Final Evaluation ====================
  console.log('\nFinal Evaluation:')
  const finalMetrics = await trainer.evaluate(testLoader)
  console.log(`Accuracy: ${finalMetrics.accuracy?.toFixed(2) ?? 'N/A'}%`)

  console.log('\n=== Done ===')
}

main()
