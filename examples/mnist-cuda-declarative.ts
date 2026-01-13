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
import type { Dataset, TensorPair } from '@ts-torch/core'
import { nn } from '@ts-torch/nn'
import { MNIST } from '@ts-torch/datasets'
import { Trainer, Adam, type EpochContext } from '@ts-torch/train'

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

  // Create data pipelines with GPU transfer
  const trainLoader = Data.pipeline(trainData).shuffle().to(gpu)
  const testLoader = Data.pipeline(testData).to(gpu)

  // ==================== Model on GPU ====================
  const model = nn.mlp(gpu, [784, 128, 64, 10])
  console.log('Model: 784 -> 128 -> 64 -> 10\n')

  // ==================== Training ====================
  const trainer = new Trainer(model)

  console.log('Training...\n')
  await trainer.fit(trainLoader, {
    epochs: 3,
    optimizer: Adam({ lr: 1e-3 }),
    loss: 'crossEntropy',
    metrics: { loss: true, accuracy: true },
    validateOn: testLoader,
    onEpochEnd: ({ epoch, metrics, valMetrics }: EpochContext) => {
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
}

main()
