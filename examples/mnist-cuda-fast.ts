/**
 * MNIST Training Example - FAST version with native batching
 *
 * Uses MNIST's built-in batch generator for maximum performance.
 * Avoids per-item tensor creation overhead.
 */

import { device, cuda } from '@ts-torch/core'
import { nn } from '@ts-torch/nn'
import { MNIST } from '@ts-torch/datasets'
import { Trainer, Adam } from '@ts-torch/train'

async function main() {
  console.log('=== MNIST FAST - Native Batching + CUDA ===\n')

  if (!cuda.isAvailable()) {
    console.log('CUDA not available')
    return
  }

  const gpu = device.cuda(0)
  console.log(`Using CUDA device ${gpu.index}\n`)

  // Load MNIST
  console.log('Loading MNIST...')
  const mnistTrain = new MNIST('./data/mnist', true)
  const mnistTest = new MNIST('./data/mnist', false)
  await mnistTrain.load()
  await mnistTest.load()
  console.log(`Train: ${mnistTrain.length}, Test: ${mnistTest.length}\n`)

  // Create reusable async iterables that use MNIST's native batching
  // Returns a fresh iterable each time (generators are one-time use)
  function createLoader(mnist: MNIST, batchSize: number, shuffle: boolean) {
    return {
      [Symbol.asyncIterator]: async function* () {
        for (const batch of mnist.batches(batchSize, shuffle)) {
          // Transfer to GPU - single transfer per batch instead of 64
          const gpuImages = batch.images.move('cuda')
          const gpuLabels = batch.labelsTensor.move('cuda')

          yield {
            data: gpuImages,
            label: gpuLabels,
          }
        }
      },
    }
  }

  // Model
  const model = nn.mlp({ device: gpu, layers: [784, 128, 64, 10] })
  console.log('Model: 784 -> 128 -> 64 -> 10\n')

  // Training
  const trainer = new Trainer(model)

  console.log('Training...\n')
  const t0 = Date.now()

  await trainer.fit(createLoader(mnistTrain, 64, true), {
    epochs: 3,
    optimizer: Adam({ lr: 1e-3 }),
    loss: 'crossEntropy',
    metrics: { loss: true, accuracy: true },
    validateOn: createLoader(mnistTest, 64, false),
    onEpochEnd: ({ epoch, metrics, valMetrics }) => {
      const elapsed = ((Date.now() - t0) / 1000).toFixed(1)
      console.log(
        `Epoch ${epoch} [${elapsed}s] | Loss: ${metrics.loss.toFixed(4)} | ` +
          `Acc: ${metrics.accuracy?.toFixed(2)}% | Val: ${valMetrics?.accuracy?.toFixed(2)}%`,
      )
    },
  })

  const totalTime = ((Date.now() - t0) / 1000).toFixed(1)
  console.log(`\nTotal time: ${totalTime}s`)

  // Final evaluation
  console.log('\nFinal Evaluation:')
  const finalMetrics = await trainer.evaluate(createLoader(mnistTest, 64, false))
  console.log(`Accuracy: ${finalMetrics.accuracy?.toFixed(2)}%`)

  console.log('\n=== Done ===')
}

main()
