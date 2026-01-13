/**
 * MNIST Training Example - FAST CPU version
 */

import { device } from '@ts-torch/core'
import { nn } from '@ts-torch/nn'
import { MNIST } from '@ts-torch/datasets'
import { Trainer, Adam } from '@ts-torch/train'

async function main() {
  console.log('=== MNIST FAST - CPU ===\n')

  const cpu = device.cpu()

  // Load MNIST
  console.log('Loading MNIST...')
  const mnistTrain = new MNIST('./data/mnist', true)
  const mnistTest = new MNIST('./data/mnist', false)
  await mnistTrain.load()
  await mnistTest.load()
  console.log(`Train: ${mnistTrain.length}, Test: ${mnistTest.length}\n`)

  // Create async iterables using MNIST's native batching (stays on CPU)
  function createLoader(mnist: MNIST, batchSize: number, shuffle: boolean) {
    return {
      [Symbol.asyncIterator]: async function* () {
        for (const batch of mnist.batches(batchSize, shuffle)) {
          yield {
            data: batch.images,
            label: batch.labelsTensor,
          }
        }
      },
    }
  }

  // Model on CPU
  const model = nn.mlp({ device: cpu, layers: [784, 128, 64, 10] })
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

  console.log('\n=== Done ===')
}

main()
