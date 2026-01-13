/**
 * MNIST Training Example - CPU
 *
 * Trains a simple MLP on MNIST using CPU.
 */

import { device } from '@ts-torch/core'
import { nn } from '@ts-torch/nn'
import { Data, MNIST } from '@ts-torch/datasets'
import { Trainer, Adam } from '@ts-torch/train'

async function main() {
  console.log('=== MNIST - CPU ===\n')

  const cpu = device.cpu()

  // Load MNIST
  console.log('Loading MNIST...')
  const mnistTrain = new MNIST('./data/mnist', true)
  const mnistTest = new MNIST('./data/mnist', false)
  await mnistTrain.load()
  await mnistTest.load()
  console.log(`Train: ${mnistTrain.length}, Test: ${mnistTest.length}\n`)

  // Data pipelines (lazy - no work until iteration)
  const trainLoader = Data.pipeline(mnistTrain)
    .shuffle()
    .batch(64)
    .map(b => ({ data: b.images, label: b.labelsTensor }))

  const testLoader = Data.pipeline(mnistTest)
    .batch(64)
    .map(b => ({ data: b.images, label: b.labelsTensor }))

  // Model definition (no memory allocated yet)
  const config = nn.sequence(784,
    nn.fc(128).relu(),
    nn.fc(64).relu(),
    nn.fc(10)
  )

  // Initialize model on CPU (memory allocated here)
  const model = config.init(cpu)
  console.log('Model: 784 -> 128 -> 64 -> 10\n')

  // Training
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
          `Acc: ${metrics.accuracy?.toFixed(2)}% | Val: ${valMetrics?.accuracy?.toFixed(2)}%`,
      )
    },
  })

  const totalTime = ((Date.now() - t0) / 1000).toFixed(1)
  console.log(`\nTotal time: ${totalTime}s`)

  // Final evaluation
  console.log('\nFinal Evaluation:')
  const finalMetrics = await trainer.evaluate(testLoader)
  console.log(`Accuracy: ${finalMetrics.accuracy?.toFixed(2)}%`)

  console.log('\n=== Done ===')
}

main()
