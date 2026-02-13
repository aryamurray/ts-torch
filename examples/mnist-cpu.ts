/**
 * MNIST Training Example - CPU
 *
 * Trains a simple MLP on MNIST using CPU.
 */

import { device } from '@ts-torch/core'
import { nn } from '@ts-torch/nn'
import { Data, MNIST } from '@ts-torch/datasets'
import { Trainer, Adam, loss, logger } from '@ts-torch/train'

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
  // Batches already yield { input, target } — no .map() needed
  const trainLoader = Data.pipeline(mnistTrain).shuffle().batch(64)
  const testLoader = Data.pipeline(mnistTest).batch(64)

  // Model definition (no memory allocated yet)
  const model = nn.sequence(
    nn.input(784),
    nn.fc(128).relu(),
    nn.fc(64).relu(),
    nn.fc(10)
  ).init(cpu)

  console.log('Model: 784 -> 128 -> 64 -> 10\n')

  // Training
  const trainer = new Trainer({
    model,
    data: trainLoader,
    epochs: 3,
    optimizer: Adam({ lr: 1e-3 }),
    loss: loss.crossEntropy(),
    metrics: ['loss', 'accuracy'],
    validation: testLoader,
    callbacks: [logger.console()],
  })

  const history = await trainer.fit()

  console.log(`\nTotal time: ${history.totalTime.toFixed(1)}s`)

  // Final evaluation
  console.log('\nFinal Evaluation:')
  const finalMetrics = await trainer.evaluate(testLoader)
  console.log(`Accuracy: ${finalMetrics.accuracy?.toFixed(2)}%`)

  console.log('\n=== Done ===')
}

main()
