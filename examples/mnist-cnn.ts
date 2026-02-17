/**
 * MNIST CNN Training Example
 *
 * Trains a convolutional neural network on MNIST using the builder API.
 * Demonstrates: nn.conv2d, nn.maxPool2d, nn.flatten, spatial shape inference.
 */

import { device } from '@ts-torch/core'
import { nn } from '@ts-torch/nn'
import { Data, MNIST } from '@ts-torch/datasets'
import { Trainer, Adam, loss, logger } from '@ts-torch/train'

async function main() {
  console.log('=== MNIST CNN ===\n')

  const cpu = device.cpu()

  // Load MNIST
  console.log('Loading MNIST...')
  const mnistTrain = new MNIST('./data/mnist', true)
  const mnistTest = new MNIST('./data/mnist', false)
  await mnistTrain.load()
  await mnistTest.load()
  console.log(`Train: ${mnistTrain.length}, Test: ${mnistTest.length}\n`)

  // Data pipelines — reshape flat 784 input to [1, 28, 28] for CNN
  const trainLoader = Data.pipeline(mnistTrain)
    .shuffle()
    .batch(64)
    .map((b) => ({ input: b.input.reshape([b.input.shape[0], 1, 28, 28]), target: b.target }))
  const testLoader = Data.pipeline(mnistTest)
    .batch(64)
    .map((b) => ({ input: b.input.reshape([b.input.shape[0], 1, 28, 28]), target: b.target }))

  // CNN model definition
  // Input: [1, 28, 28] -> Conv(32,3,pad=1) -> Pool(2) -> Conv(64,3,pad=1) -> Pool(2) -> Flatten -> FC(128) -> FC(10)
  // Shape: [1,28,28] -> [32,28,28] -> [32,14,14] -> [64,14,14] -> [64,7,7] -> [3136] -> [128] -> [10]
  const model = nn
    .sequence(
      nn.input([1, 28, 28]),
      nn.conv2d(32, 3, { padding: 1 }).relu(),
      nn.maxPool2d(2),
      nn.conv2d(64, 3, { padding: 1 }).relu(),
      nn.maxPool2d(2),
      nn.flatten(),
      nn.fc(128).relu().dropout(0.5),
      nn.fc(10),
    )
    .init(cpu)

  console.log(model.summary())
  console.log()

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
