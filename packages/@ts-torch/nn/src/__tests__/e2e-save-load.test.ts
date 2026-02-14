/**
 * End-to-end test: Train MNIST → save → load → verify inference matches
 *
 * Uses real MNIST data and real FFI-backed tensors.
 */

import { describe, test, expect, afterEach } from 'vitest'
import { device, run } from '@ts-torch/core'
import { nn } from '../builders.js'
import { Trainer, Adam, loss } from '@ts-torch/train'
import { Data, MNIST } from '@ts-torch/datasets'
import { mkdtemp, rm } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'

const cpu = device.cpu()
const MNIST_ROOT = join(__dirname, '../../../../..', 'data/mnist')

let tempDirs: string[] = []

async function createTempDir(): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), 'ts-torch-e2e-'))
  tempDirs.push(dir)
  return dir
}

afterEach(async () => {
  for (const dir of tempDirs) {
    await rm(dir, { recursive: true, force: true }).catch(() => {})
  }
  tempDirs = []
})

describe('E2E: train → save → load → inference', () => {
  test('directory save: trained model matches loaded model', async () => {
    const dir = await createTempDir()
    const modelDir = join(dir, 'mnist-model')

    // Load a small subset of MNIST test data for fast inference check
    const mnistTest = new MNIST(MNIST_ROOT, false)
    await mnistTest.load()
    const testLoader = Data.pipeline(mnistTest).batch(32)

    // Define model config
    const config = nn.sequence(
      nn.input(784),
      nn.fc(32).relu(),
      nn.fc(10),
    )

    // Train for 1 epoch on a small subset of training data
    const mnistTrain = new MNIST(MNIST_ROOT, true)
    await mnistTrain.load()
    const trainLoader = Data.pipeline(mnistTrain).batch(64)

    const model = config.init(cpu)

    const trainer = new Trainer({
      model,
      data: trainLoader,
      epochs: 1,
      optimizer: Adam({ lr: 1e-3 }),
      loss: loss.crossEntropy(),
    })

    await trainer.fit()

    // Grab predictions from the trained model on a few test batches
    const originalOutputs: number[][] = []
    let batchCount = 0
    for await (const batch of testLoader) {
      if (batchCount >= 3) break
      const output = run(() => {
        const pred = (model as any).forward(batch.input)
        const data = Array.from(pred.toArray() as Float32Array)
        return data
      })
      originalOutputs.push(output)
      batchCount++
    }

    // Save the trained model to directory
    await model.save(modelDir, { epoch: 1 })

    // Load via nn.load
    const { model: loadedModel, metadata } = await nn.load(cpu, modelDir)
    expect(metadata.epoch).toBe(1)

    // Run the same test batches through the loaded model
    const loadedOutputs: number[][] = []
    batchCount = 0
    for await (const batch of testLoader) {
      if (batchCount >= 3) break
      const output = run(() => {
        const pred = (loadedModel as any).forward(batch.input)
        const data = Array.from(pred.toArray() as Float32Array)
        return data
      })
      loadedOutputs.push(output)
      batchCount++
    }

    // Verify outputs match exactly
    expect(loadedOutputs.length).toBe(originalOutputs.length)
    for (let i = 0; i < originalOutputs.length; i++) {
      expect(loadedOutputs[i]!.length).toBe(originalOutputs[i]!.length)
      for (let j = 0; j < originalOutputs[i]!.length; j++) {
        expect(loadedOutputs[i]![j]).toBeCloseTo(originalOutputs[i]![j]!, 5)
      }
    }
  }, 120_000) // 2 minute timeout for training
})
