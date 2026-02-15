#!/usr/bin/env npx tsx
// Simulates a ~10s training run with the dashboard to see it in real-time

import { Dashboard } from './dashboard.js'

const dash = new Dashboard({ title: 'ts-torch MNIST' })
dash.start()

const EPOCHS = 5
const BATCHES_PER_EPOCH = 100
const BATCH_DELAY_MS = 20 // 100 batches * 20ms = 2s per epoch, ~10s total

let loss = 2.3
let valLoss = 2.5
let accuracy = 12
let valAccuracy = 10
let lr = 1e-3

async function sleep(ms: number) {
  return new Promise((r) => setTimeout(r, ms))
}

async function run() {
  for (let epoch = 1; epoch <= EPOCHS; epoch++) {
    if (dash.quitRequested) break

    // onEpochStart
    dash.status.update('train', [
      { tag: 'Epoch', value: `${epoch}/${EPOCHS}` },
      { tag: 'LR', value: lr.toExponential(1) },
    ])

    // Training batches
    for (let batch = 0; batch < BATCHES_PER_EPOCH; batch++) {
      if (dash.quitRequested) break

      // Simulate loss decreasing with noise
      const progress = ((epoch - 1) * BATCHES_PER_EPOCH + batch) / (EPOCHS * BATCHES_PER_EPOCH)
      loss = Math.max(0.05, loss - 0.004 * Math.random() + 0.001 * Math.random())
      accuracy = Math.min(98, accuracy + 0.15 * Math.random() + progress * 0.05)

      // Push per-batch metrics
      dash.numericMetrics.push('Loss', 'train', loss + (Math.random() - 0.5) * 0.1)

      // Progress
      const taskProgress = (batch + 1) / BATCHES_PER_EPOCH
      const totalProgress = (epoch - 1) / EPOCHS + taskProgress / EPOCHS
      dash.progress.update('train', totalProgress, taskProgress)

      // Simulate console.log that would normally corrupt the TUI
      if (batch === 50) {
        console.log(`[debug] epoch ${epoch} batch ${batch} — this gets buffered!`)
      }

      await sleep(BATCH_DELAY_MS)
    }

    if (dash.quitRequested) break

    // Epoch-level text metrics
    dash.textMetrics.push('Loss', 'train', loss.toFixed(4))
    dash.textMetrics.push('Accuracy', 'train', `${accuracy.toFixed(2)}%`)
    dash.numericMetrics.push('Accuracy', 'train', accuracy)

    // Validation
    dash.status.update('valid', [{ tag: 'Epoch', value: `${epoch}/${EPOCHS}` }])
    await sleep(200) // simulate validation time

    valLoss = Math.max(0.08, valLoss - 0.003 * Math.random() * epoch + 0.001 * Math.random())
    valAccuracy = Math.min(97, valAccuracy + 0.12 * Math.random() * epoch)

    dash.numericMetrics.push('Loss', 'valid', valLoss)
    dash.numericMetrics.push('Accuracy', 'valid', valAccuracy)
    dash.textMetrics.push('Loss', 'valid', valLoss.toFixed(4))
    dash.textMetrics.push('Accuracy', 'valid', `${valAccuracy.toFixed(2)}%`)

    // Back to training status
    dash.status.update('train', [
      { tag: 'Epoch', value: `${epoch}/${EPOCHS}` },
      { tag: 'LR', value: lr.toExponential(1) },
    ])
    dash.progress.update('train', epoch / EPOCHS, 1)

    // Decay LR
    lr *= 0.8
  }

  dash.status.update('train', [{ tag: 'Status', value: 'Complete' }])
  dash.progress.update('train', 1, 1)

  // Hold the final screen for 2s so you can see it
  await sleep(2000)
  dash.destroy()
}

run()
