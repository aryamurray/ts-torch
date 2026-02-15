#!/usr/bin/env npx tsx
// ─────────────────────────────────────────────────────────────
// Demo — simulate a PPO training run with the dashboard
// ─────────────────────────────────────────────────────────────

import { Dashboard } from './dashboard.js'

const dash = new Dashboard({ title: 'ts-torch PPO CartPole' })
dash.start()

// Simulated training state
let step = 0
const totalSteps = 500_000
let loss = 2.5
let reward = 0
let entropy = 1.5
let valueLoss = 1.0
let episode = 0

function simulateStep() {
  step += 256 // batch size worth of steps
  episode += Math.random() > 0.7 ? 1 : 0

  // Simulate converging metrics
  const progress = step / totalSteps
  loss = Math.max(0.01, loss - 0.003 * Math.random() + 0.001 * Math.random())
  reward = Math.min(500, reward + (1.5 + progress * 3) * Math.random() - 0.5)
  entropy = Math.max(0.1, entropy - 0.001 * Math.random())
  valueLoss = Math.max(0.01, valueLoss - 0.002 * Math.random() + 0.0005 * Math.random())

  // Push numeric metrics
  dash.numericMetrics.push('Loss', 'train', loss + (Math.random() - 0.5) * 0.1)
  dash.numericMetrics.push('Reward', 'train', reward + (Math.random() - 0.5) * 20)
  dash.numericMetrics.push('Entropy', 'train', entropy + (Math.random() - 0.5) * 0.05)
  dash.numericMetrics.push('Value Loss', 'train', valueLoss + (Math.random() - 0.5) * 0.05)

  // Occasional validation
  if (step % 10000 < 256) {
    dash.numericMetrics.push('Loss', 'valid', loss * 1.1 + (Math.random() - 0.5) * 0.15)
    dash.numericMetrics.push('Reward', 'valid', reward * 0.9 + (Math.random() - 0.5) * 15)
  }

  // Push text metrics
  dash.textMetrics.push('Episodes', 'train', episode.toString())
  dash.textMetrics.push('Steps/s', 'train', `${(18000 + Math.random() * 2000).toFixed(0)}`)

  // Update progress
  dash.progress.update('train', step / totalSteps, (step % 50000) / 50000)

  // Update status
  dash.status.update('train', [
    { tag: 'Epoch', value: `${Math.floor(step / 50000) + 1}/10` },
    { tag: 'Step', value: `${step}/${totalSteps}` },
    { tag: 'LR', value: '3.0e-4' },
  ])

  if (step < totalSteps) {
    setTimeout(simulateStep, 20)
  } else {
    setTimeout(() => dash.destroy(), 2000)
  }
}

simulateStep()
