/**
 * DQN CartPole Example
 *
 * Demonstrates the Deep Q-Network (DQN) API on CartPole.
 * DQN is an off-policy algorithm that uses:
 * - Experience replay buffer for sample efficiency
 * - Target network for stable learning
 * - Epsilon-greedy exploration
 *
 * Note: This is a basic implementation. DQN typically requires careful
 * hyperparameter tuning for stable learning. For reliable results on
 * CartPole, consider using PPO or A2C instead.
 *
 * Run with: bun run examples/rl/dqn-cartpole.ts
 */

import { device } from '@ts-torch/core'
import { nn } from '@ts-torch/nn'
import { RL, ReplayBuffer, EpsilonGreedyStrategy } from '@ts-torch/rl'

async function main() {
  console.log('=== DQN CartPole Example ===\n')

  const dev = device.cpu()
  console.log(`Device: ${dev.type}\n`)

  // Configuration (stability-focused based on research)
  const config = {
    totalSteps: 100_000,
    batchSize: 256,         // Larger batch for smoother gradients
    bufferSize: 50_000,     // Larger buffer for better distribution
    learningStarts: 5_000,  // More initial exploration
    trainFreq: 4,           // Standard DQN
    targetUpdateFreq: 1000, // Less frequent hard updates for stability
    tau: 1.0,               // Hard updates
    gamma: 0.99,            // Standard gamma
    explorationFraction: 0.2,  // Explore for 20% of training
    evalEpisodes: 10,
  }

  // Define Q-network: observation (4) -> hidden (64x2) -> Q-values (2)
  const qNetDef = nn.sequence(4,
    nn.fc(64).relu(),
    nn.fc(64).relu(),
    nn.fc(2)
  )

  // Create DQN agent
  const agent = RL.dqn({
    device: dev,
    model: qNetDef,
    optimizer: { lr: 1e-4 },  // Lower LR with larger batch
    gamma: config.gamma,
    targetUpdateFreq: config.targetUpdateFreq,
    tau: config.tau,
    actionSpace: 2,
    maxGradNorm: 10,
  })

  // Linear epsilon decay (standard approach from CleanRL/SB3)
  const epsilonStart = 1.0
  const epsilonEnd = 0.05
  const explorationSteps = Math.floor(config.totalSteps * config.explorationFraction)

  function getEpsilon(step: number): number {
    // Linear interpolation from start to end over explorationSteps
    const progress = Math.min(1.0, step / explorationSteps)
    return epsilonStart + (epsilonEnd - epsilonStart) * progress
  }

  // Create exploration strategy (we'll manually control epsilon)
  const exploration = new EpsilonGreedyStrategy({
    start: epsilonStart,
    end: epsilonEnd,
    decay: 1.0,  // We control decay manually
  })
  agent.setStrategy(exploration)

  // Create replay buffer
  const buffer = new ReplayBuffer(config.bufferSize, 4, 1)

  // Environment
  const env = RL.envs.CartPole()

  console.log('Training DQN agent...')
  console.log(`  Steps: ${config.totalSteps}`)
  console.log(`  Buffer: ${config.bufferSize}`)
  console.log()

  // Training loop
  const t0 = Date.now()
  let obs = env.reset()
  let _episodeReward = 0
  let episodeCount = 0

  for (let step = 0; step < config.totalSteps; step++) {
    // Select action
    const action = agent.act(obs, true)

    // Step environment
    const result = env.step(action)
    const reward = typeof result.reward === 'number' ? result.reward : result.reward[0]!
    _episodeReward += reward

    // Store transition
    buffer.push({
      state: obs,
      action,
      reward,
      nextState: result.observation,
      done: result.done,
    })

    obs = result.observation

    // Handle episode end
    if (result.done) {
      episodeCount++
      obs = env.reset()
      _episodeReward = 0
    }

    // Linear epsilon decay
    exploration.setEpsilon(getEpsilon(step))

    // Train
    if (step >= config.learningStarts && step % config.trainFreq === 0) {
      const batch = buffer.sample(config.batchSize)
      const result = agent.trainStep(batch)

      // Log loss periodically
      if (step % 2000 === 0) {
        console.log(`    Loss: ${result.loss.toFixed(6)}`)
      }
    }

    // Progress
    if ((step + 1) % 2000 === 0) {
      const elapsed = ((Date.now() - t0) / 1000).toFixed(1)
      console.log(`  Step ${step + 1}/${config.totalSteps} (${elapsed}s, eps=${exploration.currentEpsilon.toFixed(3)})`)
    }
  }

  console.log(`\nTraining completed in ${((Date.now() - t0) / 1000).toFixed(1)}s`)
  console.log(`Episodes: ${episodeCount}`)

  // Evaluate
  console.log(`\n--- Evaluation (${config.evalEpisodes} episodes) ---\n`)

  const evalRewards: number[] = []
  for (let ep = 0; ep < config.evalEpisodes; ep++) {
    const evalEnv = RL.envs.CartPole()
    let totalReward = 0
    let evalObs = evalEnv.reset()

    for (let s = 0; s < 500; s++) {
      const action = agent.act(evalObs, false)  // Greedy
      const result = evalEnv.step(action)
      totalReward += typeof result.reward === 'number' ? result.reward : result.reward[0]!
      if (result.done) break
      evalObs = result.observation
    }

    evalRewards.push(totalReward)
    console.log(`  Episode ${ep + 1}: ${totalReward} reward`)
  }

  const meanReward = evalRewards.reduce((a, b) => a + b, 0) / evalRewards.length
  console.log(`\nMean reward: ${meanReward.toFixed(1)}`)

  if (meanReward >= 200) {
    console.log('DQN is learning!')
  } else {
    console.log('Note: DQN may need hyperparameter tuning. Try PPO/A2C for reliable results.')
  }

  console.log('\n=== Done ===')
}

main().catch(console.error)
