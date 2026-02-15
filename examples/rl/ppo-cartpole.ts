/**
 * PPO CartPole Example
 *
 * Trains a PPO agent on the classic CartPole-v1 environment.
 * The goal is to balance a pole on a cart by applying left/right forces.
 *
 * This demonstrates:
 * - Creating vectorized environments for parallel rollout collection
 * - Configuring PPO with custom hyperparameters
 * - Training and monitoring progress
 * - Running inference with the trained policy
 *
 * Run with: bun run ppo-cartpole.ts
 */

import { device } from '@ts-torch/core'
import { RL } from '@ts-torch/rl'

async function main() {
  console.log('=== PPO CartPole Example ===\n')

  // Use CPU for this example (change to device.cuda(0) if GPU available)
  const dev = device.cpu()
  console.log(`Device: ${dev.type}\n`)

  // Create vectorized environment with 4 parallel instances
  // More environments = faster rollout collection
  const nEnvs = 4
  const vecEnv = RL.vecEnv({
    env: RL.envs.CartPole(),
    nEnvs,
    type: 'dummy',
  })
  console.log(`Environment: CartPole (${nEnvs} parallel instances)`)
  console.log(`Observation space: 4 (x, x_dot, theta, theta_dot)`)
  console.log(`Action space: 2 (left, right)\n`)

  // Create PPO agent with tuned hyperparameters for CartPole
  const ppo = RL.ppo({
    // Policy network architecture
    policy: {
      netArch: {
        pi: [64, 64], // Policy network: 2 hidden layers of 64 units
        vf: [64, 64], // Value network: 2 hidden layers of 64 units
      },
      activation: 'tanh', // Tanh works well for CartPole
    },
    // Training hyperparameters
    learningRate: 3e-4, // Adam learning rate
    nSteps: 2048, // Steps per rollout per env (total = nSteps * nEnvs)
    batchSize: 64, // Minibatch size for updates
    nEpochs: 10, // Epochs per rollout
    gamma: 0.99, // Discount factor
    gaeLambda: 0.95, // GAE lambda for advantage estimation
    clipRange: 0.2, // PPO clip range
    entCoef: 0.0, // Entropy coefficient (0 for CartPole)
    vfCoef: 0.5, // Value function coefficient
    maxGradNorm: 0.5, // Gradient clipping
    verbose: 1, // Print progress
  }).init(dev, vecEnv)

  // Train for 100k timesteps (should solve CartPole)
  const totalTimesteps = 100_000
  console.log(`Training for ${totalTimesteps.toLocaleString()} timesteps...\n`)

  const t0 = Date.now()

  await ppo.learn({
    totalTimesteps,
    logInterval: 5, // Log every 5 rollouts
    dashboard: true,
  })

  const elapsed = ((Date.now() - t0) / 1000).toFixed(1)
  console.log(`\nTraining completed in ${elapsed}s`)
  console.log(`Total timesteps: ${ppo.numTimesteps.toLocaleString()}`)

  // Run inference demo - evaluate the trained policy
  console.log('\n--- Inference Demo ---\n')
  console.log('Running 5 evaluation episodes...\n')

  const evalResults: { steps: number; reward: number }[] = []

  for (let ep = 0; ep < 5; ep++) {
    // Create a fresh environment for evaluation
    const evalEnv = RL.envs.CartPole()

    let totalReward = 0
    let steps = 0
    let obs = evalEnv.reset()

    while (steps < 500) {
      // Get action from trained policy (deterministic for evaluation)
      const action = ppo.predict(obs, true)

      // Step the environment
      const result = evalEnv.step(action as number)
      // Handle both number and Float32Array rewards
      const reward = typeof result.reward === 'number' ? result.reward : result.reward[0]!
      totalReward += reward
      steps++

      if (result.done) {
        break
      }

      obs = result.observation
    }

    evalResults.push({ steps, reward: totalReward })
    console.log(`  Episode ${ep + 1}: ${steps} steps, reward = ${totalReward}`)
  }

  // Compute statistics
  const meanReward = evalResults.reduce((sum, r) => sum + r.reward, 0) / evalResults.length
  const meanSteps = evalResults.reduce((sum, r) => sum + r.steps, 0) / evalResults.length

  console.log(`\nEvaluation Results:`)
  console.log(`  Mean reward: ${meanReward.toFixed(1)}`)
  console.log(`  Mean steps:  ${meanSteps.toFixed(1)}`)

  // A solved CartPole typically achieves 500 steps (max episode length)
  if (meanReward >= 475) {
    console.log('\nCartPole SOLVED! (mean reward >= 475)')
  } else if (meanReward >= 200) {
    console.log('\nGood progress! Try more training for consistent 500-step episodes.')
  } else {
    console.log('\nNote: May need more training timesteps.')
  }

  console.log('\n=== Done ===')
}

main().catch(console.error)
