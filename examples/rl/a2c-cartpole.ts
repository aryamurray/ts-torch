/**
 * A2C CartPole Example
 *
 * Trains an Advantage Actor-Critic (A2C) agent on CartPole-v1.
 * A2C is simpler than PPO - single gradient step per rollout, no clipping.
 *
 * Key differences from PPO:
 * - Uses shorter rollouts (nSteps=5 vs PPO's 2048)
 * - Single gradient update per rollout (no epochs/minibatches)
 * - Faster per-update but may need more environment steps
 *
 * Run with: bun run examples/rl/a2c-cartpole.ts
 */

import { device } from '@ts-torch/core'
import { RL } from '@ts-torch/rl'

async function main() {
  console.log('=== A2C CartPole Example ===\n')

  const dev = device.cpu()
  console.log(`Device: ${dev.type}\n`)

  // Create vectorized environment
  // A2C typically uses more parallel envs with shorter rollouts
  const nEnvs = 16
  const vecEnv = RL.vecEnv({
    env: RL.envs.CartPole(),
    nEnvs,
    type: 'dummy'
  })
  console.log(`Environment: CartPole (${nEnvs} parallel instances)`)
  console.log(`Observation space: 4 (x, x_dot, theta, theta_dot)`)
  console.log(`Action space: 2 (left, right)\n`)

  // Create A2C agent with typical hyperparameters
  const a2c = RL.a2c({
    policy: {
      netArch: {
        pi: [64, 64],  // Policy network
        vf: [64, 64],  // Value network
      },
      activation: 'tanh',
    },
    // A2C typically uses shorter rollouts with more envs
    learningRate: 7e-4,
    nSteps: 5,            // Short rollouts (vs PPO's 2048)
    gamma: 0.99,
    gaeLambda: 1.0,       // A2C often uses lambda=1 (Monte Carlo returns)
    entCoef: 0.01,        // Entropy bonus for exploration
    vfCoef: 0.5,
    maxGradNorm: 0.5,
    verbose: 1,
  }).init(dev, vecEnv)

  // Train for 100k timesteps
  const totalTimesteps = 100_000
  console.log(`Training for ${totalTimesteps.toLocaleString()} timesteps...\n`)

  const t0 = Date.now()

  await a2c.learn({
    totalTimesteps,
    logInterval: 100,  // Log more frequently due to short rollouts
  })

  const elapsed = ((Date.now() - t0) / 1000).toFixed(1)
  console.log(`\nTraining completed in ${elapsed}s`)
  console.log(`Total timesteps: ${a2c.numTimesteps.toLocaleString()}`)

  // Evaluate the trained policy
  console.log('\n--- Evaluation ---\n')
  console.log('Running 10 evaluation episodes...\n')

  const evalResults: { steps: number; reward: number }[] = []

  for (let ep = 0; ep < 10; ep++) {
    const evalEnv = RL.envs.CartPole()

    let totalReward = 0
    let steps = 0
    let obs = evalEnv.reset()

    while (steps < 500) {
      const action = a2c.predict(obs, true)
      const result = evalEnv.step(action as number)
      const reward = typeof result.reward === 'number' ? result.reward : result.reward[0]!
      totalReward += reward
      steps++

      if (result.done) break
      obs = result.observation
    }

    evalResults.push({ steps, reward: totalReward })
    console.log(`  Episode ${ep + 1}: ${steps} steps, reward = ${totalReward}`)
  }

  const meanReward = evalResults.reduce((sum, r) => sum + r.reward, 0) / evalResults.length
  const meanSteps = evalResults.reduce((sum, r) => sum + r.steps, 0) / evalResults.length

  console.log(`\nEvaluation Results:`)
  console.log(`  Mean reward: ${meanReward.toFixed(1)}`)
  console.log(`  Mean steps:  ${meanSteps.toFixed(1)}`)

  if (meanReward >= 475) {
    console.log('\nCartPole SOLVED! (mean reward >= 475)')
  } else if (meanReward >= 200) {
    console.log('\nGood progress! A2C may need more timesteps than PPO.')
  } else {
    console.log('\nNote: A2C typically needs more timesteps than PPO.')
  }

  console.log('\n=== Done ===')
}

main().catch(console.error)
