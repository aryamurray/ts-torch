/**
 * PPO CartPole Rigorous Benchmark
 *
 * This is a proper validation test for the PPO implementation using:
 * - Unnormalized observations (CartPoleRaw) to match standard benchmarks
 * - 100 evaluation episodes for statistical significance
 * - Random baseline comparison
 * - Training curve tracking
 * - Comparison against Stable Baselines3 reference performance
 *
 * Expected performance (from SB3/CleanRL references):
 * - Random policy: ~20-25 mean reward
 * - Solved threshold: 475+ mean reward over 100 episodes
 * - SB3 PPO typically solves in 50k-100k timesteps
 *
 * Run with: bun run ppo-cartpole-benchmark.ts
 */

// Set thread env vars BEFORE native library loads — LibTorch, MKL, and OpenMP
// all read these at initialization. Single-threaded RL with small tensors doesn't
// benefit from intra-op parallelism; the thread spawn/join overhead hurts throughput.
process.env.OMP_NUM_THREADS = '1'
process.env.MKL_NUM_THREADS = '1'
process.env.OPENBLAS_NUM_THREADS = '1'
process.env.VECLIB_MAXIMUM_THREADS = '1' // macOS Accelerate framework

import { device, setNumThreads } from '@ts-torch/core'
import { RL } from '@ts-torch/rl'

// Also set via LibTorch API (belt and suspenders)
setNumThreads(1)

// ==================== Configuration ====================

const CONFIG = {
  // Training
  totalTimesteps: 200_000,  // Increased for harder unnormalized env
  nEnvs: 8,                 // More parallel envs for better sample efficiency

  // Evaluation
  evalEpisodes: 100,
  evalIntervalTimesteps: 20_000,  // Evaluate every N timesteps

  // Success criteria (OpenAI Gym standard)
  solvedThreshold: 475,  // Mean reward over 100 episodes
  maxEpisodeSteps: 500,

  // Reference baselines
  randomBaseline: 22,     // Expected random policy performance
  sb3Reference: 500,      // SB3 PPO achieves ~500 when solved
}

// ==================== Statistics Utilities ====================

interface EvalStats {
  mean: number
  std: number
  min: number
  max: number
  median: number
  episodes: number
}

function computeStats(values: number[]): EvalStats {
  const n = values.length
  if (n === 0) return { mean: 0, std: 0, min: 0, max: 0, median: 0, episodes: 0 }
  
  const sorted = [...values].sort((a, b) => a - b)
  const mean = values.reduce((a, b) => a + b, 0) / n
  const variance = values.reduce((sum, v) => sum + (v - mean) ** 2, 0) / n
  const std = Math.sqrt(variance)
  
  return {
    mean,
    std,
    min: sorted[0]!,
    max: sorted[n - 1]!,
    median: n % 2 === 0 
      ? (sorted[n / 2 - 1]! + sorted[n / 2]!) / 2 
      : sorted[Math.floor(n / 2)]!,
    episodes: n,
  }
}

function formatStats(stats: EvalStats): string {
  return `${stats.mean.toFixed(1)} ± ${stats.std.toFixed(1)} (min=${stats.min}, max=${stats.max}, median=${stats.median.toFixed(1)})`
}

// ==================== Evaluation ====================

function evaluatePolicy(
  ppo: any,
  numEpisodes: number,
  deterministic: boolean = true
): EvalStats {
  const rewards: number[] = []
  const lengths: number[] = []
  
  for (let ep = 0; ep < numEpisodes; ep++) {
    const evalEnv = RL.envs.CartPoleRaw()  // Unnormalized!
    
    let totalReward = 0
    let steps = 0
    let obs = evalEnv.reset()
    
    while (steps < CONFIG.maxEpisodeSteps) {
      const action = ppo.predict(obs, deterministic)
      const result = evalEnv.step(action as number)
      const reward = typeof result.reward === 'number' ? result.reward : result.reward[0]!
      totalReward += reward
      steps++
      
      if (result.done) break
      obs = result.observation
    }
    
    rewards.push(totalReward)
    lengths.push(steps)
  }
  
  return computeStats(rewards)
}

function evaluateRandomPolicy(numEpisodes: number): EvalStats {
  const rewards: number[] = []
  
  for (let ep = 0; ep < numEpisodes; ep++) {
    const evalEnv = RL.envs.CartPoleRaw()
    
    let totalReward = 0
    let steps = 0
    evalEnv.reset()
    
    while (steps < CONFIG.maxEpisodeSteps) {
      const action = Math.random() < 0.5 ? 0 : 1  // Random policy
      const result = evalEnv.step(action)
      const reward = typeof result.reward === 'number' ? result.reward : result.reward[0]!
      totalReward += reward
      steps++
      
      if (result.done) break
    }
    
    rewards.push(totalReward)
  }
  
  return computeStats(rewards)
}

// ==================== Main ====================

async function main() {
  console.log('=' .repeat(60))
  console.log('PPO CartPole Rigorous Benchmark')
  console.log('=' .repeat(60))
  console.log()
  
  console.log('Configuration:')
  console.log(`  Environment: CartPoleRaw (unnormalized observations)`)
  console.log(`  Total timesteps: ${CONFIG.totalTimesteps.toLocaleString()}`)
  console.log(`  Parallel envs: ${CONFIG.nEnvs}`)
  console.log(`  Eval episodes: ${CONFIG.evalEpisodes}`)
  console.log(`  Solved threshold: ${CONFIG.solvedThreshold}`)
  console.log()
  
  // ==================== Random Baseline ====================
  
  console.log('-'.repeat(60))
  console.log('Random Baseline (sanity check)')
  console.log('-'.repeat(60))
  
  const randomStats = evaluateRandomPolicy(CONFIG.evalEpisodes)
  console.log(`Random policy: ${formatStats(randomStats)}`)
  console.log(`Expected: ~${CONFIG.randomBaseline}`)
  
  const randomOk = randomStats.mean >= 15 && randomStats.mean <= 35
  console.log(`Status: ${randomOk ? '✓ PASS' : '✗ FAIL'} (environment behaves as expected)`)
  console.log()
  
  // ==================== PPO Training ====================
  
  console.log('-'.repeat(60))
  console.log('PPO Training')
  console.log('-'.repeat(60))
  
  const dev = device.cpu()
  
  // Use CartPoleRaw for unnormalized observations
  const vecEnv = RL.vecEnv({
    env: RL.envs.CartPoleRaw(),
    nEnvs: CONFIG.nEnvs,
    type: 'dummy'
  })
  
  // PPO with tuned hyperparameters for unnormalized CartPole
  // Note: Unnormalized observations have different scales which makes learning harder
  const ppo = RL.ppo({
    policy: {
      netArch: { pi: [64, 64], vf: [64, 64] },
      activation: 'tanh',
    },
    learningRate: 3e-4,
    nSteps: 2048,
    batchSize: 64,
    nEpochs: 10,
    gamma: 0.99,
    gaeLambda: 0.95,
    clipRange: 0.2,
    entCoef: 0.01,  // Small entropy bonus for exploration
    vfCoef: 0.5,
    maxGradNorm: 0.5,
    verbose: 0,  // Quiet during training
  }).init(dev, vecEnv)
  
  // Training with periodic evaluation
  const trainingCurve: { timesteps: number; meanReward: number; std: number }[] = []
  const t0 = Date.now()
  
  // Initial point (random-like performance before training)
  trainingCurve.push({ timesteps: 0, meanReward: randomStats.mean, std: randomStats.std })
  console.log(`  [0] Initial (untrained): ~${randomStats.mean.toFixed(1)} (assumed random-like)`)
  
  // Train in chunks with evaluation
  const chunkSize = CONFIG.evalIntervalTimesteps
  let currentTimesteps = 0
  
  while (currentTimesteps < CONFIG.totalTimesteps) {
    const targetTimesteps = Math.min(currentTimesteps + chunkSize, CONFIG.totalTimesteps)
    
    await ppo.learn({
      totalTimesteps: targetTimesteps,
      resetNumTimesteps: false,
    })
    
    currentTimesteps = ppo.numTimesteps
    
    // Quick evaluation (20 episodes for speed)
    const evalStats = evaluatePolicy(ppo, 20)
    trainingCurve.push({ 
      timesteps: currentTimesteps, 
      meanReward: evalStats.mean, 
      std: evalStats.std 
    })
    
    const elapsed = ((Date.now() - t0) / 1000).toFixed(1)
    console.log(`  [${currentTimesteps.toLocaleString()}] ${evalStats.mean.toFixed(1)} ± ${evalStats.std.toFixed(1)} (${elapsed}s)`)
  }
  
  const totalTime = (Date.now() - t0) / 1000
  const stepsPerSec = CONFIG.totalTimesteps / totalTime
  const rss = process.memoryUsage().rss / (1024 * 1024)
  console.log()
  console.log(`Training completed in ${totalTime.toFixed(1)}s`)
  console.log(`Throughput: ${(stepsPerSec / 1000).toFixed(1)}k steps/s`)
  console.log(`RSS: ${rss.toFixed(1)} MB`)
  console.log()
  
  // ==================== Final Evaluation ====================
  
  console.log('-'.repeat(60))
  console.log(`Final Evaluation (${CONFIG.evalEpisodes} episodes)`)
  console.log('-'.repeat(60))
  
  const finalStats = evaluatePolicy(ppo, CONFIG.evalEpisodes)
  console.log(`PPO policy: ${formatStats(finalStats)}`)
  console.log(`SB3 reference: ~${CONFIG.sb3Reference}`)
  console.log()
  
  // ==================== Learning Curve ====================
  
  console.log('-'.repeat(60))
  console.log('Learning Curve')
  console.log('-'.repeat(60))
  
  // Simple ASCII chart
  const maxReward = Math.max(...trainingCurve.map(p => p.meanReward))
  const chartHeight = 10
  const chartWidth = trainingCurve.length
  
  for (let row = chartHeight; row >= 0; row--) {
    const threshold = (row / chartHeight) * maxReward
    let line = row === chartHeight ? `${maxReward.toFixed(0).padStart(4)} |` :
               row === 0 ? '   0 |' :
               '     |'
    
    for (const point of trainingCurve) {
      if (point.meanReward >= threshold) {
        line += '█'
      } else {
        line += ' '
      }
    }
    console.log(line)
  }
  console.log('     +' + '-'.repeat(chartWidth))
  console.log('      0' + ' '.repeat(Math.max(0, chartWidth - 10)) + `${CONFIG.totalTimesteps / 1000}k`)
  console.log('                    timesteps')
  console.log()
  
  // ==================== Results Summary ====================
  
  console.log('='.repeat(60))
  console.log('Results Summary')
  console.log('='.repeat(60))
  console.log()
  
  // Check criteria
  const solved = finalStats.mean >= CONFIG.solvedThreshold
  const betterThanRandom = finalStats.mean > randomStats.mean * 2
  const reasonable = finalStats.mean >= 200  // At least shows learning
  
  console.log('Criteria:')
  console.log(`  [${solved ? '✓' : '✗'}] Solved (mean >= ${CONFIG.solvedThreshold}): ${finalStats.mean.toFixed(1)}`)
  console.log(`  [${betterThanRandom ? '✓' : '✗'}] Better than 2x random: ${finalStats.mean.toFixed(1)} vs ${(randomStats.mean * 2).toFixed(1)}`)
  console.log(`  [${reasonable ? '✓' : '✗'}] Shows learning (mean >= 200): ${finalStats.mean.toFixed(1)}`)
  console.log()
  
  // Comparison to baselines
  const vsRandom = ((finalStats.mean / randomStats.mean) * 100 - 100).toFixed(0)
  const vsSB3 = ((finalStats.mean / CONFIG.sb3Reference) * 100).toFixed(0)
  
  console.log('Comparison:')
  console.log(`  vs Random: +${vsRandom}% (${finalStats.mean.toFixed(1)} vs ${randomStats.mean.toFixed(1)})`)
  console.log(`  vs SB3 reference: ${vsSB3}% (${finalStats.mean.toFixed(1)} vs ${CONFIG.sb3Reference})`)
  console.log()
  
  // Final verdict
  console.log('='.repeat(60))
  if (solved) {
    console.log('VERDICT: ✓ PASSED - PPO implementation solves CartPole')
  } else if (reasonable) {
    console.log('VERDICT: ~ PARTIAL - Shows learning but does not fully solve')
    console.log('         Consider: more timesteps, hyperparameter tuning')
  } else {
    console.log('VERDICT: ✗ FAILED - PPO implementation needs debugging')
  }
  console.log('='.repeat(60))
  
  return { solved, finalStats, randomStats, trainingCurve }
}

main().catch(console.error)
