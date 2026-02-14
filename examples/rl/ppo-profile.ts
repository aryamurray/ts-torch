/**
 * PPO Training Time Breakdown
 *
 * Instruments the PPO training loop to measure where wall time is actually spent.
 * Sub-phase timing inside the minibatch loop identifies whether forward, loss,
 * backward, or optimizer step dominates.
 *
 * Usage:
 *   bun run examples/rl/ppo-profile.ts
 *   bun run examples/rl/ppo-profile.ts --vec   # Use batched CartPoleVec
 */

import { run, device, float32, getLib, Tensor } from '@ts-torch/core'
import { RL, PPO, OnPolicyAlgorithm, RolloutBuffer, CartPoleVec } from '@ts-torch/rl'

// CPU device for tensor creation in instrumented train
const cpu = device.cpu()

// ==================== Config ====================

const N_ENVS = 8
const N_STEPS = 2048
const N_EPOCHS = 10
const BATCH_SIZE = 64
const TOTAL_TIMESTEPS = 100_000 // ~6 rollout iterations
const USE_CARTPOLE_VEC = process.argv.includes('--vec')

// ==================== Timing Accumulator ====================

const t = {
  rolloutCollect: 0,
  gaeCompute: 0,
  trainTotal: 0,
  bufferGet: 0,
  advNorm: 0,
  forward: 0,
  loss: 0,
  backward: 0,
  optimizerStep: 0,
  metrics: 0,
  scopeOverhead: 0,
  iterations: 0,
  minibatches: 0,
}

function resetTimings() {
  for (const k of Object.keys(t) as (keyof typeof t)[]) t[k] = 0
}

// ==================== Monkey-Patch: Outer Phases ====================

const origCollectRollouts = (OnPolicyAlgorithm.prototype as any).collectRollouts
;(OnPolicyAlgorithm.prototype as any).collectRollouts = function (this: any) {
  const t0 = performance.now()
  const result = origCollectRollouts.call(this)
  t.rolloutCollect += performance.now() - t0
  return result
}

const origComputeGAE = RolloutBuffer.prototype.computeReturnsAndAdvantage
RolloutBuffer.prototype.computeReturnsAndAdvantage = function (this: any, ...args: any[]) {
  const t0 = performance.now()
  const result = (origComputeGAE as any).apply(this, args)
  t.gaeCompute += performance.now() - t0
  return result
}

const origGet = RolloutBuffer.prototype.get
RolloutBuffer.prototype.get = function* (this: any, batchSize?: number) {
  let tResume = performance.now()
  const gen = origGet.call(this, batchSize)
  for (const batch of gen) {
    t.bufferGet += performance.now() - tResume
    yield batch
    tResume = performance.now()
  }
  t.bufferGet += performance.now() - tResume
}

// ==================== Monkey-Patch: Sub-Phase Instrumented _trainNative ====================

// Resolve native functions once
let _nativePF: Function | null = null
let _nativeBAC: Function | null = null
try {
  _nativePF = (getLib() as any).ts_policy_forward ?? null
} catch {
  /* */
}
try {
  _nativeBAC = (getLib() as any).ts_backward_and_clip ?? null
} catch {
  /* */
}

// Replace _trainNative with an instrumented version that times each sub-phase.
// This is a copy of the real _trainNative with performance.now() calls inserted.
;(PPO.prototype as any)._trainNative = function (
  this: any,
  obsSize: number,
  needsKl: boolean,
  clipLow: number,
  clipHigh: number,
) {
  const nativePolicyForward = _nativePF!
  const nativeBackwardAndClip = _nativeBAC!

  const piParams = this.policy.policyNetParameters()
  const vfParams = this.policy.valueNetParameters()
  const piHandles = piParams.map((p: any) => p.data._handle)
  const vfHandles = vfParams.map((p: any) => p.data._handle)
  const allHandles = [...piHandles, ...vfHandles]
  const activationType = this.policy.getActivationType()
  const nActions = this.policy.getNumActions()

  for (let epoch = 0; epoch < this.nEpochs; epoch++) {
    for (const batch of this.rolloutBuffer.get(this.batchSize)) {
      const { observations, actions, oldLogProbs, advantages, returns } = batch

      // ---- Advantage normalization ----
      let tPhase = performance.now()
      if (this.normalizeAdvantage && advantages.length > 1) {
        let mean = 0
        for (let i = 0; i < advantages.length; i++) mean += advantages[i]!
        mean /= advantages.length
        let variance = 0
        for (let i = 0; i < advantages.length; i++) {
          const d = advantages[i]! - mean
          variance += d * d
        }
        const invStd = 1 / Math.sqrt(variance / advantages.length + 1e-8)
        for (let i = 0; i < advantages.length; i++) {
          advantages[i] = (advantages[i]! - mean) * invStd
        }
      }
      t.advNorm += performance.now() - tPhase

      let approxKlValue = 0

      // ---- Scope overhead: measure run() wrapper cost ----
      const tScopeStart = performance.now()

      run(() => {
        const tAfterScope = performance.now()
        t.scopeOverhead += tAfterScope - tScopeStart

        // ---- Forward ----
        let tp = performance.now()
        const fwdResult = nativePolicyForward(
          observations,
          actions,
          batch.batchSize,
          obsSize,
          nActions,
          piHandles,
          vfHandles,
          activationType,
        )
        const TensorCtor = Tensor as any
        const actionLogProbs = new TensorCtor(fwdResult.actionLogProbs, [batch.batchSize] as const, float32)
        const entropy = new TensorCtor(fwdResult.entropy, [] as const, float32)
        const values = new TensorCtor(fwdResult.values, [batch.batchSize] as const, float32)
        t.forward += performance.now() - tp

        // ---- Loss computation ----
        tp = performance.now()
        const oldLogProbsTensor = cpu.tensor(oldLogProbs, [batch.batchSize] as const)
        const advantagesTensor = cpu.tensor(advantages, [batch.batchSize] as const)
        const returnsTensor = cpu.tensor(returns, [batch.batchSize, 1] as const)

        const logDiff = (actionLogProbs as any).sub(oldLogProbsTensor)
        const ratio = (logDiff as any).exp()
        const surr1 = (ratio as any).mul(advantagesTensor)
        const clippedRatio = (ratio as any).clamp(clipLow, clipHigh)
        const surr2 = (clippedRatio as any).mul(advantagesTensor)
        const minSurr = (surr1 as any).minimum(surr2)
        const policyLoss = (minSurr as any).mean().neg()
        const valueLoss = (values as any).unsqueeze(1).mseLoss(returnsTensor)
        const entropyLoss = (entropy as any).neg()
        const scaledValueLoss = (valueLoss as any).mulScalar(this.vfCoef)
        const scaledEntropyLoss = (entropyLoss as any).mulScalar(this.entCoef)
        const totalLoss = (policyLoss as any).add(scaledValueLoss).add(scaledEntropyLoss)
        t.loss += performance.now() - tp

        // ---- Backward + grad clip ----
        tp = performance.now()
        nativeBackwardAndClip((totalLoss as any)._handle, allHandles, this.maxGradNorm)
        for (const p of piParams) {
          ;(p as any).data._gradCache = undefined
        }
        for (const p of vfParams) {
          ;(p as any).data._gradCache = undefined
        }
        t.backward += performance.now() - tp

        // ---- Optimizer step ----
        tp = performance.now()
        this.optimizer.step()
        t.optimizerStep += performance.now() - tp

        // ---- Metric extraction ----
        tp = performance.now()
        if (needsKl) {
          const klDiff = (oldLogProbsTensor as any).sub(actionLogProbs)
          const klSquared = (klDiff as any).mul(klDiff)
          approxKlValue = ((klSquared as any).mean().item?.() ?? 0) * 0.5
        }
        t.metrics += performance.now() - tp

        // Scope exit overhead is captured on next iteration's tScopeStart
      })

      t.minibatches++

      if (needsKl && approxKlValue > 1.5 * this.targetKl!) return
    }
  }
}

// Wrap _train to time total training
const origTrainDispatch = (PPO.prototype as any)._train
;(PPO.prototype as any)._train = function (this: any) {
  const t0 = performance.now()
  origTrainDispatch.call(this)
  t.trainTotal += performance.now() - t0
  t.iterations++
}

// ==================== Main ====================

async function main() {
  const dev = device.cpu()

  let vecEnv
  if (USE_CARTPOLE_VEC) {
    console.log('Using CartPoleVec (SoA batched)')
    vecEnv = new CartPoleVec(N_ENVS)
  } else {
    console.log('Using DummyVecEnv (standard)')
    vecEnv = RL.vecEnv({
      env: RL.envs.CartPole(),
      nEnvs: N_ENVS,
      type: 'dummy',
    })
  }

  const ppo = RL.ppo({
    policy: {
      netArch: { pi: [64, 64], vf: [64, 64] },
      activation: 'tanh',
    },
    learningRate: 3e-4,
    nSteps: N_STEPS,
    batchSize: BATCH_SIZE,
    nEpochs: N_EPOCHS,
    gamma: 0.99,
    gaeLambda: 0.95,
    clipRange: 0.2,
    entCoef: 0.01,
    vfCoef: 0.5,
    maxGradNorm: 0.5,
    verbose: 0,
  }).init(dev, vecEnv)

  // Warmup
  console.log('Warmup (1 iteration)...')
  await ppo.learn({ totalTimesteps: N_STEPS * N_ENVS })
  resetTimings()

  // Profile run
  console.log(`Profiling ${TOTAL_TIMESTEPS.toLocaleString()} timesteps...`)
  console.log()

  const t0 = performance.now()
  await ppo.learn({ totalTimesteps: TOTAL_TIMESTEPS, resetNumTimesteps: false })
  const wallMs = performance.now() - t0

  // ==================== Report ====================

  const stepsPerSec = TOTAL_TIMESTEPS / (wallMs / 1000)
  const nIter = t.iterations
  const nBatch = t.minibatches
  const netRollout = t.rolloutCollect - t.gaeCompute
  const netTrain = t.trainTotal - t.bufferGet
  const overhead = wallMs - t.rolloutCollect - t.trainTotal
  // Scope exit time = trainTotal - (advNorm + forward + loss + backward + optimizerStep + metrics + bufferGet + scopeOverhead)
  const scopeExit =
    netTrain - t.advNorm - t.forward - t.loss - t.backward - t.optimizerStep - t.metrics - t.scopeOverhead

  console.log('='.repeat(64))
  console.log('PPO Training Time Breakdown')
  console.log('='.repeat(64))
  console.log()
  console.log(`Wall time:     ${wallMs.toFixed(0)}ms (${stepsPerSec.toFixed(0)} steps/s)`)
  console.log(`Iterations:    ${nIter}`)
  console.log(`Minibatches:   ${nBatch}  (${(nBatch / nIter).toFixed(0)}/iter)`)
  console.log(`Per-minibatch: ${((netTrain / nBatch) * 1000).toFixed(0)}μs`)
  console.log()

  console.log('Phase breakdown (% of wall time):')
  console.log()
  p('Rollout collection', t.rolloutCollect, wallMs)
  p('  Env step + policy fwd', netRollout, wallMs)
  p('  GAE computation', t.gaeCompute, wallMs)
  console.log()
  p('Training update', t.trainTotal, wallMs)
  p('  Buffer get() iter', t.bufferGet, wallMs)
  p('  Advantage normalization', t.advNorm, wallMs)
  p('  Forward (native fused)', t.forward, wallMs)
  p('  Loss computation (TS)', t.loss, wallMs)
  p('  Backward + grad clip', t.backward, wallMs)
  p('  Optimizer step', t.optimizerStep, wallMs)
  p('  Metric extraction', t.metrics, wallMs)
  p('  run() scope enter', t.scopeOverhead, wallMs)
  p('  run() scope exit', scopeExit, wallMs)
  console.log()
  p('Other overhead', overhead, wallMs)
  console.log()

  // Per-minibatch averages (microseconds)
  console.log('Per-minibatch averages (μs):')
  const us = (ms: number) => ((ms / nBatch) * 1000).toFixed(0)
  console.log(`  Forward:     ${us(t.forward)}`)
  console.log(`  Loss:        ${us(t.loss)}`)
  console.log(`  Backward:    ${us(t.backward)}`)
  console.log(`  Optimizer:   ${us(t.optimizerStep)}`)
  console.log(`  Metrics:     ${us(t.metrics)}`)
  console.log(`  Scope enter: ${us(t.scopeOverhead)}`)
  console.log(`  Scope exit:  ${us(scopeExit)}`)
  console.log(`  AdvNorm:     ${us(t.advNorm)}`)
  console.log(`  Buffer get:  ${us(t.bufferGet)}`)
  console.log()

  const rss = process.memoryUsage().rss / (1024 * 1024)
  console.log(`RSS: ${rss.toFixed(1)} MB`)
  console.log('='.repeat(64))
}

function p(label: string, ms: number, total: number) {
  const pctStr = ((ms / total) * 100).toFixed(1).padStart(5) + '%'
  const msStr = ms.toFixed(0).padStart(6) + 'ms'
  console.log(`  ${label.padEnd(28)} ${msStr}  ${pctStr}`)
}

main().catch(console.error)
