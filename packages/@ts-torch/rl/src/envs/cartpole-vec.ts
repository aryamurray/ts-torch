/**
 * CartPoleVec - SoA Batched CartPole Environment
 *
 * PufferLib-inspired: single VecEnv that steps all N envs in one tight loop
 * with flat TypedArrays. No per-env objects, no FunctionalEnv wrapper.
 *
 * Key design:
 * - Structure-of-Arrays for cache-friendly iteration
 * - Obs written directly into batch buffer at correct offset (zero-copy)
 * - Double-buffered observations for safe lastObs reference
 * - Static empty infos array reused when no episodes terminate
 *
 * @example
 * ```ts
 * const vecEnv = new CartPoleVec(8)
 * const obs = vecEnv.reset()
 * const { observations, rewards, dones } = vecEnv.step(actions)
 * ```
 */

import type { Space } from '../spaces/index.js'
import { discrete } from '../spaces/discrete.js'
import { box } from '../spaces/box.js'
import type { VecEnv, VecEnvStepResult, EnvInfo } from '../vec-env/base.js'

// ==================== Physics Constants ====================

const GRAVITY = 9.8
const MASS_CART = 1.0
const MASS_POLE = 0.1
const TOTAL_MASS = MASS_CART + MASS_POLE
const POLE_HALF_LENGTH = 0.5
const POLE_MASS_LENGTH = MASS_POLE * POLE_HALF_LENGTH
const FORCE_MAG = 10.0
const TAU = 0.02
const X_THRESHOLD = 2.4
const THETA_THRESHOLD = 12 * Math.PI / 180
const MAX_STEPS = 500
const OBS_SIZE = 4
const N_ACTIONS = 2

// Normalization constants (same as CartPole.observe())
const INV_X_THRESHOLD = 1 / X_THRESHOLD
const INV_THETA_THRESHOLD = 1 / THETA_THRESHOLD
const INV_VEL_SCALE = 1 / 2.0

// ==================== Implementation ====================

export class CartPoleVec implements VecEnv {
  private readonly nEnvs_: number
  private readonly observationSpace_: Space
  private readonly actionSpace_: Space

  // SoA state arrays
  private readonly x: Float64Array
  private readonly xDot: Float64Array
  private readonly theta: Float64Array
  private readonly thetaDot: Float64Array
  private readonly stepCounts: Uint32Array
  private readonly episodeRewards: Float64Array

  // Double-buffered observation output (zero-copy to consumer)
  private readonly obsBufferA: Float32Array
  private readonly obsBufferB: Float32Array
  private obsBufferCurrent: Float32Array
  private obsBufferPrev: Float32Array

  // Output buffers
  private readonly rewardBuffer: Float32Array
  private readonly doneBuffer: Uint8Array

  // Static empty infos (reused when no episodes end)
  private readonly emptyInfos: readonly EnvInfo[]

  constructor(nEnvs: number) {
    this.nEnvs_ = nEnvs

    this.observationSpace_ = box({
      low: [-Infinity, -Infinity, -Infinity, -Infinity],
      high: [Infinity, Infinity, Infinity, Infinity],
      shape: [OBS_SIZE],
    })
    this.actionSpace_ = discrete(N_ACTIONS)

    // SoA state (Float64 for physics precision)
    this.x = new Float64Array(nEnvs)
    this.xDot = new Float64Array(nEnvs)
    this.theta = new Float64Array(nEnvs)
    this.thetaDot = new Float64Array(nEnvs)
    this.stepCounts = new Uint32Array(nEnvs)
    this.episodeRewards = new Float64Array(nEnvs)

    // Double-buffered obs
    this.obsBufferA = new Float32Array(nEnvs * OBS_SIZE)
    this.obsBufferB = new Float32Array(nEnvs * OBS_SIZE)
    this.obsBufferCurrent = this.obsBufferA
    this.obsBufferPrev = this.obsBufferB

    // Output buffers
    this.rewardBuffer = new Float32Array(nEnvs)
    this.doneBuffer = new Uint8Array(nEnvs)

    // Static empty infos
    const infos: EnvInfo[] = []
    for (let i = 0; i < nEnvs; i++) infos.push({})
    this.emptyInfos = Object.freeze(infos)
  }

  get nEnvs(): number {
    return this.nEnvs_
  }

  get observationSpace(): Space {
    return this.observationSpace_
  }

  get actionSpace(): Space {
    return this.actionSpace_
  }

  get observationSize(): number {
    return OBS_SIZE
  }

  get actionDim(): number {
    return N_ACTIONS
  }

  get rewardDim(): number {
    return 1
  }

  /**
   * Reset all environments
   */
  reset(): Float32Array {
    const n = this.nEnvs_
    for (let i = 0; i < n; i++) {
      this.resetEnv(i)
    }
    // Write observations
    this.writeObs(this.obsBufferCurrent)
    return new Float32Array(this.obsBufferCurrent)
  }

  /**
   * Step all environments in a single tight loop
   */
  step(actions: Int32Array | Float32Array): VecEnvStepResult {
    const n = this.nEnvs_

    // Swap double-buffers
    const writeBuffer = this.obsBufferPrev
    this.obsBufferPrev = this.obsBufferCurrent
    this.obsBufferCurrent = writeBuffer

    let hasEpisodeEnd = false

    // Single tight loop: physics + obs + reward + done + auto-reset
    for (let i = 0; i < n; i++) {
      const action = Math.round(actions[i]!)
      const force = action === 1 ? FORCE_MAG : -FORCE_MAG

      // Read state
      const x = this.x[i]!
      const xDot = this.xDot[i]!
      const theta = this.theta[i]!
      const thetaDot = this.thetaDot[i]!

      // Physics
      const cosTheta = Math.cos(theta)
      const sinTheta = Math.sin(theta)
      const temp = (force + POLE_MASS_LENGTH * thetaDot * thetaDot * sinTheta) / TOTAL_MASS
      const thetaAcc =
        (GRAVITY * sinTheta - cosTheta * temp) /
        (POLE_HALF_LENGTH * (4 / 3 - (MASS_POLE * cosTheta * cosTheta) / TOTAL_MASS))
      const xAcc = temp - (POLE_MASS_LENGTH * thetaAcc * cosTheta) / TOTAL_MASS

      // Euler integration
      const newX = x + TAU * xDot
      const newXDot = xDot + TAU * xAcc
      const newTheta = theta + TAU * thetaDot
      const newThetaDot = thetaDot + TAU * thetaAcc

      // Write new state
      this.x[i] = newX
      this.xDot[i] = newXDot
      this.theta[i] = newTheta
      this.thetaDot[i] = newThetaDot
      this.stepCounts[i]!++

      // Check termination
      const terminated =
        newX < -X_THRESHOLD ||
        newX > X_THRESHOLD ||
        newTheta < -THETA_THRESHOLD ||
        newTheta > THETA_THRESHOLD
      const truncated = !terminated && this.stepCounts[i]! >= MAX_STEPS
      const done = terminated || truncated

      // Reward: +1 for surviving, 0 on termination (truncated steps still get reward)
      const reward = terminated ? 0 : 1
      this.rewardBuffer[i] = reward
      this.episodeRewards[i]! += reward
      this.doneBuffer[i] = done ? 1 : 0

      if (done) {
        hasEpisodeEnd = true
      }

      // Write normalized observations directly into batch buffer at offset
      const obsOffset = i * OBS_SIZE
      writeBuffer[obsOffset] = newX * INV_X_THRESHOLD
      writeBuffer[obsOffset + 1] = newXDot * INV_VEL_SCALE
      writeBuffer[obsOffset + 2] = newTheta * INV_THETA_THRESHOLD
      writeBuffer[obsOffset + 3] = newThetaDot * INV_VEL_SCALE

      // Auto-reset done envs (write reset obs over terminal obs)
      if (done) {
        this.resetEnv(i)
        writeBuffer[obsOffset] = this.x[i]! * INV_X_THRESHOLD
        writeBuffer[obsOffset + 1] = this.xDot[i]! * INV_VEL_SCALE
        writeBuffer[obsOffset + 2] = this.theta[i]! * INV_THETA_THRESHOLD
        writeBuffer[obsOffset + 3] = this.thetaDot[i]! * INV_VEL_SCALE
      }
    }

    // Build infos only when episodes ended (sparse)
    let infos: readonly EnvInfo[]
    if (hasEpisodeEnd) {
      const infoArray: EnvInfo[] = []
      for (let i = 0; i < n; i++) {
        if (this.doneBuffer[i] === 1) {
          infoArray.push({
            terminal: true,
            episodeLength: this.stepCounts[i],
            episodeReward: this.episodeRewards[i],
          })
          // Reset episode tracking (state was already reset above)
          this.episodeRewards[i] = 0
        } else {
          infoArray.push({})
        }
      }
      infos = infoArray
    } else {
      infos = this.emptyInfos
    }

    return {
      observations: writeBuffer,
      rewards: this.rewardBuffer,
      dones: this.doneBuffer,
      infos: infos as EnvInfo[],
    }
  }

  /**
   * Step all environments, writing observations into a caller-provided buffer.
   *
   * Same physics as step() but writes obs into obsTarget instead of
   * the internal double-buffer. Used for shared-memory mode with RolloutBuffer.
   */
  stepInto(actions: Int32Array | Float32Array, obsTarget: Float32Array): VecEnvStepResult {
    const n = this.nEnvs_
    let hasEpisodeEnd = false

    for (let i = 0; i < n; i++) {
      const action = Math.round(actions[i]!)
      const force = action === 1 ? FORCE_MAG : -FORCE_MAG

      const x = this.x[i]!
      const xDot = this.xDot[i]!
      const theta = this.theta[i]!
      const thetaDot = this.thetaDot[i]!

      const cosTheta = Math.cos(theta)
      const sinTheta = Math.sin(theta)
      const temp = (force + POLE_MASS_LENGTH * thetaDot * thetaDot * sinTheta) / TOTAL_MASS
      const thetaAcc =
        (GRAVITY * sinTheta - cosTheta * temp) /
        (POLE_HALF_LENGTH * (4 / 3 - (MASS_POLE * cosTheta * cosTheta) / TOTAL_MASS))
      const xAcc = temp - (POLE_MASS_LENGTH * thetaAcc * cosTheta) / TOTAL_MASS

      const newX = x + TAU * xDot
      const newXDot = xDot + TAU * xAcc
      const newTheta = theta + TAU * thetaDot
      const newThetaDot = thetaDot + TAU * thetaAcc

      this.x[i] = newX
      this.xDot[i] = newXDot
      this.theta[i] = newTheta
      this.thetaDot[i] = newThetaDot
      this.stepCounts[i]!++

      const terminated =
        newX < -X_THRESHOLD ||
        newX > X_THRESHOLD ||
        newTheta < -THETA_THRESHOLD ||
        newTheta > THETA_THRESHOLD
      const truncated = !terminated && this.stepCounts[i]! >= MAX_STEPS
      const done = terminated || truncated

      const reward = terminated ? 0 : 1
      this.rewardBuffer[i] = reward
      this.episodeRewards[i]! += reward
      this.doneBuffer[i] = done ? 1 : 0

      if (done) {
        hasEpisodeEnd = true
      }

      // Write normalized observations directly into caller-provided target
      const obsOffset = i * OBS_SIZE
      obsTarget[obsOffset] = newX * INV_X_THRESHOLD
      obsTarget[obsOffset + 1] = newXDot * INV_VEL_SCALE
      obsTarget[obsOffset + 2] = newTheta * INV_THETA_THRESHOLD
      obsTarget[obsOffset + 3] = newThetaDot * INV_VEL_SCALE

      if (done) {
        this.resetEnv(i)
        obsTarget[obsOffset] = this.x[i]! * INV_X_THRESHOLD
        obsTarget[obsOffset + 1] = this.xDot[i]! * INV_VEL_SCALE
        obsTarget[obsOffset + 2] = this.theta[i]! * INV_THETA_THRESHOLD
        obsTarget[obsOffset + 3] = this.thetaDot[i]! * INV_VEL_SCALE
      }
    }

    let infos: readonly EnvInfo[]
    if (hasEpisodeEnd) {
      const infoArray: EnvInfo[] = []
      for (let i = 0; i < n; i++) {
        if (this.doneBuffer[i] === 1) {
          infoArray.push({
            terminal: true,
            episodeLength: this.stepCounts[i],
            episodeReward: this.episodeRewards[i],
          })
          this.episodeRewards[i] = 0
        } else {
          infoArray.push({})
        }
      }
      infos = infoArray
    } else {
      infos = this.emptyInfos
    }

    return {
      observations: obsTarget,
      rewards: this.rewardBuffer,
      dones: this.doneBuffer,
      infos: infos as EnvInfo[],
    }
  }

  getObservations(): Float32Array {
    this.writeObs(this.obsBufferCurrent)
    return new Float32Array(this.obsBufferCurrent)
  }

  close(): void {
    // No-op
  }

  // ==================== Internal ====================

  private resetEnv(i: number): void {
    this.x[i] = (Math.random() - 0.5) * 0.1
    this.xDot[i] = (Math.random() - 0.5) * 0.1
    this.theta[i] = (Math.random() - 0.5) * 0.1
    this.thetaDot[i] = (Math.random() - 0.5) * 0.1
    this.stepCounts[i] = 0
  }

  private writeObs(buffer: Float32Array): void {
    for (let i = 0; i < this.nEnvs_; i++) {
      const offset = i * OBS_SIZE
      buffer[offset] = this.x[i]! * INV_X_THRESHOLD
      buffer[offset + 1] = this.xDot[i]! * INV_VEL_SCALE
      buffer[offset + 2] = this.theta[i]! * INV_THETA_THRESHOLD
      buffer[offset + 3] = this.thetaDot[i]! * INV_VEL_SCALE
    }
  }
}
