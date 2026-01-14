/**
 * CartPole Environment
 *
 * Classic control problem: balance a pole on a cart by applying forces.
 * The goal is to keep the pole upright and the cart within bounds.
 *
 * State: [x, x_dot, theta, theta_dot]
 * - x: cart position
 * - x_dot: cart velocity
 * - theta: pole angle (radians from vertical)
 * - theta_dot: pole angular velocity
 *
 * Actions: 0 = push left, 1 = push right
 *
 * Reward: +1 for each step the pole stays upright
 *
 * Terminal conditions:
 * - Pole angle > 12 degrees (0.2095 radians)
 * - Cart position > 2.4 units from center
 * - Episode length > 500 steps (truncation)
 *
 * @example
 * ```ts
 * import { CartPole } from '@ts-torch/rl'
 *
 * const env = CartPole()
 *
 * let obs = env.reset()
 * let done = false
 * let totalReward = 0
 *
 * while (!done) {
 *   const action = Math.random() < 0.5 ? 0 : 1
 *   const result = env.step(action)
 *   obs = result.observation
 *   done = result.done
 *   totalReward += result.reward
 * }
 * ```
 */

import { env, type FunctionalEnv } from '../environment.js'

// ==================== Constants ====================

/** Gravity constant (m/s^2) */
const GRAVITY = 9.8

/** Cart mass (kg) */
const MASS_CART = 1.0

/** Pole mass (kg) */
const MASS_POLE = 0.1

/** Total mass */
const TOTAL_MASS = MASS_CART + MASS_POLE

/** Half pole length (m) */
const POLE_HALF_LENGTH = 0.5

/** Pole mass times length */
const POLE_MASS_LENGTH = MASS_POLE * POLE_HALF_LENGTH

/** Force magnitude applied by actions */
const FORCE_MAG = 10.0

/** Time step (seconds) */
const TAU = 0.02

/** Maximum cart position */
const X_THRESHOLD = 2.4

/** Maximum pole angle (radians) - about 12 degrees */
const THETA_THRESHOLD = 12 * Math.PI / 180

/** Maximum episode length */
const MAX_STEPS = 500

// ==================== State Type ====================

/**
 * CartPole state
 */
export interface CartPoleState {
  /** Cart position */
  x: number
  /** Cart velocity */
  xDot: number
  /** Pole angle (radians from vertical) */
  theta: number
  /** Pole angular velocity */
  thetaDot: number
}

// ==================== Implementation ====================

/**
 * Create a CartPole environment
 *
 * @returns FunctionalEnv for CartPole
 */
export function CartPole(): FunctionalEnv<CartPoleState> {
  return env<CartPoleState>({
    createState: () => ({
      // Initialize with small random values
      x: (Math.random() - 0.5) * 0.1,
      xDot: (Math.random() - 0.5) * 0.1,
      theta: (Math.random() - 0.5) * 0.1,
      thetaDot: (Math.random() - 0.5) * 0.1,
    }),

    reset: (state) => {
      // Reset with small random values
      state.x = (Math.random() - 0.5) * 0.1
      state.xDot = (Math.random() - 0.5) * 0.1
      state.theta = (Math.random() - 0.5) * 0.1
      state.thetaDot = (Math.random() - 0.5) * 0.1
    },

    step: (state, action) => {
      // Get force direction
      const force = action === 1 ? FORCE_MAG : -FORCE_MAG

      // Physics simulation using Euler integration
      const cosTheta = Math.cos(state.theta)
      const sinTheta = Math.sin(state.theta)

      // Equations of motion (simplified pendulum on cart)
      const temp = (force + POLE_MASS_LENGTH * state.thetaDot ** 2 * sinTheta) / TOTAL_MASS

      const thetaAcc =
        (GRAVITY * sinTheta - cosTheta * temp) /
        (POLE_HALF_LENGTH * (4 / 3 - (MASS_POLE * cosTheta ** 2) / TOTAL_MASS))

      const xAcc = temp - (POLE_MASS_LENGTH * thetaAcc * cosTheta) / TOTAL_MASS

      // Euler integration (mutate state in place)
      state.x += TAU * state.xDot
      state.xDot += TAU * xAcc
      state.theta += TAU * state.thetaDot
      state.thetaDot += TAU * thetaAcc

      // Check termination conditions
      const done =
        state.x < -X_THRESHOLD ||
        state.x > X_THRESHOLD ||
        state.theta < -THETA_THRESHOLD ||
        state.theta > THETA_THRESHOLD

      // Reward is 1 for each step survived
      return {
        reward: done ? 0 : 1,
        done,
      }
    },

    observe: (state) => {
      // Normalize observations to roughly [-1, 1] range
      return new Float32Array([
        state.x / X_THRESHOLD,
        state.xDot / 2.0,  // Clip velocity to reasonable range
        state.theta / THETA_THRESHOLD,
        state.thetaDot / 2.0,
      ])
    },

    actionSpace: 2,
    maxSteps: MAX_STEPS,
  })
}

/**
 * CartPole with raw (unnormalized) observations
 *
 * @returns FunctionalEnv for CartPole with raw state
 */
export function CartPoleRaw(): FunctionalEnv<CartPoleState> {
  return env<CartPoleState>({
    createState: () => ({
      x: (Math.random() - 0.5) * 0.1,
      xDot: (Math.random() - 0.5) * 0.1,
      theta: (Math.random() - 0.5) * 0.1,
      thetaDot: (Math.random() - 0.5) * 0.1,
    }),

    reset: (state) => {
      state.x = (Math.random() - 0.5) * 0.1
      state.xDot = (Math.random() - 0.5) * 0.1
      state.theta = (Math.random() - 0.5) * 0.1
      state.thetaDot = (Math.random() - 0.5) * 0.1
    },

    step: (state, action) => {
      const force = action === 1 ? FORCE_MAG : -FORCE_MAG
      const cosTheta = Math.cos(state.theta)
      const sinTheta = Math.sin(state.theta)

      const temp = (force + POLE_MASS_LENGTH * state.thetaDot ** 2 * sinTheta) / TOTAL_MASS
      const thetaAcc =
        (GRAVITY * sinTheta - cosTheta * temp) /
        (POLE_HALF_LENGTH * (4 / 3 - (MASS_POLE * cosTheta ** 2) / TOTAL_MASS))
      const xAcc = temp - (POLE_MASS_LENGTH * thetaAcc * cosTheta) / TOTAL_MASS

      state.x += TAU * state.xDot
      state.xDot += TAU * xAcc
      state.theta += TAU * state.thetaDot
      state.thetaDot += TAU * thetaAcc

      const done =
        state.x < -X_THRESHOLD ||
        state.x > X_THRESHOLD ||
        state.theta < -THETA_THRESHOLD ||
        state.theta > THETA_THRESHOLD

      return { reward: done ? 0 : 1, done }
    },

    observe: (state) => new Float32Array([state.x, state.xDot, state.theta, state.thetaDot]),

    actionSpace: 2,
    maxSteps: MAX_STEPS,
  })
}
