/**
 * Environment Abstraction
 *
 * Provides a functional, declarative API for defining RL environments.
 * Users define environments via configuration objects, not by extending classes.
 *
 * Design Philosophy:
 * - CPU Simulation: State management runs on CPU with optimized TypedArrays
 * - Structured Mutation: step() mutates state in place to avoid GC pressure
 * - Multi-Objective Native: Rewards are Float32Array vectors by default
 *
 * @example
 * ```ts
 * import { env } from '@ts-torch/rl'
 *
 * type GridState = { x: number; y: number }
 *
 * const gridWorld = env<GridState>({
 *   createState: () => ({ x: 0, y: 0 }),
 *   step: (state, action) => {
 *     // Mutate in place for performance
 *     if (action === 0) state.y += 1      // up
 *     else if (action === 1) state.y -= 1 // down
 *     else if (action === 2) state.x -= 1 // left
 *     else if (action === 3) state.x += 1 // right
 *
 *     const done = state.x === 4 && state.y === 4
 *     return { reward: done ? 10 : -0.1, done }
 *   },
 *   observe: (state) => new Float32Array([state.x / 4, state.y / 4]),
 *   actionSpace: 4
 * })
 * ```
 */

// ==================== Types ====================

/**
 * Result of a single environment step
 */
export interface StepResult {
  /**
   * Reward for this step.
   * - number: Single-objective reward
   * - Float32Array: Multi-objective reward vector
   */
  reward: Float32Array | number

  /**
   * Whether the episode has ended
   */
  done: boolean

  /**
   * Optional truncation flag (time limit reached, not terminal state)
   */
  truncated?: boolean

  /**
   * Optional additional info for debugging/logging
   */
  info?: Record<string, unknown>
}

/**
 * Environment configuration - the user defines these functions
 *
 * @template S - State type (user-defined)
 */
export interface EnvConfig<S> {
  /**
   * Allocates the memory slab for the simulation.
   * Runs once at initialization and on reset.
   *
   * @returns Initial state object
   */
  createState: () => S

  /**
   * Executes one simulation step.
   *
   * CONTRACT: Mutate 'state' in place for performance. Do not re-allocate.
   *
   * @param state - Current state (mutate this!)
   * @param action - Action to take (discrete integer)
   * @returns Step result with reward and done flag
   */
  step: (state: S, action: number) => StepResult

  /**
   * Transforms internal state S into a Tensor-ready Float32Array.
   * This is the bridge between CPU Simulation and GPU Network.
   *
   * @param state - Current state
   * @returns Observation as Float32Array
   */
  observe: (state: S) => Float32Array

  /**
   * Optional: Reset existing state in place (avoids allocation).
   * If not provided, createState() is called on reset.
   *
   * @param state - State to reset (mutate this!)
   */
  reset?: (state: S) => void

  /**
   * Number of discrete actions available.
   * Required for agents to know the action space size.
   */
  actionSpace?: number

  /**
   * Optional: Maximum steps per episode (for truncation)
   */
  maxSteps?: number
}

// ==================== Implementation ====================

/**
 * Internal wrapper class for functional environments.
 * Hidden from user - they interact via the env() factory.
 *
 * @template S - State type
 */
export class FunctionalEnv<S> {
  private state: S
  private readonly config_: EnvConfig<S>
  private currentStep: number = 0
  private observationSize_: number | undefined

  constructor(config: EnvConfig<S>) {
    this.config_ = config
    this.state = config.createState()

    // Cache observation size from first observation
    const obs = config.observe(this.state)
    this.observationSize_ = obs.length
  }

  /**
   * Get the environment configuration (for cloning)
   * @internal
   */
  get config(): EnvConfig<S> {
    return this.config_
  }

  /**
   * Reset the environment to initial state
   *
   * @returns Initial observation
   */
  reset(): Float32Array {
    this.currentStep = 0

    if (this.config.reset) {
      // Use in-place reset if provided
      this.config.reset(this.state)
    } else {
      // Otherwise create new state
      this.state = this.config.createState()
    }

    return this.config.observe(this.state)
  }

  /**
   * Take a step in the environment
   *
   * @param action - Discrete action to take
   * @returns Step result with observation, reward, and done flag
   */
  step(action: number): {
    observation: Float32Array
    reward: Float32Array | number
    done: boolean
    truncated: boolean
    info?: Record<string, unknown>
  } {
    this.currentStep++

    const result = this.config.step(this.state, action)

    // Check for truncation (max steps reached)
    const truncated =
      this.config.maxSteps !== undefined && this.currentStep >= this.config.maxSteps && !result.done

    return {
      observation: this.config.observe(this.state),
      reward: result.reward,
      done: result.done || truncated,
      truncated: result.truncated ?? truncated,
      info: result.info,
    }
  }

  /**
   * Get current observation without stepping
   */
  observe(): Float32Array {
    return this.config.observe(this.state)
  }

  /**
   * Get the current internal state (for debugging/inspection)
   */
  getState(): S {
    return this.state
  }

  /**
   * Size of observation vector
   */
  get observationSize(): number {
    return this.observationSize_!
  }

  /**
   * Number of discrete actions (undefined if not specified)
   */
  get actionSpace(): number | undefined {
    return this.config.actionSpace
  }

  /**
   * Current step count in episode
   */
  get stepCount(): number {
    return this.currentStep
  }
}

// ==================== Factory ====================

/**
 * Create a functional environment from configuration.
 *
 * This is the main entry point for defining environments.
 * Configuration is separated from execution for clarity and testability.
 *
 * @template S - State type
 * @param config - Environment configuration
 * @returns FunctionalEnv wrapper
 *
 * @example
 * ```ts
 * const myEnv = env({
 *   createState: () => ({ position: 0, velocity: 0 }),
 *   step: (state, action) => {
 *     state.velocity += action === 1 ? 0.1 : -0.1
 *     state.position += state.velocity
 *     return { reward: -Math.abs(state.position), done: false }
 *   },
 *   observe: (state) => new Float32Array([state.position, state.velocity]),
 *   actionSpace: 2
 * })
 * ```
 */
export function env<S>(config: EnvConfig<S>): FunctionalEnv<S> {
  return new FunctionalEnv(config)
}
