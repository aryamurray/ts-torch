/**
 * Declarative RL Trainer
 *
 * Provides a high-level `fit()` function that handles the training loop,
 * experience replay, exploration decay, and callbacks. The user specifies
 * WHAT they want (episodes, strategy, memory config) and the trainer handles HOW.
 *
 * Features:
 * - Automatic exploration strategy management (epsilon decay)
 * - Experience replay with configurable warmup
 * - Multi-objective support with automatic weight concatenation
 * - Rich callback system for logging and early stopping
 *
 * @example
 * ```ts
 * import { fit } from '@ts-torch/rl'
 *
 * await fit(agent, env, {
 *   episodes: 1000,
 *   strategy: { type: 'epsilon_greedy', start: 1.0, end: 0.05, decay: 0.995 },
 *   memory: { capacity: 10000, batchSize: 32, warmup: 500 },
 *   onEpisodeEnd: ({ episode, totalReward }) => console.log(`Ep ${episode}: ${totalReward}`)
 * })
 * ```
 */

import { Logger } from '@ts-torch/core'
import type { FunctionalEnv } from './environment.js'
import { ReplayBuffer, type PERConfig } from './replay-buffer.js'
import type { Agent } from './agents/base.js'
import type { DQNAgent } from './agents/dqn.js'
import {
  EpsilonGreedyStrategy,
  EnvelopeQStrategy,
  type ExplorationStrategyConfig,
} from './strategies/index.js'
import { conditionObservation } from './utils/morl.js'
import type { Optimizer, LRScheduler } from '@ts-torch/optim'
import {
  StepLR,
  ExponentialLR,
  CosineAnnealingLR,
  ReduceLROnPlateau,
  LinearWarmup,
} from '@ts-torch/optim'

// ==================== Types ====================

/**
 * Context passed to step callbacks
 */
export interface StepContext {
  /** Current episode (1-indexed) */
  episode: number
  /** Current step within episode (0-indexed) */
  step: number
  /** Total steps across all episodes */
  totalSteps: number
  /** Reward from this step */
  reward: number | Float32Array
  /** Training loss (if warmup complete) */
  loss?: number
  /** Current exploration epsilon */
  epsilon: number
}

/**
 * Context passed to episode end callbacks
 */
export interface EpisodeContext {
  /** Current episode (1-indexed) */
  episode: number
  /** Total reward accumulated in episode */
  totalReward: number | Float32Array
  /** Number of steps in episode */
  steps: number
  /** Current exploration epsilon */
  epsilon: number
  /** Average loss for this episode (if any training occurred) */
  avgLoss?: number
}

/**
 * Training history returned by fit()
 */
export interface RLHistory {
  /** Total reward per episode */
  episodeRewards: (number | Float32Array)[]
  /** Steps per episode */
  episodeSteps: number[]
  /** Epsilon per episode */
  epsilons: number[]
  /** Average loss per episode */
  losses: number[]
  /** Total training steps */
  totalSteps: number
  /** Total episodes */
  totalEpisodes: number
}

/**
 * Memory/replay buffer configuration
 */
export interface MemoryConfig {
  /** Maximum number of transitions to store */
  capacity: number
  /** Batch size for training */
  batchSize: number
  /** Steps before training starts (fill buffer first) */
  warmup: number
  /** Prioritized Experience Replay configuration */
  per?: PERConfig
}

/**
 * Learning rate scheduler configuration
 *
 * Supports several scheduler types from @ts-torch/optim:
 * - step: Decay by gamma every stepSize epochs
 * - exponential: Decay by gamma every epoch
 * - cosine: Cosine annealing from base lr to etaMin
 * - plateau: Reduce lr when metric stops improving
 * - warmup: Linear warmup from 0 to base lr
 */
export type SchedulerConfig =
  | { type: 'step'; stepSize: number; gamma?: number }
  | { type: 'exponential'; gamma: number }
  | { type: 'cosine'; tMax: number; etaMin?: number }
  | { type: 'plateau'; patience: number; factor?: number; mode?: 'min' | 'max' }
  | { type: 'warmup'; warmupSteps: number }

/**
 * Training options for fit()
 */
export interface RLFitOptions {
  /** Number of episodes to train */
  episodes: number

  /** Maximum steps per episode (default: Infinity) */
  maxSteps?: number

  /** Exploration strategy configuration */
  strategy: ExplorationStrategyConfig

  /** Replay buffer configuration */
  memory: MemoryConfig

  /** Train every N steps (default: 1) */
  trainFreq?: number

  /** Learning rate scheduler configuration */
  scheduler?: SchedulerConfig

  /** When to step the scheduler (default: 'episode') */
  scheduleEvery?: 'episode' | 'step'

  /** Callback after each step */
  onStep?: (ctx: StepContext) => void | Promise<void>

  /** Callback after each episode */
  onEpisodeEnd?: (ctx: EpisodeContext) => void | Promise<void>

  /** Log progress every N episodes (0 = no logging) */
  logEvery?: number

  /** Early stopping: stop if avg reward exceeds this */
  targetReward?: number

  /** Early stopping: window size for averaging reward */
  targetRewardWindow?: number
}

// ==================== Implementation ====================

/**
 * Train an RL agent on an environment
 *
 * @param agent - The agent to train (e.g., DQN)
 * @param env - The environment to train on
 * @param options - Training configuration
 * @returns Training history
 *
 * @example
 * ```ts
 * // Single-objective training
 * const history = await fit(agent, cartpole, {
 *   episodes: 500,
 *   strategy: { type: 'epsilon_greedy', start: 1.0, end: 0.01, decay: 0.995 },
 *   memory: { capacity: 10000, batchSize: 64, warmup: 1000 },
 *   onEpisodeEnd: ({ episode, totalReward, epsilon }) => {
 *     console.log(`Episode ${episode}: reward=${totalReward}, eps=${epsilon.toFixed(3)}`)
 *   }
 * })
 *
 * // Multi-objective training
 * const history = await fit(agent, multiObjEnv, {
 *   episodes: 1000,
 *   strategy: {
 *     type: 'envelope_q_learning',
 *     start: 1.0, end: 0.05, decay: 0.995,
 *     rewardDim: 3
 *   },
 *   memory: { capacity: 50000, batchSize: 64, warmup: 1000 }
 * })
 * ```
 */
export async function fit<S>(
  agent: Agent,
  env: FunctionalEnv<S>,
  options: RLFitOptions,
): Promise<RLHistory> {
  const {
    episodes,
    maxSteps = Infinity,
    strategy: strategyConfig,
    memory,
    trainFreq = 1,
    scheduler: schedulerConfig,
    scheduleEvery = 'episode',
    onStep,
    onEpisodeEnd,
    logEvery = 0,
    targetReward,
    targetRewardWindow = 100,
  } = options

  // Create exploration strategy
  const strategy = createExplorationStrategy(strategyConfig)

  // Connect strategy to agent
  if ('setStrategy' in agent) {
    ;(agent as DQNAgent).setStrategy(strategy)
  }

  // Determine reward dimensionality
  const rewardDim =
    strategyConfig.type === 'envelope_q_learning' ? strategyConfig.rewardDim : 1

  // Create replay buffer (with optional PER)
  const buffer = new ReplayBuffer(memory.capacity, env.observationSize, rewardDim, memory.per)

  // Create learning rate scheduler (if configured)
  const scheduler = schedulerConfig
    ? createScheduler(schedulerConfig, getAgentOptimizer(agent))
    : null

  // Training history
  const history: RLHistory = {
    episodeRewards: [],
    episodeSteps: [],
    epsilons: [],
    losses: [],
    totalSteps: 0,
    totalEpisodes: 0,
  }

  // Tracking
  let totalSteps = 0
  const recentRewards: number[] = []

  // Set agent to training mode
  agent.train()

  // Main training loop
  for (let episode = 1; episode <= episodes; episode++) {
    let observation = env.reset()
    let episodeReward: number | Float32Array = rewardDim === 1 ? 0 : new Float32Array(rewardDim)
    let episodeSteps = 0
    let episodeLoss = 0
    let lossCount = 0

    // For MORL: sample new weights at start of episode
    let weights: Float32Array | null = null
    if (strategy instanceof EnvelopeQStrategy) {
      weights = strategy.sampleWeights()
    }

    // Episode loop
    let done = false
    while (!done && episodeSteps < maxSteps) {
      // Condition observation with weights for MORL
      const conditionedObs = weights
        ? conditionObservation(observation, weights)
        : observation

      // Select action
      const action = agent.act(conditionedObs, true)

      // Take step in environment
      const stepResult = env.step(action)
      totalSteps++
      episodeSteps++

      // Accumulate reward
      episodeReward = accumulateReward(episodeReward, stepResult.reward, rewardDim)

      // Condition next observation for storage
      const conditionedNextObs = weights
        ? conditionObservation(stepResult.observation, weights)
        : stepResult.observation

      // Store transition
      buffer.push({
        state: conditionedObs,
        action,
        reward: stepResult.reward,
        nextState: conditionedNextObs,
        done: stepResult.done,
      })

      // Training step (if warmup complete and at train frequency)
      let loss: number | undefined
      if (buffer.size >= memory.warmup && totalSteps % trainFreq === 0) {
        const batch = buffer.sample(memory.batchSize)
        const result = agent.trainStep(batch)
        loss = result.loss
        episodeLoss += loss
        lossCount++

        // Update priorities for PER (if enabled and TD-errors available)
        if (buffer.isPrioritized && batch.indices && result.tdErrors) {
          buffer.updatePriorities(batch.indices, result.tdErrors)
        }

        // Step scheduler per-step (if configured)
        if (scheduler && scheduleEvery === 'step') {
          if (scheduler instanceof ReduceLROnPlateau) {
            scheduler.step(loss)
          } else {
            scheduler.step()
          }
        }
      }

      // Step callback
      if (onStep) {
        await onStep({
          episode,
          step: episodeSteps - 1,
          totalSteps,
          reward: stepResult.reward,
          loss,
          epsilon: getEpsilon(strategy),
        })
      }

      // Decay exploration
      strategy.step()

      // Update state
      observation = stepResult.observation
      done = stepResult.done
    }

    // Episode complete
    history.episodeRewards.push(episodeReward)
    history.episodeSteps.push(episodeSteps)
    history.epsilons.push(getEpsilon(strategy))
    history.losses.push(lossCount > 0 ? episodeLoss / lossCount : 0)
    history.totalSteps = totalSteps
    history.totalEpisodes = episode

    // Track recent rewards for early stopping
    const scalarReward = typeof episodeReward === 'number'
      ? episodeReward
      : episodeReward.reduce((a, b) => a + b, 0)
    recentRewards.push(scalarReward)
    if (recentRewards.length > targetRewardWindow) {
      recentRewards.shift()
    }

    // Step scheduler per-episode (if configured)
    if (scheduler && scheduleEvery === 'episode') {
      if (scheduler instanceof ReduceLROnPlateau) {
        // Use average loss as metric for plateau scheduler
        const avgLoss = lossCount > 0 ? episodeLoss / lossCount : 0
        scheduler.step(avgLoss)
      } else {
        scheduler.step()
      }
    }

    // Episode end callback
    if (onEpisodeEnd) {
      await onEpisodeEnd({
        episode,
        totalReward: episodeReward,
        steps: episodeSteps,
        epsilon: getEpsilon(strategy),
        avgLoss: lossCount > 0 ? episodeLoss / lossCount : undefined,
      })
    }

    // Logging
    if (logEvery > 0 && episode % logEvery === 0) {
      const avgReward = recentRewards.reduce((a, b) => a + b, 0) / recentRewards.length
      Logger.info(
        `Episode ${episode}/${episodes} | ` +
        `Steps: ${episodeSteps} | ` +
        `Reward: ${scalarReward.toFixed(2)} | ` +
        `Avg(${Math.min(episode, targetRewardWindow)}): ${avgReward.toFixed(2)} | ` +
        `Epsilon: ${getEpsilon(strategy).toFixed(4)}`
      )
    }

    // Early stopping check
    if (targetReward !== undefined && recentRewards.length >= targetRewardWindow) {
      const avgReward = recentRewards.reduce((a, b) => a + b, 0) / recentRewards.length
      if (avgReward >= targetReward) {
        Logger.info(`Target reward ${targetReward} reached at episode ${episode}!`)
        break
      }
    }
  }

  return history
}

// ==================== Helper Functions ====================

/**
 * Create exploration strategy from config
 */
function createExplorationStrategy(
  config: ExplorationStrategyConfig,
): EpsilonGreedyStrategy | EnvelopeQStrategy {
  if (config.type === 'epsilon_greedy') {
    return new EpsilonGreedyStrategy({
      start: config.start,
      end: config.end,
      decay: config.decay,
    })
  } else if (config.type === 'envelope_q_learning') {
    return new EnvelopeQStrategy({
      start: config.start,
      end: config.end,
      decay: config.decay,
      rewardDim: config.rewardDim,
    })
  }

  throw new Error(`Unknown strategy type: ${(config as { type: string }).type}`)
}

/**
 * Get current epsilon from strategy
 */
function getEpsilon(strategy: EpsilonGreedyStrategy | EnvelopeQStrategy): number {
  return strategy.currentEpsilon
}

/**
 * Accumulate reward (handles both scalar and vector)
 */
function accumulateReward(
  total: number | Float32Array,
  reward: number | Float32Array,
  rewardDim: number,
): number | Float32Array {
  if (rewardDim === 1) {
    // Scalar rewards
    const r = typeof reward === 'number' ? reward : reward[0]!
    return (total as number) + r
  } else {
    // Vector rewards
    const totalVec = total as Float32Array
    const rewardVec = reward as Float32Array
    for (let i = 0; i < rewardDim; i++) {
      totalVec[i] = totalVec[i]! + rewardVec[i]!
    }
    return totalVec
  }
}

/**
 * Get optimizer from agent using the Agent interface
 */
function getAgentOptimizer(agent: Agent): Optimizer | null {
  return agent.getOptimizer()
}

/**
 * Create learning rate scheduler from config
 */
function createScheduler(
  config: SchedulerConfig,
  optimizer: Optimizer | null,
): LRScheduler | null {
  if (!optimizer) {
    Logger.warn('Scheduler configured but no optimizer found on agent')
    return null
  }

  switch (config.type) {
    case 'step':
      return new StepLR(optimizer, config.stepSize, config.gamma ?? 0.1)

    case 'exponential':
      return new ExponentialLR(optimizer, config.gamma)

    case 'cosine':
      return new CosineAnnealingLR(optimizer, config.tMax, config.etaMin ?? 0)

    case 'plateau':
      return new ReduceLROnPlateau(optimizer, config.mode ?? 'min', {
        patience: config.patience,
        factor: config.factor ?? 0.1,
      })

    case 'warmup':
      return new LinearWarmup(optimizer, config.warmupSteps)

    default:
      throw new Error(`Unknown scheduler type: ${(config as { type: string }).type}`)
  }
}
