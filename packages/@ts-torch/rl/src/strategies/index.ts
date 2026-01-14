/**
 * Exploration Strategies
 *
 * This module provides various exploration strategies for RL agents.
 */

import { EpsilonGreedyStrategy, epsilonGreedy } from './epsilon-greedy.js'
import type { EpsilonGreedyConfig } from './epsilon-greedy.js'
import { EnvelopeQStrategy, envelopeQ } from './envelope.js'
import type { EnvelopeConfig } from './envelope.js'

// Re-export classes
export { EpsilonGreedyStrategy, epsilonGreedy }
export { EnvelopeQStrategy, envelopeQ }

// Re-export types
export type { EpsilonGreedyConfig, EnvelopeConfig }

/**
 * Union type for all strategy configurations
 */
export type ExplorationStrategyConfig =
  | ({ type: 'epsilon_greedy' } & EpsilonGreedyConfig)
  | ({ type: 'envelope_q_learning' } & EnvelopeConfig)

/**
 * Union type for all strategy instances
 */
export type ExplorationStrategy = EpsilonGreedyStrategy | EnvelopeQStrategy

/**
 * Create a strategy from configuration
 */
export function createStrategy(config: ExplorationStrategyConfig): ExplorationStrategy {
  switch (config.type) {
    case 'epsilon_greedy':
      return new EpsilonGreedyStrategy(config)
    case 'envelope_q_learning':
      return new EnvelopeQStrategy(config)
    default:
      throw new Error(`Unknown strategy type: ${(config as { type: string }).type}`)
  }
}
