/**
 * Action and Observation Spaces
 *
 * Defines the structure of action and observation spaces for RL environments.
 * Inspired by Gymnasium/Gym spaces.
 *
 * @example
 * ```ts
 * import { discrete, box } from './spaces'
 *
 * // Discrete action space (e.g., CartPole)
 * const actionSpace = discrete(2)
 *
 * // Continuous action space (e.g., Pendulum)
 * const actionSpace = box({ low: [-2], high: [2], shape: [1] })
 * ```
 */

export { discrete } from './discrete.js'
export type { DiscreteSpace } from './discrete.js'

export { box, boxUniform } from './box.js'
export type { BoxSpace, BoxConfig } from './box.js'

// ==================== Union Types ====================

import type { DiscreteSpace } from './discrete.js'
import type { BoxSpace } from './box.js'

/**
 * Any action or observation space
 */
export type Space = DiscreteSpace | BoxSpace

/**
 * Type guard for discrete space
 */
export function isDiscreteSpace(space: Space): space is DiscreteSpace {
  return space.type === 'discrete'
}

/**
 * Type guard for box space
 */
export function isBoxSpace(space: Space): space is BoxSpace {
  return space.type === 'box'
}

/**
 * Get the flat size of a space
 */
export function getSpaceSize(space: Space): number {
  if (space.type === 'discrete') {
    return 1
  }
  return space.shape.reduce((a, b) => a * b, 1)
}

/**
 * Get the number of actions for an action space
 * For discrete: returns n
 * For box: returns the flat size (number of continuous actions)
 */
export function getActionDim(space: Space): number {
  if (space.type === 'discrete') {
    return space.n
  }
  return space.shape.reduce((a, b) => a * b, 1)
}
