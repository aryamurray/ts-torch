/**
 * Training Callbacks
 *
 * Callbacks for customizing training behavior.
 *
 * Two APIs are available:
 * 1. Class-based (BaseCallback) - for advanced use cases
 * 2. Declarative (Callbacks interface) - recommended for most use cases
 */

// Class-based callbacks (legacy/advanced)
export {
  BaseCallback,
  CallbackList,
  StopTrainingCallback,
  EpisodeTrackingCallback,
  callbackList,
  maybeCallback,
} from './base.js'

export type {
  CallbackLocals,
  BaseAlgorithmRef,
  MetricsLogger,
} from './base.js'

// Declarative callbacks (recommended)
export type {
  Callbacks,
  TrainingStartData,
  TrainingEndData,
  StepData,
  EpisodeStartData,
  EpisodeEndData,
  RolloutStartData,
  RolloutEndData,
  EvalStartData,
  EvalEndData,
  CheckpointData,
  BestModelData,
} from './types.js'
