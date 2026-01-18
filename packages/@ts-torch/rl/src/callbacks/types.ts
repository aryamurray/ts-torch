/**
 * Declarative Callback Types
 *
 * Data structures and interfaces for the declarative callback system.
 * Users can pass simple inline functions instead of callback classes.
 *
 * @example
 * ```ts
 * await agent.learn({
 *   totalTimesteps: 100_000,
 *   callbacks: {
 *     onEpisodeEnd: (data) => {
 *       console.log(`Episode reward: ${data.episodeReward}`)
 *       if (data.episodeReward > 200) return false  // early stop
 *     },
 *     onStep: (data) => {
 *       // Access step data
 *       console.log(`Step ${data.timestep}, rewards: ${data.rewards}`)
 *       return true
 *     }
 *   }
 * })
 * ```
 */

// ==================== Lifecycle Data ====================

/**
 * Data passed when training starts
 */
export interface TrainingStartData {
  /** Total timesteps to train */
  totalTimesteps: number
  /** Number of parallel environments */
  nEnvs: number
  /** Algorithm name */
  algorithm: string
}

/**
 * Data passed when training ends
 */
export interface TrainingEndData {
  /** Total timesteps completed */
  totalTimesteps: number
  /** Total episodes completed */
  totalEpisodes: number
  /** Total training time in milliseconds */
  totalTime: number
  /** Final mean reward */
  finalReward: number
}

// ==================== Step Data ====================

/**
 * Data passed on each environment step
 */
export interface StepData {
  /** Current timestep count */
  timestep: number
  /** Observations from all environments [nEnvs * obsSize] */
  observations: Float32Array
  /** Actions taken [nEnvs * actionDim] or [nEnvs] for discrete */
  actions: Float32Array | Int32Array
  /** Rewards from all environments [nEnvs] */
  rewards: Float32Array
  /** Done flags for all environments */
  dones: boolean[]
  /** Additional info from environments */
  infos: Record<string, unknown>[]
}

// ==================== Episode Data ====================

/**
 * Data passed when an episode starts
 */
export interface EpisodeStartData {
  /** Index of the environment that started a new episode */
  envIndex: number
  /** Current timestep */
  timestep: number
}

/**
 * Data passed when an episode ends
 */
export interface EpisodeEndData {
  /** Index of the environment that finished */
  envIndex: number
  /** Total reward for the episode */
  episodeReward: number
  /** Number of steps in the episode */
  episodeLength: number
  /** Current timestep */
  timestep: number
  /** Additional episode info */
  info: Record<string, unknown>
}

// ==================== Rollout Data ====================

/**
 * Data passed when a rollout starts
 */
export interface RolloutStartData {
  /** Current timestep */
  timestep: number
  /** Rollout iteration number */
  iteration: number
}

/**
 * Data passed when a rollout ends
 */
export interface RolloutEndData {
  /** Current timestep */
  timestep: number
  /** Mean reward during this rollout */
  rolloutReward: number
  /** Number of steps in the rollout */
  rolloutLength: number
  /** Number of episodes completed during this rollout */
  episodesCompleted: number
}

// ==================== Evaluation Data ====================

/**
 * Data passed when evaluation starts
 */
export interface EvalStartData {
  /** Current timestep */
  timestep: number
  /** Number of episodes to evaluate */
  nEpisodes: number
}

/**
 * Data passed when evaluation ends
 */
export interface EvalEndData {
  /** Current timestep */
  timestep: number
  /** Mean reward across evaluation episodes */
  meanReward: number
  /** Standard deviation of rewards */
  stdReward: number
  /** Mean episode length */
  meanLength: number
  /** All episode rewards */
  episodeRewards: number[]
}

// ==================== Checkpoint Data ====================

/**
 * Data passed when a checkpoint is saved
 */
export interface CheckpointData {
  /** Current timestep */
  timestep: number
  /** Path where checkpoint was saved */
  path: string
  /** Current metrics */
  metrics: Record<string, number>
}

/**
 * Data passed when a new best model is saved
 */
export interface BestModelData extends CheckpointData {
  /** Previous best metric value */
  previousBest: number
  /** New best metric value */
  newBest: number
  /** Name of the metric that improved */
  metric: string
}

// ==================== Callbacks Interface ====================

/**
 * Declarative callbacks interface
 *
 * All callbacks are optional. Return `false` from callbacks that support
 * early stopping to halt training.
 *
 * @example
 * ```ts
 * const callbacks: Callbacks = {
 *   onTrainingStart: ({ totalTimesteps, algorithm }) => {
 *     console.log(`Starting ${algorithm} for ${totalTimesteps} steps`)
 *   },
 *
 *   onEpisodeEnd: ({ episodeReward, episodeLength }) => {
 *     console.log(`Episode done: reward=${episodeReward}, length=${episodeLength}`)
 *     // Return false to stop training early
 *     if (episodeReward > 500) return false
 *   },
 *
 *   onTrainingEnd: ({ totalTime, finalReward }) => {
 *     console.log(`Training complete in ${totalTime}ms, final reward: ${finalReward}`)
 *   }
 * }
 * ```
 */
export interface Callbacks {
  // ===== Lifecycle =====
  /** Called when training starts */
  onTrainingStart?: (data: TrainingStartData) => void
  /** Called when training ends */
  onTrainingEnd?: (data: TrainingEndData) => void

  // ===== Per-Step =====
  /** Called every environment step. Return false to stop training. */
  onStep?: (data: StepData) => boolean | void

  // ===== Episodes =====
  /** Called when any environment starts a new episode */
  onEpisodeStart?: (data: EpisodeStartData) => void
  /** Called when any environment finishes an episode. Return false to stop training. */
  onEpisodeEnd?: (data: EpisodeEndData) => boolean | void

  // ===== Rollouts =====
  /** Called when a rollout collection starts */
  onRolloutStart?: (data: RolloutStartData) => void
  /** Called when a rollout collection ends */
  onRolloutEnd?: (data: RolloutEndData) => void

  // ===== Evaluation =====
  /** Called when evaluation starts (requires evalFreq in learn config) */
  onEvalStart?: (data: EvalStartData) => void
  /** Called when evaluation ends. Return false to stop training. */
  onEvalEnd?: (data: EvalEndData) => boolean | void

  // ===== Checkpointing =====
  /** Called when a checkpoint is saved (requires checkpointFreq in learn config) */
  onCheckpoint?: (data: CheckpointData) => void
  /** Called when a new best model is saved */
  onBestModel?: (data: BestModelData) => void
}
