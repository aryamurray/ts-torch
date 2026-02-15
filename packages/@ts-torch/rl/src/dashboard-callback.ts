/**
 * Dashboard Callback for RL Agents
 *
 * Glue code that wires a @ts-torch/dashboard instance to the RL Callbacks interface.
 * Dynamically imports @ts-torch/dashboard so the dependency is optional.
 *
 * Because the RL training loop is synchronous (no awaits between rollout steps),
 * setInterval-based rendering never fires. Instead we call dash.requestRender()
 * after pushing data so the dashboard renders inline, throttled automatically.
 */

import type { Callbacks } from './callbacks/index.js'

function formatMetric(value: number): string {
  const abs = Math.abs(value)
  if (abs >= 1000) return value.toFixed(0)
  if (abs >= 100) return value.toFixed(1)
  if (abs >= 1) return value.toFixed(2)
  if (abs >= 0.001) return value.toFixed(4)
  return value.toExponential(2)
}

export async function createRLDashboardCallback(title?: string): Promise<Callbacks> {
  const { Dashboard } = await import('@ts-torch/dashboard')
  const dash = new Dashboard({ title: title ?? 'ts-torch RL' })

  let totalTimesteps = 0
  let recentEpisodeRewards: number[] = []

  return {
    onTrainingStart(data) {
      totalTimesteps = data.totalTimesteps
      dash.start()
      dash.status.update('train', [
        { tag: 'Algorithm', value: data.algorithm },
        { tag: 'Envs', value: String(data.nEnvs) },
        { tag: 'Timesteps', value: `0/${totalTimesteps.toLocaleString()}` },
      ])
    },

    onEpisodeEnd(data) {
      recentEpisodeRewards.push(data.episodeReward)
      if (recentEpisodeRewards.length > 100) {
        recentEpisodeRewards = recentEpisodeRewards.slice(-100)
      }

      const meanReward = recentEpisodeRewards.reduce((a, b) => a + b, 0) / recentEpisodeRewards.length

      dash.numericMetrics.push('Episode Reward', 'train', data.episodeReward)
      dash.numericMetrics.push('Episode Length', 'train', data.episodeLength)
      dash.numericMetrics.push('Mean Reward (100)', 'train', meanReward)

      const progress = totalTimesteps > 0 ? data.timestep / totalTimesteps : 0
      dash.progress.update('train', progress, progress)

      dash.status.update('train', [
        { tag: 'Timesteps', value: `${data.timestep.toLocaleString()}/${totalTimesteps.toLocaleString()}` },
        { tag: 'Episodes', value: String(recentEpisodeRewards.length) },
        { tag: 'Mean Reward', value: formatMetric(meanReward) },
      ])

      // Synchronous render since the RL loop blocks the event loop
      dash.requestRender()

      if (dash.quitRequested) return false
      return undefined
    },

    onRolloutEnd(data) {
      if (data.rolloutReward !== 0) {
        dash.numericMetrics.push('Rollout Reward', 'train', data.rolloutReward)
      }
      dash.textMetrics.push('Rollout', 'train', `${data.rolloutLength} steps, ${data.episodesCompleted} eps`)
      dash.requestRender()
    },

    onEvalStart(data) {
      dash.status.update('valid', [
        { tag: 'Timestep', value: data.timestep.toLocaleString() },
        { tag: 'Episodes', value: String(data.nEpisodes) },
      ])
      dash.requestRender()
    },

    onEvalEnd(data) {
      dash.numericMetrics.push('Eval Reward', 'valid', data.meanReward)
      dash.textMetrics.push(
        'Eval Reward',
        'valid',
        `${formatMetric(data.meanReward)} ± ${formatMetric(data.stdReward)}`,
      )
      dash.textMetrics.push('Eval Length', 'valid', formatMetric(data.meanLength))

      dash.status.update('train', [
        { tag: 'Timesteps', value: `${data.timestep.toLocaleString()}/${totalTimesteps.toLocaleString()}` },
      ])
      dash.requestRender()

      if (dash.quitRequested) return false
      return undefined
    },

    onTrainingEnd(data) {
      dash.status.update('train', [
        { tag: 'Status', value: 'Complete' },
        { tag: 'Time', value: `${(data.totalTime / 1000).toFixed(1)}s` },
        { tag: 'Episodes', value: String(data.totalEpisodes) },
        { tag: 'Final Reward', value: formatMetric(data.finalReward) },
      ])
      dash.progress.update('train', 1, 1)
      dash.requestRender()
      setTimeout(() => dash.destroy(), 500)
    },
  }
}
