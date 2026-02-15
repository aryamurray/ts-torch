import type { Callback } from './callbacks.js'

function formatMetric(name: string, value: number): string {
  if (name === 'accuracy' || name.endsWith('Accuracy')) return `${value.toFixed(2)}%`
  const abs = Math.abs(value)
  if (abs >= 100) return value.toFixed(1)
  if (abs >= 1) return value.toFixed(4)
  if (abs >= 0.001) return value.toFixed(5)
  return value.toExponential(2)
}

export async function createDashboardCallback(): Promise<Callback> {
  const { DashboardProcess } = await import('@ts-torch/dashboard')
  const dash = new DashboardProcess()
  let batchCount = 0

  return {
    async onTrainStart() {
      await dash.start()
    },
    onEpochStart(ctx) {
      batchCount = 0
      dash.status.update('train', [{ tag: 'Epoch', value: `${ctx.epoch}/${ctx.totalEpochs ?? '?'}` }])
    },
    onBatchEnd(ctx) {
      batchCount++
      dash.numericMetrics.push('Loss', 'train', ctx.loss)
      const taskProgress = ctx.totalBatches ? batchCount / ctx.totalBatches : 0
      const epochProgress = ctx.totalEpochs ? (ctx.epoch - 1) / ctx.totalEpochs + taskProgress / ctx.totalEpochs : 0
      dash.progress.update('train', epochProgress, taskProgress)
    },
    onValidationStart() {
      dash.status.update('valid', [])
    },
    onEpochEnd(ctx) {
      for (const [name, value] of Object.entries(ctx.metrics)) {
        dash.textMetrics.push(name, 'train', formatMetric(name, value))
        if (name !== 'loss') {
          dash.numericMetrics.push(name, 'train', value)
        }
      }
      if (ctx.valMetrics) {
        for (const [name, value] of Object.entries(ctx.valMetrics)) {
          dash.textMetrics.push(name, 'valid', formatMetric(name, value))
          dash.numericMetrics.push(name, 'valid', value)
        }
      }
      dash.progress.update('train', ctx.totalEpochs ? ctx.epoch / ctx.totalEpochs : 0, 1)
      dash.status.update('train', [{ tag: 'Epoch', value: `${ctx.epoch}/${ctx.totalEpochs ?? '?'}` }])
      if (dash.quitRequested) return { stop: true }
      return undefined
    },
    onTrainEnd() {
      dash.status.update('train', [{ tag: 'Status', value: 'Complete' }])
      setTimeout(() => dash.destroy(), 500)
    },
  }
}
