import { describe, it, expect } from 'vitest'
import {
  RecentHistory,
  FullHistory,
  NumericMetricsState,
  ProgressState,
  TextMetricsState,
  StatusState,
} from '../data.js'

describe('RecentHistory', () => {
  it('stores and retrieves points', () => {
    const h = new RecentHistory()
    h.push('train', 0.5)
    h.push('train', 0.4)
    h.push('train', 0.3)

    const datasets = h.getDatasets()
    expect(datasets).toHaveLength(1)
    expect(datasets[0].split).toBe('train')
    expect(datasets[0].points).toHaveLength(3)
    expect(datasets[0].points[0][1]).toBe(0.5)
    expect(datasets[0].points[2][1]).toBe(0.3)
  })

  it('tracks multiple splits', () => {
    const h = new RecentHistory()
    h.push('train', 1.0)
    h.push('valid', 1.1)

    const datasets = h.getDatasets()
    expect(datasets).toHaveLength(2)
    const splits = datasets.map((d) => d.split)
    expect(splits).toContain('train')
    expect(splits).toContain('valid')
  })

  it('computes bounds correctly', () => {
    const h = new RecentHistory()
    h.push('train', 10)
    h.push('train', 5)
    h.push('valid', 20)

    const bounds = h.getBounds()
    expect(bounds.yMin).toBe(5)
    expect(bounds.yMax).toBe(20)
    expect(bounds.xMin).toBe(0)
    expect(bounds.xMax).toBe(2)
  })

  it('trims old points beyond maxSamples', () => {
    const h = new RecentHistory(10)
    for (let i = 0; i < 50; i++) {
      h.push('train', i)
    }

    const datasets = h.getDatasets()
    expect(datasets[0].points.length).toBeLessThanOrEqual(10)
  })
})

describe('FullHistory', () => {
  it('stores and retrieves points', () => {
    const h = new FullHistory()
    h.push('train', 1.0)
    h.push('train', 0.8)

    const datasets = h.getDatasets()
    expect(datasets).toHaveLength(1)
    expect(datasets[0].points.length).toBeGreaterThanOrEqual(1)
  })

  it('computes bounds', () => {
    const h = new FullHistory()
    h.push('train', 3)
    h.push('train', 7)

    const bounds = h.getBounds()
    expect(bounds.yMin).toBe(3)
    expect(bounds.yMax).toBe(7)
  })

  it('computes bars (averages)', () => {
    const h = new FullHistory()
    h.push('train', 2)
    h.push('train', 4)

    const bars = h.getBars()
    expect(bars).toHaveLength(1)
    expect(bars[0].split).toBe('train')
    expect(bars[0].avg).toBe(3)
  })

  it('downsamples when exceeding maxSamples', () => {
    const h = new FullHistory(10)
    for (let i = 0; i < 50; i++) {
      h.push('train', i)
    }

    const datasets = h.getDatasets()
    expect(datasets[0].points.length).toBeLessThanOrEqual(10)
  })
})

describe('NumericMetricsState', () => {
  it('tracks metrics and supports navigation', () => {
    const state = new NumericMetricsState()
    state.push('Loss', 'train', 0.5)
    state.push('Accuracy', 'train', 90)

    expect(state.names).toEqual(['Loss', 'Accuracy'])
    expect(state.selected).toBe(0)
    expect(state.current?.name).toBe('Loss')

    state.nextMetric()
    expect(state.selected).toBe(1)
    expect(state.current?.name).toBe('Accuracy')

    state.nextMetric()
    expect(state.selected).toBe(0) // wraps around

    state.prevMetric()
    expect(state.selected).toBe(1) // wraps backward
  })

  it('cycles plot kinds', () => {
    const state = new NumericMetricsState()
    expect(state.plotKind).toBe('full')

    state.switchKind()
    expect(state.plotKind).toBe('recent')

    state.switchKind()
    expect(state.plotKind).toBe('summary')

    state.switchKind()
    expect(state.plotKind).toBe('full')
  })

  it('returns null for current when empty', () => {
    const state = new NumericMetricsState()
    expect(state.current).toBeNull()
  })
})

describe('TextMetricsState', () => {
  it('stores and retrieves text entries', () => {
    const state = new TextMetricsState()
    state.push('loss', 'train', '0.1234')
    state.push('loss', 'valid', '0.2345')
    state.push('accuracy', 'train', '95.50%')

    const lines = state.getLines()
    expect(lines).toHaveLength(2)
    expect(lines[0].name).toBe('loss')
    expect(lines[0].values).toHaveLength(2)
    expect(lines[1].name).toBe('accuracy')
    expect(lines[1].values[0].formatted).toBe('95.50%')
  })

  it('overwrites previous value for same name+split', () => {
    const state = new TextMetricsState()
    state.push('loss', 'train', '0.5')
    state.push('loss', 'train', '0.3')

    const lines = state.getLines()
    expect(lines[0].values[0].formatted).toBe('0.3')
  })
})

describe('ProgressState', () => {
  it('tracks progress values', () => {
    const state = new ProgressState()
    expect(state.progressTotal).toBe(0)
    expect(state.progressTask).toBe(0)

    state.update('train', 0.5, 0.8)
    expect(state.progressTotal).toBe(0.5)
    expect(state.progressTask).toBe(0.8)
    expect(state.split).toBe('train')
  })

  it('clamps values to [0, 1]', () => {
    const state = new ProgressState()
    state.update('train', 1.5, -0.5)
    expect(state.progressTotal).toBe(1)
    expect(state.progressTask).toBe(0)
  })

  it('returns --- for eta before warmup', () => {
    const state = new ProgressState()
    expect(state.eta).toBe('---')
  })
})

describe('StatusState', () => {
  it('tracks mode and entries', () => {
    const state = new StatusState()
    expect(state.mode).toBe('train')

    state.update('valid', [{ tag: 'Epoch', value: '3/10' }])
    expect(state.mode).toBe('valid')
    expect(state.entries).toHaveLength(1)
    expect(state.entries[0].tag).toBe('Epoch')
  })
})
