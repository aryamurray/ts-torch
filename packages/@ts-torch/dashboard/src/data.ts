// ─────────────────────────────────────────────────────────────
// Data models — metric state, history tracking, plot data
// ─────────────────────────────────────────────────────────────

export type Split = 'train' | 'valid' | 'test'

export function splitColor(split: Split): string {
  const { fg } = await_import_ansi()
  switch (split) {
    case 'train':
      return fg(255, 100, 100) // lightRed
    case 'valid':
      return fg(120, 160, 255) // lightBlue
    case 'test':
      return fg(100, 220, 100) // lightGreen
  }
}

// Lazy import to avoid circular deps
function await_import_ansi() {
  return { fg: (r: number, g: number, b: number) => `\x1b[38;2;${r};${g};${b}m` }
}

// ── Numeric metric history ──

export class RecentHistory {
  private data: Map<Split, { points: [number, number][]; cursor: number }> = new Map()
  private maxSamples: number
  private step = 0

  constructor(maxSamples = 1000) {
    this.maxSamples = maxSamples
  }

  push(split: Split, value: number) {
    if (!this.data.has(split)) {
      this.data.set(split, { points: [], cursor: 0 })
    }
    const entry = this.data.get(split)!
    entry.points.push([this.step, value])

    // Trim old points
    if (entry.points.length - entry.cursor > this.maxSamples * 2) {
      entry.points = entry.points.slice(entry.cursor)
      entry.cursor = 0
    }
    while (entry.points.length - entry.cursor > this.maxSamples) {
      entry.cursor++
    }

    this.step++
  }

  getDatasets(): { split: Split; points: [number, number][] }[] {
    const result: { split: Split; points: [number, number][] }[] = []
    for (const [split, entry] of this.data) {
      result.push({
        split,
        points: entry.points.slice(entry.cursor),
      })
    }
    return result
  }

  getBounds(): { xMin: number; xMax: number; yMin: number; yMax: number } {
    let xMin = Infinity,
      xMax = -Infinity,
      yMin = Infinity,
      yMax = -Infinity
    for (const [, entry] of this.data) {
      for (let i = entry.cursor; i < entry.points.length; i++) {
        const [x, y] = entry.points[i]!
        if (x < xMin) xMin = x
        if (x > xMax) xMax = x
        if (y < yMin) yMin = y
        if (y > yMax) yMax = y
      }
    }
    return { xMin, xMax, yMin, yMax }
  }
}

export class FullHistory {
  private data: Map<Split, { points: [number, number][]; stepSize: number; sum: number; count: number }> = new Map()
  private maxSamples: number
  private step = 0

  constructor(maxSamples = 250) {
    this.maxSamples = maxSamples
  }

  push(split: Split, value: number) {
    if (!this.data.has(split)) {
      this.data.set(split, { points: [], stepSize: 1, sum: 0, count: 0 })
    }
    const entry = this.data.get(split)!
    entry.sum += value
    entry.count++

    if (this.step % entry.stepSize === 0) {
      entry.points.push([this.step, value])

      // Resize: keep half, double step size (like Burn)
      if (entry.points.length > this.maxSamples) {
        const newPoints: [number, number][] = []
        for (let i = 0; i < entry.points.length; i += 2) {
          newPoints.push(entry.points[i]!)
        }
        entry.points = newPoints
        entry.stepSize *= 2
      }
    }

    this.step++
  }

  getDatasets(): { split: Split; points: [number, number][] }[] {
    const result: { split: Split; points: [number, number][] }[] = []
    for (const [split, entry] of this.data) {
      result.push({ split, points: entry.points })
    }
    return result
  }

  getBounds(): { xMin: number; xMax: number; yMin: number; yMax: number } {
    let xMin = Infinity,
      xMax = -Infinity,
      yMin = Infinity,
      yMax = -Infinity
    for (const [, entry] of this.data) {
      for (const [x, y] of entry.points) {
        if (x < xMin) xMin = x
        if (x > xMax) xMax = x
        if (y < yMin) yMin = y
        if (y > yMax) yMax = y
      }
    }
    return { xMin, xMax, yMin, yMax }
  }

  getBars(): { split: Split; avg: number }[] {
    const result: { split: Split; avg: number }[] = []
    for (const [split, entry] of this.data) {
      if (entry.count > 0) {
        result.push({ split, avg: entry.sum / entry.count })
      }
    }
    return result
  }
}

// ── Numeric metrics state (manages all numeric metrics) ──

export type PlotKind = 'full' | 'recent' | 'summary'

export class NumericMetricsState {
  private metrics: Map<string, { recent: RecentHistory; full: FullHistory }> = new Map()
  private _names: string[] = []
  selected = 0
  plotKind: PlotKind = 'full'

  get names(): string[] {
    return this._names
  }

  push(name: string, split: Split, value: number) {
    if (!this.metrics.has(name)) {
      this.metrics.set(name, {
        recent: new RecentHistory(),
        full: new FullHistory(),
      })
      this._names.push(name)
    }
    const m = this.metrics.get(name)!
    m.recent.push(split, value)
    m.full.push(split, value)
  }

  get current() {
    if (this._names.length === 0) return null
    const name = this._names[this.selected]!
    return { name, ...this.metrics.get(name)! }
  }

  nextMetric() {
    if (this._names.length > 0) {
      this.selected = (this.selected + 1) % this._names.length
    }
  }

  prevMetric() {
    if (this._names.length > 0) {
      this.selected = this.selected > 0 ? this.selected - 1 : this._names.length - 1
    }
  }

  switchKind() {
    const cycle: PlotKind[] = ['full', 'recent', 'summary']
    const idx = cycle.indexOf(this.plotKind)
    this.plotKind = cycle[(idx + 1) % cycle.length]!
  }
}

// ── Text metrics state ──

export interface TextMetricEntry {
  name: string
  split: Split
  formatted: string
}

export class TextMetricsState {
  private entries: Map<string, Map<Split, string>> = new Map()
  private _names: string[] = []

  push(name: string, split: Split, formatted: string) {
    if (!this.entries.has(name)) {
      this.entries.set(name, new Map())
      this._names.push(name)
    }
    this.entries.get(name)!.set(split, formatted)
  }

  getLines(): { name: string; values: { split: Split; formatted: string }[] }[] {
    return this._names.map((name) => {
      const m = this.entries.get(name)!
      const values: { split: Split; formatted: string }[] = []
      for (const [split, formatted] of m) {
        values.push({ split, formatted })
      }
      return { name, values }
    })
  }
}

// ── Progress state ──

export interface ProgressInfo {
  itemsProcessed: number
  itemsTotal: number
}

export class ProgressState {
  progressTotal = 0
  progressTask = 0
  split: Split = 'train'
  private startedAt = Date.now()
  private warmupDone = false
  private warmupTime = 0
  private warmupProgress = 0

  update(split: Split, total: number, task: number) {
    this.split = split
    this.progressTotal = Math.max(0, Math.min(1, total))
    this.progressTask = Math.max(0, Math.min(1, task))

    if (!this.warmupDone && Date.now() - this.startedAt > 10_000) {
      this.warmupDone = true
      this.warmupTime = Date.now()
      this.warmupProgress = total
    }
  }

  get eta(): string {
    if (!this.warmupDone || this.progressTotal <= this.warmupProgress) return '---'
    const elapsed = (Date.now() - this.warmupTime) / 1000
    const progressSinceWarmup = this.progressTotal - this.warmupProgress
    if (progressSinceWarmup <= 0) return '---'
    const totalEstimated = elapsed / progressSinceWarmup
    const remaining = (1 - this.progressTotal) * totalEstimated
    return formatEta(remaining)
  }
}

function formatEta(secs: number): string {
  secs = Math.round(secs)
  if (secs < 0) return '---'
  const d = Math.floor(secs / 86400)
  const h = Math.floor(secs / 3600) % 24
  const m = Math.floor(secs / 60) % 60
  const s = secs % 60
  if (d > 1) return `${d} days`
  if (d === 1) return '1 day'
  if (h > 1) return `${h} hours`
  if (h === 1) return '1 hour'
  if (m > 1) return `${m} mins`
  if (m === 1) return '1 min'
  if (s > 1) return `${s} secs`
  return '1 sec'
}

// ── Status state ──

export interface StatusEntry {
  tag: string
  value: string
}

export class StatusState {
  mode: Split = 'train'
  entries: StatusEntry[] = []

  update(mode: Split, entries: StatusEntry[]) {
    this.mode = mode
    this.entries = entries
  }
}
