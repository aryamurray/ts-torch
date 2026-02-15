// ─────────────────────────────────────────────────────────────
// Dashboard — main renderer, assembles layout & render loop
// Mirrors Burn's MetricsView layout: base.rs
// ─────────────────────────────────────────────────────────────

import { readSync, openSync, closeSync, constants as fsConstants } from 'node:fs'
import { ReadStream } from 'node:tty'
import { ansi } from './ansi.js'
import { Buffer } from './buffer.js'
import { Constraint, splitRect, type Rect } from './layout.js'
import {
  drawBlock,
  drawParagraph,
  drawGauge,
  drawTabs,
  drawChart,
  drawBarChart,
  Color,
  type ParagraphLine,
  type ChartDataset,
  type ChartAxes,
  type BarData,
} from './widgets.js'
import { NumericMetricsState, TextMetricsState, ProgressState, StatusState, splitColor } from './data.js'

// ── Dashboard class ──

export interface DashboardOptions {
  title?: string
  refreshRate?: number // ms, default 100
  onQuit?: () => void
}

export class Dashboard {
  readonly numericMetrics = new NumericMetricsState()
  readonly textMetrics = new TextMetricsState()
  readonly progress = new ProgressState()
  readonly status = new StatusState()

  /** Set to true when the user presses 'q'. Check this for cooperative quit. */
  quitRequested = false

  private opts: DashboardOptions
  private _title: string
  private refreshRate: number
  private interval: ReturnType<typeof setInterval> | null = null
  private destroyed = false
  private stdinFd: number | null = null
  private stdinBuf = globalThis.Buffer.alloc(32)
  private ttyStream: ReadStream | null = null

  constructor(opts: DashboardOptions = {}) {
    this.opts = opts
    this._title = opts.title ?? 'ts-torch'
    this.refreshRate = opts.refreshRate ?? 100
  }

  /** Start the render loop. Call this once. */
  start() {
    // Enter alternate screen, hide cursor
    process.stdout.write(ansi.enterAlt() + ansi.hideCursor() + ansi.clear())

    if (process.stdin.isTTY) {
      // In-process: use stdin events for keyboard input
      process.stdin.setRawMode(true)
      process.stdin.resume()
      process.stdin.setEncoding('utf8')
      process.stdin.on('data', (key: string) => this.handleKey(key))
    }

    // Open /dev/tty for non-blocking keyboard polling.
    // Works both in-process (for blocked event loops) and in child processes (where stdin is /dev/null).
    try {
      this.stdinFd = openSync('/dev/tty', fsConstants.O_RDONLY | fsConstants.O_NONBLOCK)
      // If stdin isn't a TTY (child process), set raw mode via /dev/tty directly
      if (!process.stdin.isTTY) {
        const ttyFd = openSync('/dev/tty', 'r')
        this.ttyStream = new ReadStream(ttyFd)
        this.ttyStream.setRawMode(true)
        this.ttyStream.pause()
      }
    } catch {
      // No TTY available (e.g. CI) — keyboard input won't work
    }

    // Handle resize
    process.stdout.on('resize', () => this.render())

    // Handle cleanup
    const cleanup = () => this.destroy()
    process.on('exit', cleanup)
    process.on('SIGINT', () => {
      cleanup()
      process.exit(0)
    })
    process.on('SIGTERM', () => {
      cleanup()
      process.exit(0)
    })

    this.interval = setInterval(() => {
      this.pollStdin()
      this.render()
    }, this.refreshRate)
    this.render()
  }

  /** Stop rendering and restore terminal. */
  destroy() {
    if (this.destroyed) return
    this.destroyed = true
    if (this.interval) clearInterval(this.interval)
    if (this.stdinFd !== null) {
      try {
        closeSync(this.stdinFd)
      } catch {
        /* ignore */
      }
      this.stdinFd = null
    }
    if (this.ttyStream) {
      try {
        this.ttyStream.setRawMode(false)
        this.ttyStream.destroy()
      } catch {
        /* ignore */
      }
      this.ttyStream = null
    }
    if (process.stdin.isTTY) {
      process.stdin.setRawMode(false)
      process.stdin.pause()
    }

    process.stdout.write(ansi.showCursor() + ansi.leaveAlt())
  }

  // ── Input handling ──

  private handleKey(key: string) {
    switch (key) {
      case 'q':
        this.quitRequested = true
        this.opts.onQuit?.()
        break
      case '\x03': // Ctrl+C
        this.destroy()
        process.exit(0)
        break
      case '\x1b[C': // Right arrow
        this.numericMetrics.nextMetric()
        break
      case '\x1b[D': // Left arrow
        this.numericMetrics.prevMetric()
        break
      case '\x1b[A': // Up arrow
      case '\x1b[B': // Down arrow
        this.numericMetrics.switchKind()
        break
    }
  }

  // ── Render ──

  private lastRenderTime = 0

  /**
   * Request a synchronous render. Throttled to at most once per refreshRate ms.
   * Use this when the event loop is blocked (e.g. RL training) and setInterval can't fire.
   * Also polls stdin for keypresses since event-based input can't fire in a blocked loop.
   */
  requestRender() {
    this.pollStdin()
    const now = Date.now()
    if (now - this.lastRenderTime >= this.refreshRate) {
      this.render()
    }
  }

  /** Non-blocking read from /dev/tty to handle keypresses when the event loop is blocked. */
  private pollStdin() {
    if (this.stdinFd === null) return
    try {
      const n = readSync(this.stdinFd, this.stdinBuf, 0, this.stdinBuf.length, null)
      if (n > 0) {
        const str = this.stdinBuf.toString('utf8', 0, n)
        this.handleKey(str)
      }
    } catch {
      // EAGAIN / EWOULDBLOCK — no data available, which is fine
    }
  }

  private render() {
    if (this.destroyed) return
    this.lastRenderTime = Date.now()

    const cols = process.stdout.columns || 80
    const rows = process.stdout.rows || 24
    const buf = new Buffer(cols, rows)
    const fullRect: Rect = { row: 1, col: 1, width: cols, height: rows }

    this.renderLayout(buf, fullRect)
    buf.flush((data) => process.stdout.write(data))
  }

  /**
   * Main layout — mirrors base.rs MetricsView::render
   *
   * ┌──────────────────────────────────────────────┐
   * │ ┌─ Controls ──┐  ┌─ Plots ─────────────────┐ │
   * │ │             │  │  Tabs: Loss │ Accuracy   │ │
   * │ ├─ Metrics ───┤  │  [chart area]            │ │
   * │ │ Loss: 0.12  │  │                          │ │
   * │ │ Acc:  0.95  │  │     ⠁⠂⠄⡀⢀⠠⠐⠈            │ │
   * │ ├─ Status ────┤  │                          │ │
   * │ │ Mode: Train │  └──────────────────────────┘ │
   * │ └─────────────┘                               │
   * │ ┌─ Progress ──────────────────────────────────┐│
   * │ │ ████████████░░░░░░░░░  42%    (3 mins)    ││
   * │ └─────────────────────────────────────────────┘│
   * └──────────────────────────────────────────────┘
   */
  private renderLayout(buf: Buffer, size: Rect) {
    // Split: main area + progress bar at bottom
    const mainSplit = splitRect(size, 'vertical', [Constraint.min(16), Constraint.max(4)])
    const mainArea = mainSplit[0]!
    const progressArea = mainSplit[1]!

    // Split main: left panel (38%) + right panel (62%)
    const hSplit = splitRect(mainArea, 'horizontal', [Constraint.percentage(38), Constraint.percentage(62)])
    const leftPanel = hSplit[0]!
    const rightPanel = hSplit[1]!

    // Left panel: controls (5) + metrics (flex) + status (6)
    const leftSplit = splitRect(leftPanel, 'vertical', [Constraint.max(5), Constraint.min(6), Constraint.max(6)])
    const controlsArea = leftSplit[0]!
    const metricsArea = leftSplit[1]!
    const statusArea = leftSplit[2]!

    this.renderControls(buf, controlsArea)
    this.renderTextMetrics(buf, metricsArea)
    this.renderStatus(buf, statusArea)
    this.renderNumericMetrics(buf, rightPanel)
    this.renderProgress(buf, progressArea)
  }

  // ── Controls panel ──

  private renderControls(buf: Buffer, rect: Rect) {
    const inner = drawBlock(buf, rect, { title: this._title, style: { fg: Color.gray } })
    drawParagraph(buf, inner, [
      {
        spans: [
          { text: ' Quit          : ', style: { fg: Color.yellow, bold: true } },
          { text: 'q  ', style: { bold: true } },
          { text: '  Stop the training.', style: { italic: true, fg: Color.gray } },
        ],
      },
      {
        spans: [
          { text: ' Plots Metrics : ', style: { fg: Color.yellow, bold: true } },
          { text: '← →', style: { bold: true } },
          { text: '  Switch between metrics.', style: { italic: true, fg: Color.gray } },
        ],
      },
      {
        spans: [
          { text: ' Plots Type    : ', style: { fg: Color.yellow, bold: true } },
          { text: '↑ ↓', style: { bold: true } },
          { text: '  Switch between types.', style: { italic: true, fg: Color.gray } },
        ],
      },
    ])
  }

  // ── Text metrics panel ──

  private renderTextMetrics(buf: Buffer, rect: Rect) {
    const inner = drawBlock(buf, rect, { title: 'Metrics', style: { fg: Color.gray } })
    const lines: ParagraphLine[] = []

    for (const { name, values } of this.textMetrics.getLines()) {
      lines.push({ spans: [{ text: ` ${name} `, style: { fg: Color.yellow, bold: true } }] })
      for (const { split, formatted } of values) {
        const splitLabel = split.charAt(0).toUpperCase() + split.slice(1)
        lines.push({
          spans: [
            { text: ` ${splitLabel} `, style: { bold: true } },
            { text: formatted, style: { italic: true } },
          ],
        })
      }
      lines.push({ spans: [{ text: '' }] })
    }

    drawParagraph(buf, inner, lines)
  }

  // ── Status panel ──

  private renderStatus(buf: Buffer, rect: Rect) {
    const inner = drawBlock(buf, rect, { title: 'Status', style: { fg: Color.gray } })
    const mode = this.status.mode.charAt(0).toUpperCase() + this.status.mode.slice(1)
    const modeMap: Record<string, string> = { train: 'Training', valid: 'Validating', test: 'Evaluation' }

    const lines: ParagraphLine[] = [
      {
        spans: [
          { text: ` Mode : `, style: { fg: Color.yellow, bold: true } },
          { text: modeMap[this.status.mode] ?? mode, style: { italic: true } },
        ],
      },
    ]

    for (const entry of this.status.entries) {
      lines.push({
        spans: [
          { text: ` ${entry.tag} : `, style: { fg: Color.yellow, bold: true } },
          { text: entry.value, style: { italic: true } },
        ],
      })
    }

    drawParagraph(buf, inner, lines)
  }

  // ── Numeric metrics (plots) panel ──

  private renderNumericMetrics(buf: Buffer, rect: Rect) {
    const current = this.numericMetrics.current
    if (!current) {
      drawBlock(buf, rect, { title: 'Plots', style: { fg: Color.gray } })
      return
    }

    const kind = this.numericMetrics.plotKind
    const blockTitle = kind === 'summary' ? 'Summary' : 'Plots'
    const inner = drawBlock(buf, rect, { title: blockTitle, style: { fg: Color.gray } })

    // Tabs row + plot type label + chart
    const plotSplit = splitRect(inner, 'vertical', [Constraint.length(2), Constraint.length(1), Constraint.min(0)])
    const tabsArea = plotSplit[0]!
    const labelArea = plotSplit[1]!
    const chartArea = plotSplit[2]!

    // Tabs
    drawTabs(buf, tabsArea, {
      titles: this.numericMetrics.names,
      selected: this.numericMetrics.selected,
    })

    // Plot type label
    const kindLabel = kind === 'full' ? 'Full History' : kind === 'recent' ? 'Recent History' : 'Summary'
    const labelRow = labelArea.row - 1
    const labelCol = labelArea.col - 1 + Math.floor((labelArea.width - kindLabel.length) / 2)
    buf.putStr(labelRow, labelCol, kindLabel, { fg: Color.white, bold: true })

    // Chart
    if (kind === 'summary') {
      const bars = current.full.getBars()
      const barData: BarData[] = bars.map((b) => ({
        label: b.split.charAt(0).toUpperCase() + b.split.slice(1),
        value: b.avg,
        color: splitColor(b.split),
        displayValue: b.avg.toFixed(4),
      }))
      drawBarChart(buf, chartArea, barData)
    } else {
      const source = kind === 'full' ? current.full : current.recent
      const datasets = source.getDatasets()
      const bounds = source.getBounds()

      const chartDatasets: ChartDataset[] = datasets.map((ds) => ({
        name: ds.split.charAt(0).toUpperCase() + ds.split.slice(1),
        data: ds.points,
        color: splitColor(ds.split),
        lineType: kind === 'full' ? 'line' : 'scatter',
      }))

      const axes: ChartAxes = {
        xBounds: [bounds.xMin, bounds.xMax],
        yBounds: [bounds.yMin, bounds.yMax],
        xLabels: [Math.floor(bounds.xMin).toString(), Math.floor(bounds.xMax).toString()],
        yLabels: [formatFloat(bounds.yMin), formatFloat(bounds.yMax)],
      }

      drawChart(buf, chartArea, chartDatasets, axes)
    }
  }

  // ── Progress bar panel ──

  private renderProgress(buf: Buffer, rect: Rect) {
    const inner = drawBlock(buf, rect, { title: 'Progress', style: { fg: Color.gray } })

    // Split: task progress + total progress
    const progressSplit = splitRect(inner, 'vertical', [Constraint.ratio(1, 2), Constraint.ratio(1, 2)])
    const taskRow = progressSplit[0]!
    const totalRow = progressSplit[1]!

    // ETA label
    const eta = this.progress.eta
    const etaStr = `(${eta})`
    const etaWidth = etaStr.length + 2 // 1 space padding each side

    // Gauge width: inner width minus space reserved for ETA
    const gaugeWidth = Math.max(0, taskRow.width - etaWidth)

    // Task progress gauge
    const taskGaugeRect: Rect = { ...taskRow, width: gaugeWidth }
    drawGauge(buf, taskGaugeRect, {
      ratio: this.progress.progressTask,
      color: splitColor(this.progress.split),
    })

    // Total progress gauge
    const totalGaugeRect: Rect = { ...totalRow, width: gaugeWidth }
    drawGauge(buf, totalGaugeRect, {
      ratio: this.progress.progressTotal,
      color: Color.yellow,
    })

    // ETA text — right-aligned in the total row
    const etaCol = totalRow.col - 1 + totalRow.width - etaStr.length - 1
    buf.putStr(totalRow.row - 1, etaCol, etaStr, { fg: Color.gray, italic: true })
  }
}

// ── Helpers ──

function formatFloat(value: number): string {
  if (!Number.isFinite(value)) return '0'
  const abs = Math.abs(value)
  if (abs >= 100) return value.toFixed(0)
  if (abs >= 10) return value.toFixed(1)
  if (abs >= 1) return value.toFixed(2)
  if (abs >= 0.01) return value.toFixed(3)
  return value.toExponential(1)
}
