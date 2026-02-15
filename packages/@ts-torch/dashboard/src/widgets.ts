// ─────────────────────────────────────────────────────────────
// Widgets — immediate-mode widgets that render into a Buffer
// ─────────────────────────────────────────────────────────────

import { Buffer, type Cell } from './buffer.js'
import { type Rect, innerRect } from './layout.js'
import { ansi } from './ansi.js'

type Style = Partial<Omit<Cell, 'char'>>

// ── Colors ──

export const Color = {
  gray: ansi.fg(128, 128, 128),
  darkGray: ansi.fg(80, 80, 80),
  white: ansi.fg(220, 220, 220),
  yellow: ansi.fg(230, 200, 60),
  lightYellow: ansi.fg(255, 255, 120),
  red: ansi.fg(220, 60, 60),
  lightRed: ansi.fg(255, 100, 100),
  blue: ansi.fg(80, 120, 220),
  lightBlue: ansi.fg(120, 160, 255),
  green: ansi.fg(60, 180, 60),
  lightGreen: ansi.fg(100, 220, 100),
  cyan: ansi.fg(80, 200, 200),
} as const

// ── Box-drawing characters ──

const BOX = {
  topLeft: '┌',
  topRight: '┐',
  bottomLeft: '└',
  bottomRight: '┘',
  horizontal: '─',
  vertical: '│',
} as const

// ── Block widget ──

export interface BlockOptions {
  title?: string
  titleAlignment?: 'left' | 'center' | 'right'
  borders?: boolean
  style?: Style
}

/**
 * Draw a bordered block, return inner rect for content.
 */
export function drawBlock(buf: Buffer, rect: Rect, opts: BlockOptions = {}): Rect {
  const { title, titleAlignment = 'left', borders = true, style = {} } = opts
  const r0 = rect.row - 1 // 0-based
  const c0 = rect.col - 1
  const borderStyle: Style = { fg: style.fg || Color.gray, ...style }

  if (!borders) return rect

  // Corners
  buf.set(r0, c0, BOX.topLeft, borderStyle)
  buf.set(r0, c0 + rect.width - 1, BOX.topRight, borderStyle)
  buf.set(r0 + rect.height - 1, c0, BOX.bottomLeft, borderStyle)
  buf.set(r0 + rect.height - 1, c0 + rect.width - 1, BOX.bottomRight, borderStyle)

  // Horizontal edges
  for (let c = 1; c < rect.width - 1; c++) {
    buf.set(r0, c0 + c, BOX.horizontal, borderStyle)
    buf.set(r0 + rect.height - 1, c0 + c, BOX.horizontal, borderStyle)
  }

  // Vertical edges
  for (let r = 1; r < rect.height - 1; r++) {
    buf.set(r0 + r, c0, BOX.vertical, borderStyle)
    buf.set(r0 + r, c0 + rect.width - 1, BOX.vertical, borderStyle)
  }

  // Title
  if (title) {
    const maxW = rect.width - 4
    const t = title.length > maxW ? title.slice(0, maxW) : title
    let col: number
    if (titleAlignment === 'center') {
      col = c0 + Math.floor((rect.width - t.length) / 2)
    } else if (titleAlignment === 'right') {
      col = c0 + rect.width - t.length - 2
    } else {
      col = c0 + 1
    }
    buf.putStr(r0, col, t, { fg: Color.white, bold: true })
  }

  return innerRect(rect)
}

// ── Paragraph widget ──

export interface ParagraphLine {
  spans: { text: string; style?: Style }[]
}

export function drawParagraph(buf: Buffer, rect: Rect, lines: ParagraphLine[]) {
  const r0 = rect.row - 1
  const c0 = rect.col - 1

  for (let i = 0; i < lines.length && i < rect.height; i++) {
    let col = c0
    for (const span of lines[i]!.spans) {
      const written = buf.putStr(r0 + i, col, span.text, span.style || {})
      col += written
    }
  }
}

// ── Gauge (progress bar) ──

export interface GaugeOptions {
  ratio: number // 0..1
  color: string
  label?: string
}

export function drawGauge(buf: Buffer, rect: Rect, opts: GaugeOptions) {
  const r0 = rect.row - 1
  const c0 = rect.col - 1
  const ratio = Math.max(0, Math.min(1, opts.ratio))
  const filled = Math.round(ratio * rect.width)
  const pctText = `${Math.round(ratio * 100)}%`
  const midCol = Math.floor((rect.width - pctText.length) / 2)

  const bgColor = opts.color.replace('38;2', '48;2')
  for (let c = 0; c < rect.width; c++) {
    const isFilled = c < filled
    const isText = c >= midCol && c < midCol + pctText.length
    if (isText) {
      buf.set(r0, c0 + c, pctText[c - midCol]!, {
        fg: isFilled ? Color.white : Color.gray,
        bg: isFilled ? bgColor : '',
        bold: true,
      })
    } else {
      buf.set(r0, c0 + c, isFilled ? '█' : '░', {
        fg: isFilled ? opts.color : Color.darkGray,
        bg: '',
      })
    }
  }
}

// ── Tabs widget ──

export interface TabsOptions {
  titles: string[]
  selected: number
  style?: Style
  highlightStyle?: Style
}

export function drawTabs(buf: Buffer, rect: Rect, opts: TabsOptions) {
  const r0 = rect.row - 1
  const c0 = rect.col - 1
  let col = c0 + 1

  for (let i = 0; i < opts.titles.length; i++) {
    const isSelected = i === opts.selected
    const title = opts.titles[i]!
    const style: Style = isSelected
      ? { fg: Color.lightYellow, bold: true, ...opts.highlightStyle }
      : { fg: Color.yellow, ...(opts.style || {}) }

    buf.putStr(r0, col, title, style)
    col += title.length

    // Separator
    if (i < opts.titles.length - 1) {
      buf.putStr(r0, col, ' │ ', { fg: Color.darkGray })
      col += 3
    }
  }
}

// ── Sparkline (recent history, single row) ──

const SPARK_CHARS = '▁▂▃▄▅▆▇█'

export function drawSparkline(buf: Buffer, rect: Rect, data: number[], color: string) {
  const r0 = rect.row - 1
  const c0 = rect.col - 1
  const width = rect.width
  const slice = data.slice(-width)
  if (slice.length === 0) return

  const min = Math.min(...slice)
  const max = Math.max(...slice)
  const range = max - min || 1

  for (let i = 0; i < slice.length; i++) {
    const idx = Math.floor(((slice[i]! - min) / range) * 7)
    buf.set(r0, c0 + i, SPARK_CHARS[idx]!, { fg: color })
  }
}

// ── Braille scatter/line chart (multi-series, like ratatui Chart) ──

// Braille dot patterns: each 2×4 cell block
// (col_offset, row_offset) → bit
const BRAILLE_BASE = 0x2800
const BRAILLE_MAP: [number, number, number][] = [
  [0, 0, 0x01],
  [0, 1, 0x02],
  [0, 2, 0x04],
  [1, 0, 0x08],
  [1, 1, 0x10],
  [1, 2, 0x20],
  [0, 3, 0x40],
  [1, 3, 0x80],
]

export interface ChartDataset {
  name: string
  data: [number, number][] // [x, y][]
  color: string
  lineType?: 'scatter' | 'line'
}

export interface ChartAxes {
  xBounds: [number, number]
  yBounds: [number, number]
  xLabels?: string[]
  yLabels?: string[]
}

export function drawChart(buf: Buffer, rect: Rect, datasets: ChartDataset[], axes: ChartAxes) {
  if (rect.width < 4 || rect.height < 4) return

  const r0 = rect.row - 1
  const c0 = rect.col - 1

  // Reserve space for axis labels
  const yLabelWidth = axes.yLabels ? Math.max(...axes.yLabels.map((l) => l.length)) + 1 : 0
  const xLabelHeight = axes.xLabels ? 1 : 0

  const plotCol = c0 + yLabelWidth
  const plotWidth = rect.width - yLabelWidth
  const plotHeight = rect.height - xLabelHeight - 1 // -1 for x axis line

  if (plotWidth < 2 || plotHeight < 2) return

  // Draw axis lines
  for (let c = 0; c < plotWidth; c++) {
    buf.set(r0 + plotHeight, plotCol + c, '─', { fg: Color.darkGray })
  }
  for (let r = 0; r < plotHeight; r++) {
    buf.set(r0 + r, plotCol, '│', { fg: Color.darkGray })
  }
  buf.set(r0 + plotHeight, plotCol, '└', { fg: Color.darkGray })

  // Y labels
  if (axes.yLabels && axes.yLabels.length >= 2) {
    buf.putStr(r0, c0, axes.yLabels[1]!, { fg: Color.gray, bold: true })
    buf.putStr(r0 + plotHeight - 1, c0, axes.yLabels[0]!, {
      fg: Color.gray,
      bold: true,
    })
  }

  // X labels
  if (axes.xLabels && axes.xLabels.length >= 2) {
    const xRow = r0 + plotHeight + 1
    buf.putStr(xRow, plotCol + 1, axes.xLabels[0]!, { fg: Color.gray, bold: true })
    const endLabel = axes.xLabels[1]!
    buf.putStr(xRow, plotCol + plotWidth - endLabel.length, endLabel, {
      fg: Color.gray,
      bold: true,
    })
  }

  // Braille canvas: each terminal cell = 2 cols × 4 rows of dots
  const canvasW = plotWidth - 1 // exclude axis line
  const canvasH = plotHeight
  const dotW = canvasW * 2
  const dotH = canvasH * 4

  // For each dataset, compute braille patterns
  for (const ds of datasets) {
    const grid = new Map<string, number>() // "cellR,cellC" → braille bits

    const [xMin, xMax] = axes.xBounds
    const [yMin, yMax] = axes.yBounds
    const xRange = xMax - xMin || 1
    const yRange = yMax - yMin || 1

    const points = ds.data
    for (let i = 0; i < points.length; i++) {
      const [x, y] = points[i]!
      const dotX = Math.floor(((x - xMin) / xRange) * (dotW - 1))
      const dotY = Math.floor(((yMax - y) / yRange) * (dotH - 1)) // Y flipped

      const cellC = Math.floor(dotX / 2)
      const cellR = Math.floor(dotY / 4)
      const subX = dotX % 2
      const subY = dotY % 4

      // Find matching braille bit
      for (const [bx, by, bit] of BRAILLE_MAP) {
        if (bx === subX && by === subY) {
          const key = `${cellR},${cellC}`
          grid.set(key, (grid.get(key) || 0) | bit)
          break
        }
      }

      // Line mode: connect consecutive points
      if (ds.lineType === 'line' && i > 0) {
        const [px, py] = points[i - 1]!
        const prevDotX = Math.floor(((px - xMin) / xRange) * (dotW - 1))
        const prevDotY = Math.floor(((yMax - py) / yRange) * (dotH - 1))
        interpolateDots(prevDotX, prevDotY, dotX, dotY, grid, dotW, dotH)
      }
    }

    // Render braille chars
    for (const [key, bits] of grid) {
      const parts = key.split(',').map(Number)
      const cr = parts[0]!
      const cc = parts[1]!
      if (cr >= 0 && cr < canvasH && cc >= 0 && cc < canvasW) {
        const existing = buf.get(r0 + cr, plotCol + 1 + cc)
        let combinedBits = bits
        if (existing && existing.char.codePointAt(0)! >= BRAILLE_BASE) {
          combinedBits |= existing.char.codePointAt(0)! - BRAILLE_BASE
        }
        buf.set(r0 + cr, plotCol + 1 + cc, String.fromCodePoint(BRAILLE_BASE + combinedBits), {
          fg: ds.color,
        })
      }
    }
  }

  // Legend (right side)
  const legendCol = plotCol + plotWidth - 20
  if (legendCol > plotCol + 5) {
    for (let i = 0; i < datasets.length; i++) {
      const ds = datasets[i]!
      buf.putStr(r0 + 1 + i, legendCol, `■ ${ds.name}`, {
        fg: ds.color,
        bold: true,
      })
    }
  }
}

/** Bresenham line interpolation in dot-space */
function interpolateDots(
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  grid: Map<string, number>,
  maxW: number,
  maxH: number,
) {
  const dx = Math.abs(x1 - x0)
  const dy = Math.abs(y1 - y0)
  const sx = x0 < x1 ? 1 : -1
  const sy = y0 < y1 ? 1 : -1
  let err = dx - dy
  let cx = x0,
    cy = y0
  const steps = Math.max(dx, dy)
  // Limit interpolation steps to avoid perf issues
  const maxSteps = Math.min(steps, 200)
  const skip = steps > maxSteps ? Math.ceil(steps / maxSteps) : 1
  let count = 0

  while (true) {
    if (count % skip === 0) {
      const cellC = Math.floor(cx / 2)
      const cellR = Math.floor(cy / 4)
      const subX = cx % 2
      const subY = cy % 4

      if (cellC >= 0 && cellC < Math.ceil(maxW / 2) && cellR >= 0 && cellR < Math.ceil(maxH / 4)) {
        for (const [bx, by, bit] of BRAILLE_MAP) {
          if (bx === subX && by === subY) {
            const key = `${cellR},${cellC}`
            grid.set(key, (grid.get(key) || 0) | bit)
            break
          }
        }
      }
    }

    if (cx === x1 && cy === y1) break
    const e2 = 2 * err
    if (e2 > -dy) {
      err -= dy
      cx += sx
    }
    if (e2 < dx) {
      err += dx
      cy += sy
    }
    count++
    if (count > steps + 1) break
  }
}

// ── Bar chart widget ──

export interface BarData {
  label: string
  value: number
  color: string
  displayValue?: string
}

export function drawBarChart(buf: Buffer, rect: Rect, bars: BarData[], maxValue?: number) {
  if (bars.length === 0 || rect.height < 4) return

  const r0 = rect.row - 1
  const c0 = rect.col - 1
  const chartHeight = rect.height - 2 // Reserve for labels + values
  const max = maxValue ?? Math.max(...bars.map((b) => b.value), 1)

  const barWidth = Math.max(3, Math.floor((rect.width - 2) / bars.length) - 2)
  let col = c0 + 2

  for (const bar of bars) {
    const barHeight = Math.round((bar.value / max) * chartHeight)
    // Draw bar
    for (let r = 0; r < barHeight; r++) {
      const row = r0 + chartHeight - r
      for (let c = 0; c < barWidth; c++) {
        buf.set(row, col + c, '█', { fg: bar.color })
      }
    }
    // Value above bar
    const valStr = bar.displayValue ?? bar.value.toFixed(2)
    buf.putStr(r0 + chartHeight - barHeight, col, valStr, {
      fg: bar.color,
      bold: true,
    })
    // Label below
    buf.putStr(r0 + chartHeight + 1, col, bar.label.slice(0, barWidth), {
      fg: Color.gray,
    })

    col += barWidth + 2
  }
}
