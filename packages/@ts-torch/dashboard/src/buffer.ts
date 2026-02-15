// ─────────────────────────────────────────────────────────────
// Buffer — cell-based screen buffer (ratatui::Buffer equivalent)
// ─────────────────────────────────────────────────────────────

import { ansi } from './ansi.js'
import { type Rect } from './layout.js'

export interface Cell {
  char: string
  fg: string // ANSI fg sequence or ''
  bg: string // ANSI bg sequence or ''
  bold: boolean
  italic: boolean
  dim: boolean
}

function emptyCell(): Cell {
  return { char: ' ', fg: '', bg: '', bold: false, italic: false, dim: false }
}

/**
 * A 2D cell buffer that widgets render into.
 * After all widgets render, the buffer is flushed to stdout in one write.
 */
export class Buffer {
  readonly width: number
  readonly height: number
  private cells: Cell[]

  constructor(width: number, height: number) {
    this.width = width
    this.height = height
    this.cells = new Array(width * height)
    for (let i = 0; i < this.cells.length; i++) {
      this.cells[i] = emptyCell()
    }
  }

  /** Get cell at (row, col) — 0-based. */
  get(row: number, col: number): Cell | undefined {
    if (row < 0 || row >= this.height || col < 0 || col >= this.width) return undefined
    return this.cells[row * this.width + col]
  }

  /** Set a single character with style at (row, col) — 0-based. */
  set(row: number, col: number, char: string, style: Partial<Omit<Cell, 'char'>> = {}) {
    if (row < 0 || row >= this.height || col < 0 || col >= this.width) return
    const cell = this.cells[row * this.width + col]!
    cell.char = char
    if (style.fg !== undefined) cell.fg = style.fg
    if (style.bg !== undefined) cell.bg = style.bg
    if (style.bold !== undefined) cell.bold = style.bold
    if (style.italic !== undefined) cell.italic = style.italic
    if (style.dim !== undefined) cell.dim = style.dim
  }

  /**
   * Write a string into the buffer at (row, col), applying style.
   * Returns number of chars written.
   */
  putStr(row: number, col: number, text: string, style: Partial<Omit<Cell, 'char'>> = {}): number {
    let written = 0
    for (let i = 0; i < text.length; i++) {
      if (col + i >= this.width) break
      this.set(row, col + i, text[i]!, style)
      written++
    }
    return written
  }

  /** Fill a rect region with a character and style. */
  fill(rect: Rect, char = ' ', style: Partial<Omit<Cell, 'char'>> = {}) {
    const r0 = rect.row - 1 // convert to 0-based
    const c0 = rect.col - 1
    for (let r = r0; r < r0 + rect.height && r < this.height; r++) {
      for (let c = c0; c < c0 + rect.width && c < this.width; c++) {
        this.set(r, c, char, style)
      }
    }
  }

  /**
   * Flush buffer to stdout as a single write (minimal flicker).
   * Uses diff-style: emits ANSI moves + style changes only when needed.
   */
  flush(writeFn?: (data: string) => void): void {
    const parts: string[] = [ansi.hideCursor()]
    let lastFg = ''
    let lastBg = ''
    let lastBold = false
    let lastItalic = false
    let lastDim = false

    for (let r = 0; r < this.height; r++) {
      parts.push(ansi.moveTo(r + 1, 1))
      for (let c = 0; c < this.width; c++) {
        const cell = this.cells[r * this.width + c]!

        // Build style delta
        const needsReset = (lastBold && !cell.bold) || (lastItalic && !cell.italic) || (lastDim && !cell.dim)

        if (needsReset) {
          parts.push(ansi.reset)
          lastFg = ''
          lastBg = ''
          lastBold = false
          lastItalic = false
          lastDim = false
        }

        if (cell.fg !== lastFg) {
          parts.push(cell.fg || ansi.resetFg)
          lastFg = cell.fg
        }
        if (cell.bg !== lastBg) {
          parts.push(cell.bg || ansi.resetBg)
          lastBg = cell.bg
        }
        if (cell.bold && !lastBold) {
          parts.push(ansi.bold)
          lastBold = true
        }
        if (cell.italic && !lastItalic) {
          parts.push(ansi.italic)
          lastItalic = true
        }
        if (cell.dim && !lastDim) {
          parts.push(ansi.dim)
          lastDim = true
        }

        parts.push(cell.char)
      }
    }

    parts.push(ansi.reset)
    const output = parts.join('')
    if (writeFn) {
      writeFn(output)
    } else {
      process.stdout.write(output)
    }
  }
}
