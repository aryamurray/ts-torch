// ─────────────────────────────────────────────────────────────
// Layout engine — ratatui-style constraint-based rect splitting
// ─────────────────────────────────────────────────────────────

/** An axis-aligned rectangle in terminal coordinates (1-based rows/cols). */
export interface Rect {
  row: number
  col: number
  width: number
  height: number
}

/** Layout constraint, mirrors ratatui::Constraint */
export type Constraint =
  | { type: 'min'; value: number }
  | { type: 'max'; value: number }
  | { type: 'length'; value: number }
  | { type: 'percentage'; value: number }
  | { type: 'ratio'; num: number; den: number }

export const Constraint = {
  min: (value: number): Constraint => ({ type: 'min', value }),
  max: (value: number): Constraint => ({ type: 'max', value }),
  length: (value: number): Constraint => ({ type: 'length', value }),
  percentage: (value: number): Constraint => ({ type: 'percentage', value }),
  ratio: (num: number, den: number): Constraint => ({ type: 'ratio', num, den }),
}

export type Direction = 'horizontal' | 'vertical'

/**
 * Split a Rect into sub-rects along a direction using constraints.
 * Mirrors ratatui Layout::split() with a simplified solver.
 */
export function splitRect(rect: Rect, direction: Direction, constraints: Constraint[]): Rect[] {
  const total = direction === 'vertical' ? rect.height : rect.width
  const sizes = solveConstraints(constraints, total)

  let offset = 0
  return sizes.map((size) => {
    const r: Rect =
      direction === 'vertical'
        ? { row: rect.row + offset, col: rect.col, width: rect.width, height: size }
        : { row: rect.row, col: rect.col + offset, width: size, height: rect.height }
    offset += size
    return r
  })
}

/**
 * Shrink a rect by 1 on each side (for block borders).
 */
export function innerRect(rect: Rect): Rect {
  if (rect.width < 2 || rect.height < 2) return { ...rect, width: 0, height: 0 }
  return {
    row: rect.row + 1,
    col: rect.col + 1,
    width: rect.width - 2,
    height: rect.height - 2,
  }
}

// ── constraint solver ──

function solveConstraints(constraints: Constraint[], total: number): number[] {
  const n = constraints.length
  const sizes = new Array(n).fill(0)

  // First pass: compute ideal sizes
  const ideals = constraints.map((c) => {
    switch (c.type) {
      case 'length':
        return c.value
      case 'min':
        return c.value
      case 'max':
        return c.value
      case 'percentage':
        return Math.floor((c.value / 100) * total)
      case 'ratio':
        return Math.floor((c.num / c.den) * total)
    }
  })

  // Assign ideals, clamped to total
  let used = 0
  for (let i = 0; i < n; i++) {
    sizes[i] = Math.min(ideals[i]!, total - used)
    used += sizes[i]!
  }

  // Distribute remaining space to 'min' constraints
  let remaining = total - used
  if (remaining > 0) {
    for (let i = 0; i < n; i++) {
      if (constraints[i]!.type === 'min') {
        sizes[i] = sizes[i]! + remaining
        remaining = 0
        break
      }
    }
    // If no min constraint, give to first percentage/ratio
    if (remaining > 0) {
      for (let i = 0; i < n; i++) {
        const t = constraints[i]!.type
        if (t === 'percentage' || t === 'ratio') {
          sizes[i] = sizes[i]! + remaining
          remaining = 0
          break
        }
      }
    }
    // Last resort: give to last
    if (remaining > 0) {
      sizes[n - 1] = sizes[n - 1]! + remaining
    }
  }

  return sizes
}
