import { describe, it, expect } from 'vitest'
import { splitRect, innerRect, Constraint, type Rect } from '../layout.js'

describe('splitRect', () => {
  const base: Rect = { row: 1, col: 1, width: 80, height: 40 }

  it('splits vertically with fixed lengths', () => {
    const [top, bottom] = splitRect(base, 'vertical', [Constraint.length(10), Constraint.length(30)])
    expect(top.height).toBe(10)
    expect(top.row).toBe(1)
    expect(bottom.height).toBe(30)
    expect(bottom.row).toBe(11)
    expect(top.width).toBe(80)
    expect(bottom.width).toBe(80)
  })

  it('splits horizontally with fixed lengths', () => {
    const [left, right] = splitRect(base, 'horizontal', [Constraint.length(30), Constraint.length(50)])
    expect(left.width).toBe(30)
    expect(left.col).toBe(1)
    expect(right.width).toBe(50)
    expect(right.col).toBe(31)
    expect(left.height).toBe(40)
  })

  it('distributes remaining space to min constraint', () => {
    const [fixed, flex] = splitRect(base, 'vertical', [Constraint.length(10), Constraint.min(5)])
    expect(fixed.height).toBe(10)
    expect(flex.height).toBe(30) // 40 - 10
  })

  it('handles percentage constraints', () => {
    const [left, right] = splitRect(base, 'horizontal', [Constraint.percentage(25), Constraint.percentage(75)])
    expect(left.width).toBe(20) // 25% of 80
    expect(right.width).toBe(60) // 75% of 80
  })

  it('handles ratio constraints', () => {
    const [a, b] = splitRect(base, 'vertical', [Constraint.ratio(1, 4), Constraint.ratio(3, 4)])
    expect(a.height).toBe(10) // 1/4 of 40
    expect(b.height).toBe(30) // 3/4 of 40
  })

  it('handles max constraints', () => {
    const [header, body] = splitRect(base, 'vertical', [Constraint.max(5), Constraint.min(10)])
    expect(header.height).toBe(5)
    expect(body.height).toBe(35)
  })

  it('handles three-way split', () => {
    const [a, b, c] = splitRect(base, 'vertical', [Constraint.max(5), Constraint.min(6), Constraint.max(6)])
    expect(a.height).toBe(5)
    expect(c.height).toBe(6)
    expect(b.height).toBe(29) // gets remaining
    expect(a.row + a.height).toBe(b.row)
    expect(b.row + b.height).toBe(c.row)
  })

  it('preserves col for vertical splits', () => {
    const rects = splitRect({ row: 5, col: 10, width: 60, height: 20 }, 'vertical', [
      Constraint.length(10),
      Constraint.length(10),
    ])
    expect(rects[0].col).toBe(10)
    expect(rects[1].col).toBe(10)
  })

  it('preserves row for horizontal splits', () => {
    const rects = splitRect({ row: 5, col: 10, width: 60, height: 20 }, 'horizontal', [
      Constraint.length(30),
      Constraint.length(30),
    ])
    expect(rects[0].row).toBe(5)
    expect(rects[1].row).toBe(5)
  })
})

describe('innerRect', () => {
  it('shrinks rect by 1 on each side', () => {
    const inner = innerRect({ row: 1, col: 1, width: 10, height: 10 })
    expect(inner).toEqual({ row: 2, col: 2, width: 8, height: 8 })
  })

  it('returns zero-size for too-small rects', () => {
    const inner = innerRect({ row: 1, col: 1, width: 1, height: 1 })
    expect(inner.width).toBe(0)
    expect(inner.height).toBe(0)
  })
})
