// ─────────────────────────────────────────────────────────────
// ANSI primitives — zero dependencies, immediate-mode TUI
// ─────────────────────────────────────────────────────────────

const ESC = '\x1b['

export const ansi = {
  clear: () => `${ESC}2J${ESC}H`,
  moveTo: (row: number, col: number) => `${ESC}${row};${col}H`,
  hideCursor: () => `${ESC}?25l`,
  showCursor: () => `${ESC}?25h`,
  enterAlt: () => `${ESC}?1049h`,
  leaveAlt: () => `${ESC}?1049l`,
  fg: (r: number, g: number, b: number) => `${ESC}38;2;${r};${g};${b}m`,
  bg: (r: number, g: number, b: number) => `${ESC}48;2;${r};${g};${b}m`,
  reset: `${ESC}0m`,
  resetFg: `${ESC}39m`,
  resetBg: `${ESC}49m`,
  bold: `${ESC}1m`,
  dim: `${ESC}2m`,
  italic: `${ESC}3m`,
  underline: `${ESC}4m`,
} as const
