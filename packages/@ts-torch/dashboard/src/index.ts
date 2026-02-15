// ─────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────

export { Dashboard, type DashboardOptions } from './dashboard.js'
export { DashboardProcess, type DashboardProcessOptions } from './process.js'
export { NumericMetricsState, TextMetricsState, ProgressState, StatusState } from './data.js'
export type { Split, StatusEntry, PlotKind } from './data.js'

// Low-level primitives (for custom dashboards)
export { Buffer } from './buffer.js'
export { type Rect, type Constraint, type Direction, splitRect, innerRect } from './layout.js'
export {
  drawBlock,
  drawParagraph,
  drawGauge,
  drawTabs,
  drawChart,
  drawBarChart,
  drawSparkline,
  Color,
} from './widgets.js'
export { ansi } from './ansi.js'
