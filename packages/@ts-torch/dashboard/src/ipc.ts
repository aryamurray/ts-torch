// ─────────────────────────────────────────────────────────────
// IPC protocol types + child process entry point
//
// Control messages (init/ready/shutdown/quit) use Node IPC.
// Data messages are read from a temp file that the parent writes
// to synchronously with writeSync, so data flows even when the
// parent's event loop is blocked by a training loop.
// ─────────────────────────────────────────────────────────────

import { openSync, readSync, writeFileSync } from 'node:fs'
import { Dashboard } from './dashboard.js'
import type { Split, StatusEntry } from './data.js'

// ── Message types ──

export type ToChild =
  | { t: 'init'; title: string; refreshRate: number }
  | { t: 'np'; n: string; s: Split; v: number }
  | { t: 'tp'; n: string; s: Split; f: string }
  | { t: 'pu'; s: Split; total: number; task: number }
  | { t: 'su'; m: Split; e: StatusEntry[] }
  | { t: 'shutdown' }

export type ToParent = { t: 'ready' } | { t: 'quit' }

// ── Child process entry point ──

const dataPath = process.argv[2]!
const quitPath = process.argv[3]!

let dash: Dashboard | null = null
let dataFd: number | null = null
let readPos = 0
let partial = ''
const readBuf = globalThis.Buffer.alloc(131072) // 128KB
let pollTimer: ReturnType<typeof setInterval> | null = null

function handleDataMsg(msg: ToChild) {
  switch (msg.t) {
    case 'np':
      dash?.numericMetrics.push(msg.n, msg.s, msg.v)
      break
    case 'tp':
      dash?.textMetrics.push(msg.n, msg.s, msg.f)
      break
    case 'pu':
      dash?.progress.update(msg.s, msg.total, msg.task)
      break
    case 'su':
      dash?.status.update(msg.m, msg.e)
      break
  }
}

function pollData() {
  if (dataFd === null) return
  const n = readSync(dataFd, readBuf, 0, readBuf.length, readPos)
  if (n > 0) {
    readPos += n
    partial += readBuf.toString('utf8', 0, n)
    const lines = partial.split('\n')
    partial = lines.pop()! // last element may be incomplete
    for (const line of lines) {
      if (line) {
        const batch = JSON.parse(line) as ToChild[]
        for (const msg of batch) handleDataMsg(msg)
      }
    }
  }
}

// Register IPC listener synchronously (no async gap — no messages lost)
process.on('message', (msg: ToChild) => {
  switch (msg.t) {
    case 'init':
      dash = new Dashboard({
        title: msg.title,
        refreshRate: msg.refreshRate,
        onQuit: () => {
          // Write flag file so parent can detect quit synchronously
          try {
            writeFileSync(quitPath, '')
          } catch { /* ignore */ }
          process.send!({ t: 'quit' } satisfies ToParent)
        },
      })
      dash.start()
      // Open data file for reading
      dataFd = openSync(dataPath, 'r')
      // Poll data file at 50ms intervals
      pollTimer = setInterval(pollData, 50)
      process.send!({ t: 'ready' } satisfies ToParent)
      break
    case 'shutdown':
      pollData() // drain remaining data
      if (pollTimer) clearInterval(pollTimer)
      dash?.destroy()
      process.exit(0)
      break
  }
})

process.on('disconnect', () => {
  pollData()
  if (pollTimer) clearInterval(pollTimer)
  dash?.destroy()
  process.exit(0)
})
