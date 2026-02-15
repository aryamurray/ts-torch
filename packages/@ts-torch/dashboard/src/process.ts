// ─────────────────────────────────────────────────────────────
// DashboardProcess — parent-side proxy that spawns the dashboard
// as a child process and communicates via IPC
//
// Data flows through a temp file using writeSync (truly synchronous,
// works even when the event loop is blocked by a training loop).
// Node IPC is used only for control messages (init/ready/shutdown)
// which are sent when the event loop is free.
// ─────────────────────────────────────────────────────────────

import { fork, type ChildProcess } from 'node:child_process'
import { openSync, writeSync, closeSync, unlinkSync, existsSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import type { Split, StatusEntry } from './data.js'
import type { ToChild, ToParent } from './ipc.js'

export interface DashboardProcessOptions {
  title?: string
  refreshRate?: number
}

// Terminal reset sequence — leave alt screen + show cursor.
// Written as a last resort if the child didn't get to clean up.
const TERMINAL_RESET = '\x1b[?1049l\x1b[?25h'

export class DashboardProcess {
  readonly numericMetrics: { push: (name: string, split: Split, value: number) => void }
  readonly textMetrics: { push: (name: string, split: Split, formatted: string) => void }
  readonly progress: { update: (split: Split, total: number, task: number) => void }
  readonly status: { update: (mode: Split, entries: StatusEntry[]) => void }

  private child: ChildProcess | null = null
  private pending: ToChild[] = []
  private flushTimer: ReturnType<typeof setInterval> | null = null
  private lastFlushTime = 0
  private opts: DashboardProcessOptions
  private destroyed = false

  // Synchronous data channel (temp file)
  private dataPath: string
  private dataFd: number | null = null
  // Synchronous quit detection (flag file)
  private quitPath: string
  private _quitRequested = false

  // Bound cleanup for signal/exit handlers (so we can remove them)
  private _onExit: (() => void) | null = null
  private _onSignal: (() => void) | null = null

  constructor(opts: DashboardProcessOptions = {}) {
    this.opts = opts
    const id = `ts-torch-dash-${process.pid}-${Date.now()}`
    this.dataPath = join(tmpdir(), `${id}.data`)
    this.quitPath = join(tmpdir(), `${id}.quit`)

    this.numericMetrics = {
      push: (name, split, value) => this.enqueue({ t: 'np', n: name, s: split, v: value }),
    }
    this.textMetrics = {
      push: (name, split, formatted) => this.enqueue({ t: 'tp', n: name, s: split, f: formatted }),
    }
    this.progress = {
      update: (split, total, task) => this.enqueue({ t: 'pu', s: split, total, task }),
    }
    this.status = {
      update: (mode, entries) => this.enqueue({ t: 'su', m: mode, e: entries }),
    }
  }

  /** Check synchronously whether the user pressed 'q'. Works even with a blocked event loop. */
  get quitRequested(): boolean {
    if (!this._quitRequested) {
      this._quitRequested = existsSync(this.quitPath)
    }
    return this._quitRequested
  }

  /**
   * Spawn the child process.
   * Can be awaited to wait for the child to be ready, or called fire-and-forget
   * (messages are buffered and flushed once the child is up).
   */
  start(): Promise<void> {
    // Open data file for synchronous writes
    this.dataFd = openSync(this.dataPath, 'w')

    const childModule = join(dirname(fileURLToPath(import.meta.url)), 'ipc.mjs')
    this.child = fork(childModule, [this.dataPath, this.quitPath], {
      stdio: ['ignore', 'inherit', 'inherit', 'ipc'],
    })

    this.child.on('message', (msg: ToParent) => {
      if (msg.t === 'quit') {
        this._quitRequested = true
      }
    })

    this.child.on('exit', () => {
      this.stopFlushing()
      this.child = null
    })

    // Send init over IPC (event loop is free at this point)
    this.child.send({
      t: 'init',
      title: this.opts.title ?? 'ts-torch',
      refreshRate: this.opts.refreshRate ?? 100,
    } satisfies ToChild)

    this.startFlushing()
    this.installCleanupHandlers()

    // Return a promise that resolves when the child signals ready
    return new Promise<void>((resolve) => {
      const onMessage = (msg: ToParent) => {
        if (msg.t === 'ready') {
          this.child?.off('message', onMessage)
          resolve()
        }
      }
      this.child!.on('message', onMessage)
    })
  }

  /** No-op — the child process renders on its own schedule. */
  requestRender() {}

  /** Shut down the child process and restore the terminal. */
  destroy() {
    if (this.destroyed) return
    this.destroyed = true

    this.removeCleanupHandlers()
    this.stopFlushing()
    this.flush()

    // Close data file
    if (this.dataFd !== null) {
      try {
        closeSync(this.dataFd)
      } catch { /* ignore */ }
      this.dataFd = null
    }

    if (this.child) {
      if (this.child.connected) {
        this.child.send({ t: 'shutdown' } satisfies ToChild)
      }
      const child = this.child
      const timer = setTimeout(() => {
        try {
          child.kill('SIGKILL')
        } catch { /* ignore */ }
      }, 1000)
      child.on('exit', () => clearTimeout(timer))
      this.child = null
    }

    // Fallback: reset terminal in case the child didn't get to clean up.
    // Writing these sequences is idempotent — no harm if the child already did it.
    try {
      process.stdout.write(TERMINAL_RESET)
    } catch { /* ignore — stdout may be closed */ }

    // Clean up temp files
    try {
      unlinkSync(this.dataPath)
    } catch { /* ignore */ }
    try {
      unlinkSync(this.quitPath)
    } catch { /* ignore */ }
  }

  private installCleanupHandlers() {
    // 'exit' fires for normal exit, uncaughtException, etc.
    // Cannot do async work here, but destroy() is synchronous enough.
    this._onExit = () => this.destroy()
    process.on('exit', this._onExit)

    // For SIGINT/SIGTERM, destroy then re-raise so the parent exits normally.
    this._onSignal = () => {
      this.destroy()
      // Remove our handlers so the default behavior (exit) takes over on re-raise
      this.removeCleanupHandlers()
    }
    process.on('SIGINT', this._onSignal)
    process.on('SIGTERM', this._onSignal)
  }

  private removeCleanupHandlers() {
    if (this._onExit) {
      process.off('exit', this._onExit)
      this._onExit = null
    }
    if (this._onSignal) {
      process.off('SIGINT', this._onSignal)
      process.off('SIGTERM', this._onSignal)
      this._onSignal = null
    }
  }

  private enqueue(msg: ToChild) {
    this.pending.push(msg)
    // Sync flush — works even when the event loop is blocked
    const now = Date.now()
    if (now - this.lastFlushTime >= 50) {
      this.flush()
    }
  }

  private flush() {
    if (this.pending.length > 0 && this.dataFd !== null) {
      writeSync(this.dataFd, JSON.stringify(this.pending) + '\n')
      this.pending = []
      this.lastFlushTime = Date.now()
    }
  }

  private startFlushing() {
    this.flushTimer = setInterval(() => this.flush(), 50)
  }

  private stopFlushing() {
    if (this.flushTimer) {
      clearInterval(this.flushTimer)
      this.flushTimer = null
    }
  }
}
