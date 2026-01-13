/**
 * Worker pool for managing worker threads
 */

import { Worker } from 'node:worker_threads'
import type {
  BatchMetadata,
  BatchTask,
  MainToWorkerMessage,
  WorkerConfig,
  WorkerToMainMessage,
} from './types.js'

interface PendingTask {
  workerId: number
  resolve: (metadata: BatchMetadata) => void
  reject: (error: Error) => void
}

/**
 * Pool of worker threads for parallel data loading
 */
export class WorkerPool {
  private workers: Worker[] = []
  private available: number[] = []
  private pending: Map<number, PendingTask> = new Map()
  private taskIdCounter = 0
  private isShutdown = false
  private readyPromises: Map<number, { resolve: () => void; reject: (e: Error) => void }> =
    new Map()

  constructor(
    private numWorkers: number,
    private config: WorkerConfig,
    private workerPath: string | URL,
  ) {
    this.spawnWorkers()
  }

  /**
   * Spawn all worker threads
   */
  private spawnWorkers(): void {
    for (let i = 0; i < this.numWorkers; i++) {
      this.spawnWorker(i)
    }
  }

  /**
   * Spawn a single worker thread
   */
  private spawnWorker(id: number): void {
    const worker = new Worker(this.workerPath, {
      workerData: this.config,
    })

    // Create a promise that resolves when the worker is ready
    this.readyPromises.set(id, {
      resolve: () => {},
      reject: () => {},
    })

    // Create actual promise and update the map
    const readyPromise = new Promise<void>((resolve, reject) => {
      this.readyPromises.set(id, { resolve, reject })
    })

    // Store promise for later use (not directly awaited here)
    void readyPromise

    worker.on('message', (message: WorkerToMainMessage) => {
      this.handleMessage(id, message)
    })

    worker.on('error', (error: Error) => {
      this.handleError(id, error)
    })

    worker.on('exit', (code: number) => {
      if (code !== 0 && !this.isShutdown) {
        console.error(`Worker ${id} exited with code ${code}`)
        this.handleWorkerExit(id)
      }
    })

    this.workers[id] = worker
  }

  /**
   * Wait for all workers to be ready
   */
  async waitForReady(): Promise<void> {
    // Create promises that resolve when each worker sends 'ready'
    const promises: Promise<void>[] = []

    for (let i = 0; i < this.numWorkers; i++) {
      promises.push(
        new Promise<void>((resolve, reject) => {
          const existing = this.readyPromises.get(i)
          if (existing) {
            // Chain our resolve/reject to the existing ones
            const originalResolve = existing.resolve
            const originalReject = existing.reject
            this.readyPromises.set(i, {
              resolve: () => {
                originalResolve()
                resolve()
              },
              reject: (e) => {
                originalReject(e)
                reject(e)
              },
            })
          } else {
            this.readyPromises.set(i, { resolve, reject })
          }
        }),
      )
    }

    // Wait with timeout
    await Promise.race([
      Promise.all(promises),
      new Promise<void>((_, reject) =>
        setTimeout(() => reject(new Error('Workers failed to initialize in time')), 30000),
      ),
    ])
  }

  /**
   * Handle a message from a worker
   */
  private handleMessage(workerId: number, message: WorkerToMainMessage): void {
    switch (message.type) {
      case 'ready': {
        const entry = this.readyPromises.get(workerId)
        if (entry) {
          entry.resolve()
          this.readyPromises.delete(workerId)
        }
        this.available.push(workerId)
        break
      }

      case 'batchReady': {
        const pending = this.pending.get(message.taskId)
        if (pending) {
          this.pending.delete(message.taskId)
          this.available.push(pending.workerId)
          pending.resolve(message.metadata)
        }
        break
      }

      case 'error': {
        const errorPending = this.pending.get(message.taskId)
        if (errorPending) {
          this.pending.delete(message.taskId)
          this.available.push(errorPending.workerId)
          errorPending.reject(new Error(message.error))
        }
        break
      }
    }
  }

  /**
   * Handle a worker error
   */
  private handleError(workerId: number, error: Error): void {
    console.error(`Worker ${workerId} error:`, error)

    // Reject all pending tasks for this worker
    for (const [taskId, pending] of this.pending.entries()) {
      if (pending.workerId === workerId) {
        this.pending.delete(taskId)
        pending.reject(new Error(`Worker ${workerId} crashed: ${error.message}`))
      }
    }

    // Reject ready promise if still pending
    const readyEntry = this.readyPromises.get(workerId)
    if (readyEntry) {
      readyEntry.reject(error)
      this.readyPromises.delete(workerId)
    }
  }

  /**
   * Handle a worker exiting unexpectedly
   */
  private handleWorkerExit(workerId: number): void {
    // Remove from available list
    const availableIdx = this.available.indexOf(workerId)
    if (availableIdx !== -1) {
      this.available.splice(availableIdx, 1)
    }

    // Reject any pending tasks
    for (const [taskId, pending] of this.pending.entries()) {
      if (pending.workerId === workerId) {
        this.pending.delete(taskId)
        pending.reject(new Error(`Worker ${workerId} exited unexpectedly`))
      }
    }

    // Respawn the worker if not shutting down
    if (!this.isShutdown) {
      console.log(`Respawning worker ${workerId}`)
      this.spawnWorker(workerId)
    }
  }

  /**
   * Submit a task to an available worker
   */
  async submitTask(
    task: Omit<BatchTask, 'taskId'>,
    buffer: SharedArrayBuffer,
  ): Promise<BatchMetadata> {
    if (this.isShutdown) {
      throw new Error('Worker pool is shut down')
    }

    // Wait for an available worker
    while (this.available.length === 0) {
      await new Promise((resolve) => setTimeout(resolve, 1))

      if (this.isShutdown) {
        throw new Error('Worker pool is shut down')
      }
    }

    const workerId = this.available.pop()!
    const taskId = ++this.taskIdCounter

    return new Promise<BatchMetadata>((resolve, reject) => {
      this.pending.set(taskId, { workerId, resolve, reject })

      const message: MainToWorkerMessage = {
        type: 'loadBatch',
        task: { ...task, taskId },
        buffer,
      }

      this.workers[workerId]!.postMessage(message)
    })
  }

  /**
   * Get number of available workers
   */
  get availableCount(): number {
    return this.available.length
  }

  /**
   * Get number of pending tasks
   */
  get pendingCount(): number {
    return this.pending.size
  }

  /**
   * Wait for all pending tasks to complete
   */
  async drain(): Promise<void> {
    while (this.pending.size > 0) {
      await new Promise((resolve) => setTimeout(resolve, 10))
    }
  }

  /**
   * Shutdown all workers
   */
  async shutdown(): Promise<void> {
    if (this.isShutdown) return
    this.isShutdown = true

    // Wait for pending tasks
    await this.drain()

    // Terminate all workers
    const terminatePromises = this.workers.map(async (worker) => {
      const message: MainToWorkerMessage = { type: 'shutdown' }
      worker.postMessage(message)

      // Give worker a chance to exit gracefully
      await new Promise<void>((resolve) => {
        const timeout = setTimeout(() => {
          worker.terminate()
          resolve()
        }, 1000)

        worker.once('exit', () => {
          clearTimeout(timeout)
          resolve()
        })
      })
    })

    await Promise.all(terminatePromises)
    this.workers = []
    this.available = []
  }
}
