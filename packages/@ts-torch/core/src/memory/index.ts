/**
 * Memory management utilities for efficient tensor operations
 *
 * Provides:
 * - Scoped memory management (torch.run) for automatic cleanup
 * - Tensor pooling for allocation reuse
 * - ArrayBuffer pooling for raw memory
 * - Memory usage tracking
 *
 * @module memory
 */

// ==================== Scoped Memory Management ====================

export {
  run,
  runAsync,
  registerTensor,
  escapeTensor,
  inScope,
  scopeDepth,
  currentScopeId,
  scopeTensorCount,
  type ScopedTensor,
} from './scope.js'

// ==================== Tensor Pooling ====================

export { TensorPool, globalTensorPool, type PoolableTensor, type PoolStats } from './pool.js'

// ==================== ArrayBuffer Memory Pool ====================

/**
 * Memory pool for ArrayBuffer allocations
 * Separate from TensorPool - this pools raw buffers, not tensor objects
 */
export class MemoryPool {
  private pools: Map<number, ArrayBuffer[]> = new Map()

  /**
   * Allocate a buffer from the pool
   */
  allocate(size: number): ArrayBuffer {
    const pool = this.pools.get(size)
    if (pool && pool.length > 0) {
      const buffer = pool.pop()
      if (buffer) return buffer
    }
    return new ArrayBuffer(size)
  }

  /**
   * Return a buffer to the pool
   */
  deallocate(buffer: ArrayBuffer): void {
    const size = buffer.byteLength
    if (!this.pools.has(size)) {
      this.pools.set(size, [])
    }
    this.pools.get(size)!.push(buffer)
  }

  /**
   * Clear all pools
   */
  clear(): void {
    this.pools.clear()
  }

  /**
   * Get memory statistics
   */
  stats(): { totalBuffers: number; totalSize: number } {
    let totalBuffers = 0
    let totalSize = 0

    for (const [size, buffers] of this.pools.entries()) {
      totalBuffers += buffers.length
      totalSize += size * buffers.length
    }

    return { totalBuffers, totalSize }
  }
}

/**
 * Global memory pool instance for ArrayBuffers
 */
export const globalMemoryPool = new MemoryPool()

// ==================== Memory Usage Tracking ====================

/**
 * Memory usage tracker
 */
export class MemoryTracker {
  private allocated: number = 0
  private peak: number = 0

  track(size: number): void {
    this.allocated += size
    if (this.allocated > this.peak) {
      this.peak = this.allocated
    }
  }

  release(size: number): void {
    this.allocated -= size
  }

  reset(): void {
    this.allocated = 0
    this.peak = 0
  }

  getCurrentUsage(): number {
    return this.allocated
  }

  getPeakUsage(): number {
    return this.peak
  }
}

/**
 * Global memory tracker instance
 */
export const globalMemoryTracker = new MemoryTracker()
