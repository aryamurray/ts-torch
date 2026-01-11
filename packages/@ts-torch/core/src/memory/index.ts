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
 * Configuration options for MemoryPool
 */
export interface MemoryPoolOptions {
  /** Maximum number of buffers to cache per size (default: 32) */
  maxBuffersPerSize?: number
  /** Maximum total memory to cache in bytes (default: 256MB) */
  maxTotalBytes?: number
  /** Maximum buffer size to cache (default: 64MB) - larger buffers are not pooled */
  maxBufferSize?: number
}

/**
 * Memory pool for ArrayBuffer allocations
 * Separate from TensorPool - this pools raw buffers, not tensor objects
 */
export class MemoryPool {
  private pools: Map<number, ArrayBuffer[]> = new Map()
  private totalCachedBytes = 0
  private readonly maxBuffersPerSize: number
  private readonly maxTotalBytes: number
  private readonly maxBufferSize: number

  /**
   * Create a new memory pool
   *
   * @param options - Pool configuration options
   */
  constructor(options: MemoryPoolOptions = {}) {
    this.maxBuffersPerSize = options.maxBuffersPerSize ?? 32
    this.maxTotalBytes = options.maxTotalBytes ?? 256 * 1024 * 1024 // 256MB default
    this.maxBufferSize = options.maxBufferSize ?? 64 * 1024 * 1024 // 64MB default
  }

  /**
   * Allocate a buffer from the pool
   */
  allocate(size: number): ArrayBuffer {
    const pool = this.pools.get(size)
    if (pool && pool.length > 0) {
      const buffer = pool.pop()
      if (buffer) {
        this.totalCachedBytes -= size
        return buffer
      }
    }
    return new ArrayBuffer(size)
  }

  /**
   * Return a buffer to the pool
   * If the pool is full or buffer is too large, the buffer is discarded (GC will reclaim it)
   */
  deallocate(buffer: ArrayBuffer): void {
    const size = buffer.byteLength

    // Don't pool buffers that are too large
    if (size > this.maxBufferSize) {
      return // Let GC handle it
    }

    // Don't exceed total memory limit
    if (this.totalCachedBytes + size > this.maxTotalBytes) {
      return // Let GC handle it
    }

    if (!this.pools.has(size)) {
      this.pools.set(size, [])
    }

    const pool = this.pools.get(size)!

    // Don't exceed per-size limit
    if (pool.length >= this.maxBuffersPerSize) {
      return // Let GC handle it
    }

    pool.push(buffer)
    this.totalCachedBytes += size
  }

  /**
   * Clear all pools
   */
  clear(): void {
    this.pools.clear()
    this.totalCachedBytes = 0
  }

  /**
   * Prune the pool to reduce memory usage
   *
   * @param targetBytes - Target total cached bytes (default: half current size)
   */
  prune(targetBytes?: number): void {
    const target = targetBytes ?? Math.floor(this.totalCachedBytes / 2)

    if (this.totalCachedBytes <= target) {
      return
    }

    // Remove buffers from pools until we hit target
    for (const [size, pool] of this.pools.entries()) {
      while (pool.length > 0 && this.totalCachedBytes > target) {
        pool.pop()
        this.totalCachedBytes -= size
      }

      // Remove empty pools
      if (pool.length === 0) {
        this.pools.delete(size)
      }

      if (this.totalCachedBytes <= target) {
        break
      }
    }
  }

  /**
   * Get memory statistics
   */
  stats(): { totalBuffers: number; totalSize: number; maxTotalBytes: number; maxBufferSize: number } {
    let totalBuffers = 0
    let totalSize = 0

    for (const [size, buffers] of this.pools.entries()) {
      totalBuffers += buffers.length
      totalSize += size * buffers.length
    }

    return {
      totalBuffers,
      totalSize,
      maxTotalBytes: this.maxTotalBytes,
      maxBufferSize: this.maxBufferSize,
    }
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
