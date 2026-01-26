/**
 * Conservative Caching Allocator (Phase 5)
 *
 * Pools tensor storage to reduce allocation overhead in training loops.
 * This is an extremely conservative first implementation that only pools:
 * - Contiguous CPU tensors
 * - Exact dtype matches
 * - Best-fit size matching (within 50% waste threshold)
 *
 * The allocator maintains metrics for monitoring hit rates and can be
 * tuned based on workload characteristics.
 */


/**
 * Allocator statistics for monitoring performance
 */
export interface AllocatorStats {
  /** Number of successful pool hits */
  hits: number
  /** Number of pool misses (new allocation required) */
  misses: number
  /** Total bytes reused from pool */
  bytesReused: number
  /** Number of evictions from pool */
  evictions: number
  /** Current hit rate (0-1) */
  hitRate: number
  /** Number of storages currently in pool */
  poolSize: number
  /** Total bytes currently cached */
  cachedBytes: number
}

/**
 * Pooled storage entry
 * @internal
 */
interface PooledStorage {
  /** Size in bytes */
  sizeBytes: number
  /** Data type */
  dtype: string
  /** Last used timestamp (for LRU eviction) */
  lastUsed: number
  /** Whether currently in use */
  inUse: boolean
  /** Native pointer (if applicable) */
  ptr?: number
}

/**
 * Configuration options for the caching allocator
 */
export interface CachingAllocatorOptions {
  /** Maximum total bytes to cache (default: 256MB) */
  maxCachedBytes?: number
  /** Maximum waste ratio when reusing storage (default: 0.5 = 50%) */
  maxWasteRatio?: number
  /** Enable detailed logging (default: false) */
  verbose?: boolean
}

/**
 * Conservative Caching Allocator
 *
 * Pools tensor storage for reuse to reduce allocation pressure during training.
 * Only pools CPU contiguous tensors with exact dtype match.
 *
 * @example
 * ```ts
 * import { CachingAllocator } from '@ts-torch/core/memory';
 *
 * const allocator = new CachingAllocator({ maxCachedBytes: 128 * 1024 * 1024 });
 *
 * // In training loop
 * for (const batch of dataLoader) {
 *   const storage = allocator.acquire(batchSize * 784 * 4, 'float32');
 *   // ... use storage ...
 *   allocator.release(storage);
 * }
 *
 * console.log(allocator.getStats());
 * // { hits: 950, misses: 50, hitRate: 0.95, ... }
 * ```
 */
export class CachingAllocator {
  private pool: Map<string, PooledStorage[]> = new Map()
  private stats = {
    hits: 0,
    misses: 0,
    bytesReused: 0,
    evictions: 0,
  }
  private totalCachedBytes = 0
  private readonly maxCachedBytes: number
  private readonly maxWasteRatio: number
  private readonly verbose: boolean

  constructor(options: CachingAllocatorOptions = {}) {
    this.maxCachedBytes = options.maxCachedBytes ?? 256 * 1024 * 1024 // 256MB default
    this.maxWasteRatio = options.maxWasteRatio ?? 0.5 // 50% max waste
    this.verbose = options.verbose ?? false
  }

  /**
   * Generate pool key for storage lookup
   */
  private getPoolKey(dtype: string): string {
    // Only pool by dtype - size matching is done in acquire
    return dtype
  }

  /**
   * Attempt to acquire storage from pool
   *
   * @param sizeBytes - Required size in bytes
   * @param dtype - Data type (must match exactly)
   * @returns Pooled storage if found, null if new allocation needed
   */
  acquire(sizeBytes: number, dtype: string): PooledStorage | null {
    const key = this.getPoolKey(dtype)
    const entries = this.pool.get(key)

    if (!entries || entries.length === 0) {
      this.stats.misses++
      if (this.verbose) {
        console.log(`[CachingAllocator] miss: no pool for dtype=${dtype}`)
      }
      return null
    }

    // Find best fit: smallest storage that fits with acceptable waste
    let bestIndex = -1
    let bestSize = Infinity

    for (let i = 0; i < entries.length; i++) {
      const entry = entries[i]!
      if (entry.inUse) continue

      // Must be large enough
      if (entry.sizeBytes < sizeBytes) continue

      // Check waste ratio
      const waste = (entry.sizeBytes - sizeBytes) / sizeBytes
      if (waste > this.maxWasteRatio) continue

      // Prefer smaller storage (less waste)
      if (entry.sizeBytes < bestSize) {
        bestSize = entry.sizeBytes
        bestIndex = i
      }
    }

    if (bestIndex === -1) {
      this.stats.misses++
      if (this.verbose) {
        console.log(`[CachingAllocator] miss: no suitable storage for ${sizeBytes} bytes`)
      }
      return null
    }

    // Found a match
    const storage = entries[bestIndex]!
    storage.inUse = true
    storage.lastUsed = Date.now()

    this.stats.hits++
    this.stats.bytesReused += sizeBytes

    if (this.verbose) {
      console.log(
        `[CachingAllocator] hit: reusing ${storage.sizeBytes} bytes for ${sizeBytes} bytes request`,
      )
    }

    return storage
  }

  /**
   * Release storage back to pool
   *
   * @param storage - Storage to release
   */
  release(storage: PooledStorage): void {
    storage.inUse = false
    storage.lastUsed = Date.now()

    // Check if we need to evict to stay under memory limit
    this.evictIfNeeded()
  }

  /**
   * Add new storage to the pool
   *
   * @param sizeBytes - Size in bytes
   * @param dtype - Data type
   * @param ptr - Optional native pointer
   * @returns New pooled storage entry
   */
  add(sizeBytes: number, dtype: string, ptr?: number): PooledStorage {
    const key = this.getPoolKey(dtype)

    if (!this.pool.has(key)) {
      this.pool.set(key, [])
    }

    const storage: PooledStorage = {
      sizeBytes,
      dtype,
      lastUsed: Date.now(),
      inUse: true,
      ...(ptr !== undefined && { ptr }),
    }

    this.pool.get(key)!.push(storage)
    this.totalCachedBytes += sizeBytes

    if (this.verbose) {
      console.log(`[CachingAllocator] added ${sizeBytes} bytes storage for dtype=${dtype}`)
    }

    return storage
  }

  /**
   * Evict old entries if over memory limit
   */
  private evictIfNeeded(): void {
    while (this.totalCachedBytes > this.maxCachedBytes) {
      // Find oldest unused entry
      let oldestTime = Infinity
      let oldestKey: string | null = null
      let oldestIndex = -1

      for (const [key, entries] of this.pool) {
        for (let i = 0; i < entries.length; i++) {
          const entry = entries[i]!
          if (!entry.inUse && entry.lastUsed < oldestTime) {
            oldestTime = entry.lastUsed
            oldestKey = key
            oldestIndex = i
          }
        }
      }

      if (oldestKey === null || oldestIndex === -1) {
        // All entries in use, can't evict
        break
      }

      // Evict the oldest entry
      const entries = this.pool.get(oldestKey)!
      const evicted = entries.splice(oldestIndex, 1)[0]!
      this.totalCachedBytes -= evicted.sizeBytes
      this.stats.evictions++

      if (this.verbose) {
        console.log(`[CachingAllocator] evicted ${evicted.sizeBytes} bytes storage`)
      }

      // Remove empty pool entries
      if (entries.length === 0) {
        this.pool.delete(oldestKey)
      }
    }
  }

  /**
   * Clear all cached storage
   */
  clear(): void {
    this.pool.clear()
    this.totalCachedBytes = 0

    if (this.verbose) {
      console.log('[CachingAllocator] cleared all cached storage')
    }
  }

  /**
   * Get current allocator statistics
   */
  getStats(): AllocatorStats {
    const total = this.stats.hits + this.stats.misses
    let poolSize = 0

    for (const entries of this.pool.values()) {
      poolSize += entries.filter((e) => !e.inUse).length
    }

    return {
      ...this.stats,
      hitRate: total > 0 ? this.stats.hits / total : 0,
      poolSize,
      cachedBytes: this.totalCachedBytes,
    }
  }

  /**
   * Reset statistics (does not clear pool)
   */
  resetStats(): void {
    this.stats = {
      hits: 0,
      misses: 0,
      bytesReused: 0,
      evictions: 0,
    }
  }
}

/**
 * Global caching allocator instance
 *
 * Use this for general tensor allocation caching in training loops.
 * Configure via environment or direct access if needed.
 */
export const globalCachingAllocator = new CachingAllocator()
