/**
 * Tensor pool for object reuse optimization
 *
 * Reduces allocation overhead by recycling tensor objects. Particularly
 * useful for training loops and hot paths where tensors of the same
 * shape are repeatedly created and destroyed.
 *
 * @example
 * ```ts
 * const pool = new TensorPool();
 *
 * // Training loop
 * for (let i = 0; i < 1000; i++) {
 *   run(() => {
 *     // Try to reuse tensor from pool
 *     const grad = pool.acquire([256, 256], "float32") ?? zeros([256, 256]);
 *     // ... use grad ...
 *     pool.release(grad); // Return to pool for reuse
 *   });
 * }
 * ```
 */

import type { Shape } from "../types/shape.js";
import type { DTypeName } from "../types/dtype.js";

/**
 * Minimal tensor interface for pooling
 * Avoids circular dependencies with actual Tensor class
 */
export interface PoolableTensor {
  readonly shape: Shape;
  readonly dtype: DTypeName;
  readonly handle: unknown;
}

/**
 * Statistics about pool performance
 */
export interface PoolStats {
  readonly size: number;
  readonly hitCount: number;
  readonly missCount: number;
  readonly hitRate: number;
  readonly pools: ReadonlyMap<string, number>;
}

/**
 * Tensor pool for reusing tensor allocations
 *
 * @remarks
 * The pool uses shape signature and dtype as keys. Tensors are stored
 * per unique (shape, dtype) combination. The pool has a maximum size
 * per key to prevent unbounded memory growth.
 */
export class TensorPool {
  private pools: Map<string, PoolableTensor[]> = new Map();
  private hitCount = 0;
  private missCount = 0;
  private maxPoolSize: number;

  /**
   * Create a new tensor pool
   *
   * @param maxPoolSize - Maximum tensors to cache per (shape, dtype) key (default: 16)
   */
  constructor(maxPoolSize = 16) {
    this.maxPoolSize = maxPoolSize;
  }

  /**
   * Acquire a tensor from the pool if available.
   * Returns null if no matching tensor is available.
   *
   * @template S - Shape type
   * @template D - DType name
   * @param shape - Desired tensor shape
   * @param dtype - Desired tensor dtype
   * @returns Pooled tensor or null
   *
   * @example
   * ```ts
   * const tensor = pool.acquire([10, 10], "float32");
   * if (tensor === null) {
   *   // Create new tensor
   *   tensor = zeros([10, 10], "float32");
   * }
   * ```
   */
  acquire<S extends Shape, D extends DTypeName>(shape: S, dtype: D): PoolableTensor | null {
    const key = this.makeKey(shape, dtype);
    const pool = this.pools.get(key);

    if (pool && pool.length > 0) {
      this.hitCount++;
      return pool.pop()!;
    }

    this.missCount++;
    return null;
  }

  /**
   * Release a tensor back to the pool for reuse.
   * If the pool for this tensor type is full, the tensor is discarded.
   *
   * @param tensor - Tensor to return to pool
   *
   * @example
   * ```ts
   * run(() => {
   *   const temp = pool.acquire([100, 100], "float32") ?? zeros([100, 100]);
   *   // ... use temp ...
   *   pool.release(temp);
   * });
   * ```
   */
  release(tensor: PoolableTensor): void {
    const key = this.makeKey(tensor.shape, tensor.dtype);

    if (!this.pools.has(key)) {
      this.pools.set(key, []);
    }

    const pool = this.pools.get(key)!;

    // Only cache up to maxPoolSize tensors per key
    if (pool.length < this.maxPoolSize) {
      pool.push(tensor);
    }
  }

  /**
   * Clear all cached tensors from the pool.
   * Does not free the tensors - they should be freed by their scopes.
   *
   * @example
   * ```ts
   * pool.clear(); // Reset pool
   * ```
   */
  clear(): void {
    this.pools.clear();
    this.hitCount = 0;
    this.missCount = 0;
  }

  /**
   * Get pool statistics including hit rate and sizes.
   *
   * @returns Pool statistics object
   *
   * @example
   * ```ts
   * const stats = pool.stats();
   * console.log(`Hit rate: ${(stats.hitRate * 100).toFixed(1)}%`);
   * console.log(`Total cached: ${stats.size}`);
   * ```
   */
  stats(): PoolStats {
    let totalSize = 0;
    const poolSizes = new Map<string, number>();

    for (const [key, pool] of this.pools.entries()) {
      totalSize += pool.length;
      poolSizes.set(key, pool.length);
    }

    const totalRequests = this.hitCount + this.missCount;
    const hitRate = totalRequests > 0 ? this.hitCount / totalRequests : 0;

    return {
      size: totalSize,
      hitCount: this.hitCount,
      missCount: this.missCount,
      hitRate,
      pools: poolSizes,
    };
  }

  /**
   * Remove all tensors for a specific (shape, dtype) key.
   *
   * @param shape - Tensor shape
   * @param dtype - Tensor dtype
   *
   * @example
   * ```ts
   * pool.clearKey([10, 10], "float32");
   * ```
   */
  clearKey(shape: Shape, dtype: DTypeName): void {
    const key = this.makeKey(shape, dtype);
    this.pools.delete(key);
  }

  /**
   * Get the number of cached tensors for a specific (shape, dtype).
   *
   * @param shape - Tensor shape
   * @param dtype - Tensor dtype
   * @returns Number of cached tensors
   *
   * @example
   * ```ts
   * const count = pool.getKeySize([10, 10], "float32");
   * console.log(`Cached tensors: ${count}`);
   * ```
   */
  getKeySize(shape: Shape, dtype: DTypeName): number {
    const key = this.makeKey(shape, dtype);
    return this.pools.get(key)?.length ?? 0;
  }

  /**
   * Create a string key from shape and dtype
   * @internal
   */
  private makeKey(shape: Shape, dtype: DTypeName): string {
    // Use JSON.stringify for shape to handle arbitrary dimensions
    // and ensure consistent ordering
    const shapeKey = `[${shape.join(",")}]`;
    return `${dtype}:${shapeKey}`;
  }

  /**
   * Get all unique (shape, dtype) keys in the pool.
   *
   * @returns Array of key strings
   *
   * @example
   * ```ts
   * const keys = pool.getKeys();
   * console.log(`Unique tensor types cached: ${keys.length}`);
   * ```
   */
  getKeys(): string[] {
    return Array.from(this.pools.keys());
  }

  /**
   * Prune the pool to reduce memory usage.
   * Removes least recently used tensors until size target is met.
   *
   * @param targetSize - Target number of tensors to keep (default: half current size)
   *
   * @example
   * ```ts
   * if (pool.stats().size > 1000) {
   *   pool.prune(500); // Reduce to 500 tensors
   * }
   * ```
   */
  prune(targetSize?: number): void {
    const stats = this.stats();
    const target = targetSize ?? Math.floor(stats.size / 2);

    if (stats.size <= target) {
      return; // Already under target
    }

    let currentSize = stats.size;

    // Remove from pools until we hit target
    for (const [key, pool] of this.pools.entries()) {
      while (pool.length > 0 && currentSize > target) {
        pool.pop();
        currentSize--;
      }

      // Remove empty pools
      if (pool.length === 0) {
        this.pools.delete(key);
      }

      if (currentSize <= target) {
        break;
      }
    }
  }
}

/**
 * Global default tensor pool instance
 * Can be used for convenience without creating custom pools
 *
 * @example
 * ```ts
 * import { globalTensorPool } from '@ts-torch/core/memory';
 *
 * const tensor = globalTensorPool.acquire([10, 10], "float32") ?? zeros([10, 10]);
 * ```
 */
export const globalTensorPool = new TensorPool();
