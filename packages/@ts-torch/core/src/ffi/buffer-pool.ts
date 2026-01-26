/**
 * Buffer pooling for FFI calls
 *
 * Reduces allocation pressure by reusing error buffers and shape arrays
 * in hot paths. This optimization is critical for reducing FFI overhead
 * during high-frequency tensor operations.
 */

// Inline constant to avoid circular dependency with error.ts
// Must match ERROR_STRUCT_SIZE in error.ts (4 bytes code + 256 bytes message = 260)
const ERROR_STRUCT_SIZE = 260

const ERROR_POOL_SIZE = 8
const SHAPE_POOL_SIZE = 16
const SHAPE_CACHE_32_SIZE = 64 // Larger cache for int32 shapes (most common case)

/**
 * Maximum safe int32 value for shape dimensions
 * Dimensions larger than this require int64 path
 */
export const MAX_INT32 = 2_147_483_647

/**
 * Pool for reusing error buffers across FFI calls
 *
 * Instead of allocating a new ArrayBuffer(260) for each FFI call,
 * this pool maintains a small set of reusable buffers.
 */
class ErrorBufferPool {
  private pool: ArrayBuffer[] = []

  constructor() {
    // Pre-allocate pool for immediate availability
    for (let i = 0; i < ERROR_POOL_SIZE; i++) {
      this.pool.push(new ArrayBuffer(ERROR_STRUCT_SIZE))
    }
  }

  /**
   * Acquire an error buffer from the pool
   * Creates a new buffer if pool is empty or buffer was detached by FFI
   */
  acquire(): ArrayBuffer {
    let buffer = this.pool.pop()
    // Check if buffer was detached by FFI (byteLength becomes 0 for detached buffers)
    // or is somehow invalid - if so, create a new one
    if (!buffer || buffer.byteLength !== ERROR_STRUCT_SIZE) {
      buffer = new ArrayBuffer(ERROR_STRUCT_SIZE)
    }
    // Reset error code to 0 (OK)
    new DataView(buffer).setInt32(0, 0, true)
    return buffer
  }

  /**
   * Release an error buffer back to the pool
   * Buffer is discarded if detached by FFI or pool is at capacity
   */
  release(buffer: ArrayBuffer): void {
    // Don't return detached or wrong-sized buffers to the pool
    if (buffer.byteLength !== ERROR_STRUCT_SIZE) {
      return
    }
    if (this.pool.length < ERROR_POOL_SIZE) {
      this.pool.push(buffer)
    }
  }
}

/**
 * Cache for reusing BigInt64Array shape buffers
 *
 * Groups buffers by dimension count (ndim) since tensor shapes
 * commonly cluster around specific dimensions (1D, 2D, 3D, 4D).
 */
class ShapeBufferCache {
  private caches = new Map<number, BigInt64Array[]>()

  /**
   * Acquire a shape buffer for the given number of dimensions
   * Creates a new buffer if cache is empty for this ndim or buffer was detached
   */
  acquire(ndim: number): BigInt64Array {
    const cache = this.caches.get(ndim)
    let buffer = cache?.pop()
    // Check if buffer was detached by FFI or is invalid
    if (!buffer || buffer.length !== ndim || buffer.buffer.byteLength === 0) {
      buffer = new BigInt64Array(ndim)
    }
    return buffer
  }

  /**
   * Release a shape buffer back to the cache
   * Buffer is discarded if detached by FFI or cache is at capacity
   */
  release(buffer: BigInt64Array): void {
    // Don't return detached buffers to the cache
    if (buffer.buffer.byteLength === 0) {
      return
    }
    const ndim = buffer.length
    let cache = this.caches.get(ndim)
    if (!cache) {
      cache = []
      this.caches.set(ndim, cache)
    }
    if (cache.length < SHAPE_POOL_SIZE) {
      cache.push(buffer)
    }
  }

  /**
   * Helper to acquire and fill a shape buffer in one step
   *
   * @param shape - Readonly shape array to convert
   * @returns Filled BigInt64Array buffer
   */
  fillShape(shape: readonly number[]): BigInt64Array {
    const buffer = this.acquire(shape.length)
    for (let i = 0; i < shape.length; i++) {
      buffer[i] = BigInt(shape[i]!)
    }
    return buffer
  }
}

/**
 * Cache for reusing Int32Array shape buffers (fast path)
 *
 * Int32 shapes avoid BigInt conversion overhead which is significant
 * in hot paths. Most tensor dimensions fit in int32.
 */
class ShapeBufferCache32 {
  private caches = new Map<number, Int32Array[]>()

  /**
   * Acquire a shape buffer for the given number of dimensions
   */
  acquire(ndim: number): Int32Array {
    const cache = this.caches.get(ndim)
    let buffer = cache?.pop()
    // Check if buffer was detached by FFI or is invalid
    if (!buffer || buffer.length !== ndim || buffer.buffer.byteLength === 0) {
      buffer = new Int32Array(ndim)
    }
    return buffer
  }

  /**
   * Release a shape buffer back to the cache
   */
  release(buffer: Int32Array): void {
    // Don't return detached buffers to the cache
    if (buffer.buffer.byteLength === 0) {
      return
    }
    const ndim = buffer.length
    let cache = this.caches.get(ndim)
    if (!cache) {
      cache = []
      this.caches.set(ndim, cache)
    }
    if (cache.length < SHAPE_CACHE_32_SIZE) {
      cache.push(buffer)
    }
  }

  /**
   * Check if shape can use int32 (all dimensions fit in int32)
   */
  canUseInt32(shape: readonly number[]): boolean {
    for (let i = 0; i < shape.length; i++) {
      const dim = shape[i]!
      if (!Number.isSafeInteger(dim) || dim < 0 || dim > MAX_INT32) {
        return false
      }
    }
    return true
  }

  /**
   * Helper to acquire and fill a shape buffer in one step
   *
   * @param shape - Readonly shape array to convert
   * @returns Filled Int32Array buffer, or null if shape requires int64
   */
  fillShape(shape: readonly number[]): Int32Array | null {
    if (!this.canUseInt32(shape)) {
      return null // Caller should fall back to int64 path
    }
    const buffer = this.acquire(shape.length)
    for (let i = 0; i < shape.length; i++) {
      buffer[i] = shape[i]!
    }
    return buffer
  }
}

/**
 * Global error buffer pool instance
 * Thread-safe for single-threaded JS execution
 */
export const errorPool = new ErrorBufferPool()

/**
 * Global shape buffer cache instance (int64 - legacy)
 * Thread-safe for single-threaded JS execution
 */
export const shapeCache = new ShapeBufferCache()

/**
 * Global int32 shape buffer cache instance (fast path)
 * Use this for most tensor operations where dimensions fit in int32
 */
export const shapeCache32 = new ShapeBufferCache32()
