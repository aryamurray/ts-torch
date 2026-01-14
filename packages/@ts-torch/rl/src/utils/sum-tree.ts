/**
 * Sum Tree Data Structure
 *
 * A binary tree where each parent node stores the sum of its children.
 * Enables O(log n) proportional sampling for Prioritized Experience Replay.
 *
 * Structure:
 * - Leaf nodes store individual priorities
 * - Internal nodes store sum of children
 * - Root stores total sum of all priorities
 *
 * Example tree for capacity=4:
 * ```
 *           [sum=10]           <- root (index 0)
 *          /        \
 *      [sum=6]    [sum=4]      <- internal nodes
 *      /    \      /    \
 *    [3]   [3]   [2]   [2]     <- leaf nodes (priorities)
 * ```
 *
 * @example
 * ```ts
 * const tree = new SumTree(1000)
 *
 * // Add priorities
 * tree.update(0, 1.5)
 * tree.update(1, 2.0)
 *
 * // Sample proportionally
 * const value = Math.random() * tree.total
 * const index = tree.sample(value)
 * ```
 */

// ==================== Implementation ====================

/**
 * Sum Tree for efficient proportional sampling
 *
 * Time complexity:
 * - update(): O(log n)
 * - sample(): O(log n)
 * - total: O(1)
 * - min: O(n) first call, then O(1) if cached
 *
 * Space complexity: O(2n) for tree storage
 */
export class SumTree {
  /** Binary tree array - internal nodes + leaves */
  private readonly tree: Float64Array

  /** Capacity (number of leaf nodes) */
  private readonly capacity: number

  /** Index of first leaf node */
  private readonly leafOffset: number

  /** Cached minimum priority (for importance sampling) */
  private minPriority: number = Infinity

  /** Whether min cache is valid */
  private minCacheValid: boolean = false

  /**
   * Create a new SumTree
   *
   * @param capacity - Number of leaf nodes (priorities to store)
   */
  constructor(capacity: number) {
    if (capacity <= 0) {
      throw new Error('SumTree capacity must be positive')
    }

    this.capacity = capacity

    // Tree size: next power of 2 that fits capacity, times 2 for full binary tree
    const leafCount = this.nextPowerOf2(capacity)
    this.leafOffset = leafCount - 1
    this.tree = new Float64Array(2 * leafCount - 1)
  }

  /**
   * Update priority at leaf index
   *
   * @param index - Leaf index (0 to capacity-1)
   * @param priority - New priority value (must be positive)
   */
  update(index: number, priority: number): void {
    if (index < 0 || index >= this.capacity) {
      throw new Error(`Index ${index} out of bounds [0, ${this.capacity})`)
    }
    if (priority < 0) {
      throw new Error('Priority must be non-negative')
    }

    // Convert to tree index
    const treeIdx = this.leafOffset + index

    // Calculate change
    const change = priority - this.tree[treeIdx]!
    this.tree[treeIdx] = priority

    // Propagate change up to root
    this.propagateUp(treeIdx, change)

    // Invalidate min cache
    this.minCacheValid = false
  }

  /**
   * Sample a leaf index proportional to priorities
   *
   * Given a value in [0, total), returns the index where
   * the cumulative sum first exceeds the value.
   *
   * @param value - Random value in [0, total)
   * @returns Leaf index
   */
  sample(value: number): number {
    if (this.tree[0] === 0) {
      // All priorities are zero, sample uniformly
      return Math.floor(Math.random() * this.capacity)
    }

    // Clamp value to valid range
    value = Math.max(0, Math.min(value, this.tree[0]! - 1e-10))

    // Traverse down the tree
    let idx = 0

    while (idx < this.leafOffset) {
      const left = 2 * idx + 1
      const right = 2 * idx + 2

      const leftSum = this.tree[left] ?? 0

      if (value < leftSum) {
        idx = left
      } else {
        value -= leftSum
        idx = right
      }
    }

    // Convert tree index to leaf index
    return Math.min(idx - this.leafOffset, this.capacity - 1)
  }

  /**
   * Get priority at leaf index
   *
   * @param index - Leaf index
   * @returns Priority value
   */
  get(index: number): number {
    if (index < 0 || index >= this.capacity) {
      return 0
    }
    return this.tree[this.leafOffset + index] ?? 0
  }

  /**
   * Total sum of all priorities
   */
  get total(): number {
    return this.tree[0] ?? 0
  }

  /**
   * Minimum non-zero priority (for importance sampling normalization)
   * Returns Infinity if all priorities are zero.
   */
  get min(): number {
    if (this.minCacheValid) {
      return this.minPriority
    }

    // Scan leaves for minimum non-zero priority
    let min = Infinity
    for (let i = 0; i < this.capacity; i++) {
      const p = this.tree[this.leafOffset + i]!
      if (p > 0 && p < min) {
        min = p
      }
    }

    this.minPriority = min
    this.minCacheValid = true
    return min
  }

  /**
   * Maximum priority
   */
  get max(): number {
    let max = 0
    for (let i = 0; i < this.capacity; i++) {
      const p = this.tree[this.leafOffset + i]!
      if (p > max) {
        max = p
      }
    }
    return max
  }

  /**
   * Number of leaf nodes
   */
  get size(): number {
    return this.capacity
  }

  /**
   * Clear all priorities to zero
   */
  clear(): void {
    this.tree.fill(0)
    this.minCacheValid = false
    this.minPriority = Infinity
  }

  // ==================== Private Methods ====================

  /**
   * Propagate a change up to the root
   */
  private propagateUp(idx: number, change: number): void {
    while (idx > 0) {
      idx = Math.floor((idx - 1) / 2)
      this.tree[idx] = this.tree[idx]! + change
    }
  }

  /**
   * Find next power of 2 >= n
   */
  private nextPowerOf2(n: number): number {
    let p = 1
    while (p < n) {
      p *= 2
    }
    return p
  }
}
