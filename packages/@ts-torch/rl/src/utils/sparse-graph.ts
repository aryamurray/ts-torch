/**
 * High-Performance Sparse Graph
 *
 * CPU-optimized graph implementation using Compressed Sparse Row (CSR) format.
 * Essential for routing problems (like the London Bus example) to run fast in JS.
 *
 * CSR Format:
 * - nodeOffsets[i] gives the starting index in edges[] for node i's neighbors
 * - edges[] contains all neighbor node IDs in a flat array
 * - weights[] contains corresponding edge weights
 *
 * Guarantees:
 * - Zero object allocation during queries
 * - O(1) access to neighbor list start/end
 * - Cache-friendly contiguous memory layout
 *
 * @example
 * ```ts
 * // Create from adjacency list
 * const graph = SparseGraph.fromAdjacencyList([
 *   [{ target: 1, weight: 1.0 }, { target: 2, weight: 2.0 }],  // Node 0
 *   [{ target: 0, weight: 1.0 }, { target: 2, weight: 1.5 }],  // Node 1
 *   [{ target: 0, weight: 2.0 }, { target: 1, weight: 1.5 }],  // Node 2
 * ])
 *
 * // Query neighbors (zero allocation)
 * const neighbors = graph.getNeighbors(0)
 * // neighbors = { edges: Int32Array[1, 2], weights: Float32Array[1.0, 2.0], count: 2 }
 * ```
 */

// ==================== Types ====================

/**
 * Entry in adjacency list (used for construction)
 */
export interface AdjacencyEntry {
  /** Target node ID */
  target: number
  /** Edge weight (default: 1.0) */
  weight?: number
}

/**
 * Result of getNeighbors() - provides views into internal arrays
 */
export interface NeighborView {
  /** Neighbor node IDs (view into edges array) */
  edges: Int32Array
  /** Edge weights (view into weights array) */
  weights: Float32Array
  /** Number of neighbors */
  count: number
}

// ==================== Implementation ====================

/**
 * Compressed Sparse Row (CSR) Graph
 *
 * Stores graph in three flat arrays for optimal cache performance:
 * - nodeOffsets: Index into edges/weights for each node's neighbor list
 * - edges: Flat array of all neighbor IDs
 * - weights: Flat array of all edge weights
 */
export class SparseGraph {
  private readonly nodeOffsets: Int32Array
  private readonly edges: Int32Array
  private readonly weights: Float32Array
  private readonly numNodes_: number
  private readonly numEdges_: number

  /**
   * Private constructor - use factory methods
   */
  private constructor(
    nodeOffsets: Int32Array,
    edges: Int32Array,
    weights: Float32Array,
  ) {
    this.nodeOffsets = nodeOffsets
    this.edges = edges
    this.weights = weights
    this.numNodes_ = nodeOffsets.length - 1
    this.numEdges_ = edges.length
  }

  /**
   * Create a SparseGraph from an adjacency list
   *
   * @param adjacency - Array of arrays, where adjacency[i] contains neighbors of node i
   * @returns SparseGraph instance
   *
   * @example
   * ```ts
   * const graph = SparseGraph.fromAdjacencyList([
   *   [{ target: 1, weight: 1.0 }],  // Node 0 -> Node 1
   *   [{ target: 0, weight: 1.0 }, { target: 2, weight: 2.0 }],  // Node 1 -> 0, 2
   *   [{ target: 1, weight: 2.0 }],  // Node 2 -> Node 1
   * ])
   * ```
   */
  static fromAdjacencyList(adjacency: AdjacencyEntry[][]): SparseGraph {
    const numNodes = adjacency.length
    const nodeOffsets = new Int32Array(numNodes + 1)

    // Count total edges
    let totalEdges = 0
    for (let i = 0; i < numNodes; i++) {
      totalEdges += adjacency[i]?.length ?? 0
    }

    const edges = new Int32Array(totalEdges)
    const weights = new Float32Array(totalEdges)

    // Fill arrays
    let offset = 0
    for (let i = 0; i < numNodes; i++) {
      nodeOffsets[i] = offset
      const neighbors = adjacency[i] ?? []

      for (const neighbor of neighbors) {
        edges[offset] = neighbor.target
        weights[offset] = neighbor.weight ?? 1.0
        offset++
      }
    }
    nodeOffsets[numNodes] = offset

    return new SparseGraph(nodeOffsets, edges, weights)
  }

  /**
   * Create a SparseGraph from a Map-based adjacency list
   *
   * @param adjacency - Map from node ID to list of neighbors
   * @returns SparseGraph instance
   */
  static fromMap(adjacency: Map<number, AdjacencyEntry[]>): SparseGraph {
    // Find max node ID to determine size
    let maxNode = -1
    for (const [node, neighbors] of adjacency) {
      maxNode = Math.max(maxNode, node)
      for (const n of neighbors) {
        maxNode = Math.max(maxNode, n.target)
      }
    }

    const numNodes = maxNode + 1
    const adjArray: AdjacencyEntry[][] = new Array(numNodes)

    for (let i = 0; i < numNodes; i++) {
      adjArray[i] = adjacency.get(i) ?? []
    }

    return SparseGraph.fromAdjacencyList(adjArray)
  }

  /**
   * Create an empty graph with given number of nodes
   */
  static empty(numNodes: number): SparseGraph {
    const nodeOffsets = new Int32Array(numNodes + 1)
    const edges = new Int32Array(0)
    const weights = new Float32Array(0)
    return new SparseGraph(nodeOffsets, edges, weights)
  }

  /**
   * Get neighbors of a node
   *
   * Returns views into internal arrays - ZERO ALLOCATION
   *
   * @param node - Node ID
   * @returns View containing edges, weights, and count
   */
  getNeighbors(node: number): NeighborView {
    if (node < 0 || node >= this.numNodes_) {
      return { edges: new Int32Array(0), weights: new Float32Array(0), count: 0 }
    }

    const start = this.nodeOffsets[node]!
    const end = this.nodeOffsets[node + 1]!
    const count = end - start

    return {
      edges: this.edges.subarray(start, end),
      weights: this.weights.subarray(start, end),
      count,
    }
  }

  /**
   * Iterate over neighbors of a node
   *
   * Alternative to getNeighbors() that yields individual edges.
   * Slightly less efficient but more convenient for some use cases.
   *
   * @param node - Node ID
   * @yields Neighbor target and weight
   */
  *neighbors(node: number): Generator<{ target: number; weight: number }> {
    if (node < 0 || node >= this.numNodes_) {
      return
    }

    const start = this.nodeOffsets[node]!
    const end = this.nodeOffsets[node + 1]!

    for (let i = start; i < end; i++) {
      yield { target: this.edges[i]!, weight: this.weights[i]! }
    }
  }

  /**
   * Check if an edge exists between two nodes
   *
   * @param from - Source node
   * @param to - Target node
   * @returns True if edge exists
   */
  hasEdge(from: number, to: number): boolean {
    if (from < 0 || from >= this.numNodes_) return false

    const start = this.nodeOffsets[from]!
    const end = this.nodeOffsets[from + 1]!

    for (let i = start; i < end; i++) {
      if (this.edges[i] === to) return true
    }

    return false
  }

  /**
   * Get weight of an edge
   *
   * @param from - Source node
   * @param to - Target node
   * @returns Edge weight, or undefined if edge doesn't exist
   */
  getWeight(from: number, to: number): number | undefined {
    if (from < 0 || from >= this.numNodes_) return undefined

    const start = this.nodeOffsets[from]!
    const end = this.nodeOffsets[from + 1]!

    for (let i = start; i < end; i++) {
      if (this.edges[i] === to) {
        return this.weights[i]
      }
    }

    return undefined
  }

  /**
   * Get degree (number of neighbors) of a node
   *
   * @param node - Node ID
   * @returns Number of neighbors
   */
  degree(node: number): number {
    if (node < 0 || node >= this.numNodes_) return 0
    return this.nodeOffsets[node + 1]! - this.nodeOffsets[node]!
  }

  /**
   * Number of nodes in the graph
   */
  get numNodes(): number {
    return this.numNodes_
  }

  /**
   * Number of edges in the graph
   */
  get numEdges(): number {
    return this.numEdges_
  }

  /**
   * Total memory used by internal arrays (bytes)
   */
  get memoryUsage(): number {
    return (
      this.nodeOffsets.byteLength +
      this.edges.byteLength +
      this.weights.byteLength
    )
  }
}
