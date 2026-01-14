import { describe, it, expect } from 'vitest'
import { SparseGraph } from '../utils/sparse-graph.js'

describe('SparseGraph', () => {
  describe('fromAdjacencyList()', () => {
    it('creates graph from adjacency list', () => {
      const graph = SparseGraph.fromAdjacencyList([
        [{ target: 1, weight: 1.0 }, { target: 2, weight: 2.0 }],
        [{ target: 0, weight: 1.0 }],
        [{ target: 0, weight: 2.0 }, { target: 1, weight: 1.5 }],
      ])

      expect(graph.numNodes).toBe(3)
      expect(graph.numEdges).toBe(5)
    })

    it('handles empty graph', () => {
      const graph = SparseGraph.fromAdjacencyList([])
      expect(graph.numNodes).toBe(0)
      expect(graph.numEdges).toBe(0)
    })

    it('handles nodes with no edges', () => {
      const graph = SparseGraph.fromAdjacencyList([
        [{ target: 1 }],
        [],
        [{ target: 0 }],
      ])

      expect(graph.numNodes).toBe(3)
      expect(graph.degree(1)).toBe(0)
    })

    it('uses default weight of 1.0', () => {
      const graph = SparseGraph.fromAdjacencyList([
        [{ target: 1 }],
        [],
      ])

      expect(graph.getWeight(0, 1)).toBe(1.0)
    })
  })

  describe('fromMap()', () => {
    it('creates graph from Map', () => {
      const adjacency = new Map([
        [0, [{ target: 1, weight: 1.0 }]],
        [1, [{ target: 2, weight: 2.0 }]],
        [2, []],
      ])

      const graph = SparseGraph.fromMap(adjacency)

      expect(graph.numNodes).toBe(3)
      expect(graph.numEdges).toBe(2)
    })
  })

  describe('empty()', () => {
    it('creates empty graph with given size', () => {
      const graph = SparseGraph.empty(5)

      expect(graph.numNodes).toBe(5)
      expect(graph.numEdges).toBe(0)
    })
  })

  describe('getNeighbors()', () => {
    it('returns neighbors as views into arrays', () => {
      const graph = SparseGraph.fromAdjacencyList([
        [{ target: 1, weight: 1.0 }, { target: 2, weight: 2.0 }],
        [],
        [],
      ])

      const neighbors = graph.getNeighbors(0)

      expect(neighbors.count).toBe(2)
      expect(neighbors.edges[0]).toBe(1)
      expect(neighbors.edges[1]).toBe(2)
      expect(neighbors.weights[0]).toBe(1.0)
      expect(neighbors.weights[1]).toBe(2.0)
    })

    it('returns empty for invalid node', () => {
      const graph = SparseGraph.fromAdjacencyList([[{ target: 1 }], []])

      const neighbors = graph.getNeighbors(99)

      expect(neighbors.count).toBe(0)
      expect(neighbors.edges.length).toBe(0)
    })

    it('returns empty for node with no neighbors', () => {
      const graph = SparseGraph.fromAdjacencyList([[{ target: 1 }], []])

      const neighbors = graph.getNeighbors(1)

      expect(neighbors.count).toBe(0)
    })
  })

  describe('neighbors()', () => {
    it('iterates over neighbors', () => {
      const graph = SparseGraph.fromAdjacencyList([
        [{ target: 1, weight: 1.0 }, { target: 2, weight: 2.0 }],
        [],
        [],
      ])

      const result = [...graph.neighbors(0)]

      expect(result.length).toBe(2)
      expect(result[0]).toEqual({ target: 1, weight: 1.0 })
      expect(result[1]).toEqual({ target: 2, weight: 2.0 })
    })
  })

  describe('hasEdge()', () => {
    it('returns true for existing edge', () => {
      const graph = SparseGraph.fromAdjacencyList([
        [{ target: 1 }],
        [],
      ])

      expect(graph.hasEdge(0, 1)).toBe(true)
    })

    it('returns false for non-existing edge', () => {
      const graph = SparseGraph.fromAdjacencyList([
        [{ target: 1 }],
        [],
      ])

      expect(graph.hasEdge(1, 0)).toBe(false)
      expect(graph.hasEdge(0, 2)).toBe(false)
    })
  })

  describe('getWeight()', () => {
    it('returns weight for existing edge', () => {
      const graph = SparseGraph.fromAdjacencyList([
        [{ target: 1, weight: 3.5 }],
        [],
      ])

      expect(graph.getWeight(0, 1)).toBe(3.5)
    })

    it('returns undefined for non-existing edge', () => {
      const graph = SparseGraph.fromAdjacencyList([
        [{ target: 1 }],
        [],
      ])

      expect(graph.getWeight(1, 0)).toBeUndefined()
    })
  })

  describe('degree()', () => {
    it('returns number of neighbors', () => {
      const graph = SparseGraph.fromAdjacencyList([
        [{ target: 1 }, { target: 2 }, { target: 3 }],
        [{ target: 0 }],
        [],
        [],
      ])

      expect(graph.degree(0)).toBe(3)
      expect(graph.degree(1)).toBe(1)
      expect(graph.degree(2)).toBe(0)
    })
  })

  describe('memoryUsage', () => {
    it('returns total byte size of internal arrays', () => {
      const graph = SparseGraph.fromAdjacencyList([
        [{ target: 1 }],
        [{ target: 0 }],
      ])

      // 3 nodes -> 3 * 4 bytes for offsets (Int32)
      // 2 edges -> 2 * 4 bytes for edges (Int32) + 2 * 4 bytes for weights (Float32)
      // Total: 12 + 8 + 8 = 28 bytes (approximately)
      expect(graph.memoryUsage).toBeGreaterThan(0)
    })
  })
})
