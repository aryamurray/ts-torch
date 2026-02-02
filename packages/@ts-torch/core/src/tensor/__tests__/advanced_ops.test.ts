/**
 * Tests for advanced tensor operations
 * (triu, tril, maskedFill, bmm, gather, scatter, topk, sort, where, nonzero, repeat, expand)
 */

import { describe, test, expect } from 'vitest'
import { device } from '../../device/index.js'
import { run } from '../../memory/scope.js'
import { DType } from '../../types/dtype.js'
import { einsum } from '../factory.js'

const int64 = DType.int64

const cpu = device.cpu()

describe('triu (upper triangular)', () => {
  test('extracts upper triangular matrix', () => {
    run(() => {
      const matrix = cpu.ones([3, 3] as const)
      const upper = matrix.triu()

      const data = upper.toArray()
      // Upper triangular: lower left should be 0
      expect(data[3]).toBe(0) // [1,0]
      expect(data[6]).toBe(0) // [2,0]
      expect(data[7]).toBe(0) // [2,1]
      // Upper right should be 1
      expect(data[0]).toBe(1) // [0,0]
      expect(data[1]).toBe(1) // [0,1]
      expect(data[4]).toBe(1) // [1,1]
    })
  })

  test('accepts diagonal parameter', () => {
    run(() => {
      const matrix = cpu.ones([4, 4] as const)

      // diagonal=1: exclude main diagonal
      const upper1 = matrix.triu(1)
      const data1 = upper1.toArray()
      expect(data1[0]).toBe(0) // [0,0] excluded

      // diagonal=-1: include one diagonal below main
      const upperNeg1 = matrix.triu(-1)
      const dataNeg1 = upperNeg1.toArray()
      expect(dataNeg1[4]).toBe(1) // [1,0] included
    })
  })

  test('preserves shape', () => {
    run(() => {
      const matrix = cpu.randn([5, 5] as const)
      const upper = matrix.triu()

      expect(upper.shape).toEqual([5, 5])
    })
  })

  test('handles non-square matrices', () => {
    run(() => {
      const matrix = cpu.ones([3, 5] as const)
      const upper = matrix.triu()

      expect(upper.shape).toEqual([3, 5])
    })
  })
})

describe('tril (lower triangular)', () => {
  test('extracts lower triangular matrix', () => {
    run(() => {
      const matrix = cpu.ones([3, 3] as const)
      const lower = matrix.tril()

      const data = lower.toArray()
      // Lower triangular: upper right should be 0
      expect(data[1]).toBe(0) // [0,1]
      expect(data[2]).toBe(0) // [0,2]
      expect(data[5]).toBe(0) // [1,2]
      // Lower left should be 1
      expect(data[0]).toBe(1) // [0,0]
      expect(data[3]).toBe(1) // [1,0]
      expect(data[6]).toBe(1) // [2,0]
    })
  })

  test('accepts diagonal parameter', () => {
    run(() => {
      const matrix = cpu.ones([4, 4] as const)

      // diagonal=-1: exclude main diagonal
      const lower = matrix.tril(-1)
      const data = lower.toArray()
      expect(data[0]).toBe(0) // [0,0] excluded
    })
  })

  test('preserves shape', () => {
    run(() => {
      const matrix = cpu.randn([5, 5] as const)
      const lower = matrix.tril()

      expect(lower.shape).toEqual([5, 5])
    })
  })
})

describe('maskedFill', () => {
  test('fills tensor values where mask is true', () => {
    run(() => {
      const tensor = cpu.ones([2, 3] as const)
      // Boolean arrays are now auto-detected and create bool dtype tensors
      const mask = cpu.tensor([true, false, true, false, true, false], [2, 3] as const)

      const result = tensor.maskedFill(mask as any, -Infinity)

      const data = result.toArray()
      expect(data[0]).toBe(-Infinity) // mask=true
      expect(data[1]).toBe(1) // mask=false
      expect(data[2]).toBe(-Infinity) // mask=true
    })
  })

  test('handles zero fill value', () => {
    run(() => {
      const tensor = cpu.ones([4] as const)
      const mask = cpu.tensor([true, true, false, false], [4] as const)

      const result = tensor.maskedFill(mask as any, 0)

      const data = result.toArray()
      expect(data[0]).toBe(0)
      expect(data[1]).toBe(0)
      expect(data[2]).toBe(1)
      expect(data[3]).toBe(1)
    })
  })

  test('preserves shape', () => {
    run(() => {
      const tensor = cpu.randn([3, 4, 5] as const)
      // Create a bool tensor with all false values
      const maskData = new Array(3 * 4 * 5).fill(false)
      const mask = cpu.tensor(maskData, [3, 4, 5] as const)

      const result = tensor.maskedFill(mask as any, 0)

      expect(result.shape).toEqual([3, 4, 5])
    })
  })
})

describe('bmm (batched matrix multiplication)', () => {
  test('multiplies batches of matrices', () => {
    run(() => {
      const a = cpu.randn([10, 3, 4] as const) // [batch, m, k]
      const b = cpu.randn([10, 4, 5] as const) // [batch, k, n]

      const result = a.bmm(b)

      expect(result.shape).toEqual([10, 3, 5])
    })
  })

  test('handles single batch', () => {
    run(() => {
      const a = cpu.randn([1, 2, 3] as const)
      const b = cpu.randn([1, 3, 4] as const)

      const result = a.bmm(b)

      expect(result.shape).toEqual([1, 2, 4])
    })
  })

  test('handles large batches', () => {
    run(() => {
      const a = cpu.randn([64, 8, 16] as const)
      const b = cpu.randn([64, 16, 32] as const)

      const result = a.bmm(b)

      expect(result.shape).toEqual([64, 8, 32])
    })
  })

  test('produces correct result for known input', () => {
    run(() => {
      // Single batch identity multiplication
      const a = cpu.tensor([1, 0, 0, 1], [1, 2, 2] as const) // Identity
      const b = cpu.tensor([1, 2, 3, 4], [1, 2, 2] as const)

      const result = a.bmm(b)

      const data = result.toArray()
      expect(data[0]).toBeCloseTo(1)
      expect(data[1]).toBeCloseTo(2)
      expect(data[2]).toBeCloseTo(3)
      expect(data[3]).toBeCloseTo(4)
    })
  })
})

describe('gather', () => {
  test('gathers values along specified dimension', () => {
    run(() => {
      const tensor = cpu.tensor([1, 2, 3, 4, 5, 6], [2, 3] as const)
      const indices = cpu.tensor([0, 1, 0, 2], [2, 2] as const, int64)

      const result = tensor.gather(1, indices)

      expect(result.shape).toEqual([2, 2])
    })
  })

  test('gathers along dimension 0', () => {
    run(() => {
      const tensor = cpu.tensor([1, 2, 3, 4, 5, 6], [2, 3] as const)
      const indices = cpu.tensor([0, 1, 0], [1, 3] as const, int64)

      const result = tensor.gather(0, indices)

      expect(result.shape).toEqual([1, 3])
    })
  })
})

describe('scatter', () => {
  test('scatters values along dimension 1', () => {
    run(() => {
      const src = cpu.tensor([1, 2, 3, 4, 5, 6], [2, 3] as const)
      const indices = cpu.tensor([0, 1, 2, 2, 1, 0], [2, 3] as const, int64)
      const input = cpu.zeros([2, 3] as const)

      const result = input.scatter(1, indices, src)

      expect(result.shape).toEqual([2, 3])
      const data = result.toArray()
      // Row 0: indices [0,1,2] scatter values [1,2,3] to positions 0,1,2
      expect(data[0]).toBe(1) // position 0
      expect(data[1]).toBe(2) // position 1
      expect(data[2]).toBe(3) // position 2
      // Row 1: indices [2,1,0] scatter values [4,5,6] to positions 2,1,0
      expect(data[3]).toBe(6) // position 0 gets value 6
      expect(data[4]).toBe(5) // position 1 gets value 5
      expect(data[5]).toBe(4) // position 2 gets value 4
    })
  })

  test('scatters values along dimension 0', () => {
    run(() => {
      const src = cpu.tensor([10, 20, 30], [1, 3] as const)
      const indices = cpu.tensor([1, 0, 1], [1, 3] as const, int64)
      const input = cpu.zeros([2, 3] as const)

      const result = input.scatter(0, indices, src)

      expect(result.shape).toEqual([2, 3])
      const data = result.toArray()
      // Column 0: index 1 -> row 1 gets 10
      // Column 1: index 0 -> row 0 gets 20
      // Column 2: index 1 -> row 1 gets 30
      expect(data[0]).toBe(0)  // [0,0]
      expect(data[1]).toBe(20) // [0,1]
      expect(data[2]).toBe(0)  // [0,2]
      expect(data[3]).toBe(10) // [1,0]
      expect(data[4]).toBe(0)  // [1,1]
      expect(data[5]).toBe(30) // [1,2]
    })
  })

  test('handles overlapping indices (last write wins)', () => {
    run(() => {
      const src = cpu.tensor([1, 2, 3], [3] as const)
      const indices = cpu.tensor([0, 0, 0], [3] as const, int64)
      const input = cpu.zeros([3] as const)

      const result = input.scatter(0, indices, src)

      expect(result.shape).toEqual([3])
      // All values scatter to index 0, last write (3) wins
      const data = result.toArray()
      expect(data[0]).toBe(3)
      expect(data[1]).toBe(0)
      expect(data[2]).toBe(0)
    })
  })
})

describe('topk', () => {
  test('returns top k values and indices', () => {
    run(() => {
      const tensor = cpu.tensor([3, 1, 4, 1, 5, 9, 2, 6], [8] as const)

      const [values, indices] = tensor.topk(3)

      expect(values.shape).toEqual([3])
      expect(indices.shape).toEqual([3])

      // Top 3 values should be 9, 6, 5 (largest)
      const valuesData = values.toArray()
      expect(valuesData[0]).toBe(9)
      expect(valuesData[1]).toBe(6)
      expect(valuesData[2]).toBe(5)
    })
  })

  test('returns smallest k when largest=false', () => {
    run(() => {
      const tensor = cpu.tensor([3, 1, 4, 1, 5, 9, 2, 6], [8] as const)

      const [values, indices] = tensor.topk(3, -1, false)

      // Smallest 3 values should be 1, 1, 2
      const valuesData = values.toArray()
      expect(valuesData[0]).toBe(1)
      expect(valuesData[1]).toBe(1)
      expect(valuesData[2]).toBe(2)
    })
  })

  test('handles 2D tensor along last dimension', () => {
    run(() => {
      const tensor = cpu.tensor([3, 1, 4, 1, 5, 9], [2, 3] as const)

      const [values, indices] = tensor.topk(2)

      expect(values.shape).toEqual([2, 2])
      expect(indices.shape).toEqual([2, 2])
    })
  })

  test('handles specified dimension', () => {
    run(() => {
      const tensor = cpu.tensor([3, 1, 4, 1, 5, 9], [2, 3] as const)

      const [values, indices] = tensor.topk(1, 0) // Top along dim 0

      expect(values.shape).toEqual([1, 3])
    })
  })
})

describe('sort', () => {
  test('sorts tensor and returns values and indices', () => {
    run(() => {
      const tensor = cpu.tensor([3, 1, 4, 1, 5, 9, 2, 6], [8] as const)

      const [sorted, indices] = tensor.sort()

      expect(sorted.shape).toEqual([8])
      expect(indices.shape).toEqual([8])

      // Check sorted in ascending order
      const data = sorted.toArray()
      for (let i = 1; i < data.length; i++) {
        expect(data[i]).toBeGreaterThanOrEqual(data[i - 1])
      }
    })
  })

  test('sorts in descending order', () => {
    run(() => {
      const tensor = cpu.tensor([3, 1, 4, 1, 5, 9, 2, 6], [8] as const)

      const [sorted, indices] = tensor.sort(-1, true) // descending=true

      const data = sorted.toArray()
      for (let i = 1; i < data.length; i++) {
        expect(data[i]).toBeLessThanOrEqual(data[i - 1])
      }
    })
  })

  test('handles 2D tensor', () => {
    run(() => {
      const tensor = cpu.tensor([3, 1, 4, 6, 5, 2], [2, 3] as const)

      const [sorted, indices] = tensor.sort(-1) // Sort along last dim

      expect(sorted.shape).toEqual([2, 3])
      expect(indices.shape).toEqual([2, 3])
    })
  })
})

describe('nonzero', () => {
  test('handles all zeros', () => {
    run(() => {
      const tensor = cpu.zeros([10] as const)

      const indices = tensor.nonzero()

      expect(indices.shape[0]).toBe(0)
    })
  })

  test('returns correct shape for nonzero elements', () => {
    run(() => {
      const tensor = cpu.randn([5, 5] as const)

      const indices = tensor.nonzero()

      // Should return [N, 2] where N is number of non-zero elements
      expect(indices.shape.length).toBe(2)
      expect(indices.shape[1]).toBe(2) // 2D coordinates
    })
  })
})

describe('repeat', () => {
  test('repeats tensor along dimensions', () => {
    run(() => {
      const tensor = cpu.tensor([1, 2, 3], [3] as const)

      const repeated = tensor.repeat([2])

      expect(repeated.shape).toEqual([6])
      const data = Array.from(repeated.toArray())
      expect(data).toEqual([1, 2, 3, 1, 2, 3])
    })
  })

  test('repeats 2D tensor', () => {
    run(() => {
      const tensor = cpu.tensor([1, 2, 3, 4], [2, 2] as const)

      const repeated = tensor.repeat([2, 3])

      expect(repeated.shape).toEqual([4, 6])
    })
  })

  // Note: Adding dimensions via repeat requires reshaping first in this implementation
  test('repeats 1D tensor with reshape for dimension expansion', () => {
    run(() => {
      const tensor = cpu.tensor([1, 2], [1, 2] as const) // Reshaped to 2D

      const repeated = tensor.repeat([2, 3])

      expect(repeated.shape).toEqual([2, 6])
    })
  })
})

describe('expand', () => {
  test('expands tensor to larger shape', () => {
    run(() => {
      const tensor = cpu.tensor([1, 2, 3], [1, 3] as const)

      const expanded = tensor.expand([4, 3])

      expect(expanded.shape).toEqual([4, 3])
    })
  })

  test('handles -1 to keep dimension unchanged', () => {
    run(() => {
      const tensor = cpu.tensor([1, 2, 3], [1, 3] as const)

      const expanded = tensor.expand([-1, 3])

      expect(expanded.shape).toEqual([1, 3])
    })
  })

  test('broadcasts scalar to larger shape', () => {
    run(() => {
      const tensor = cpu.tensor([5], [1, 1] as const)

      const expanded = tensor.expand([3, 4])

      expect(expanded.shape).toEqual([3, 4])
    })
  })
})

describe('combination of operations', () => {
  test('triu for causal attention mask', () => {
    run(() => {
      const seqLen = 5
      const ones = cpu.ones([seqLen, seqLen] as const)
      const causalMask = ones.triu(1) // Upper triangle excluding diagonal

      // This mask is used to mask out future tokens in self-attention
      const data = causalMask.toArray()
      expect(data[0]).toBe(0) // [0,0] = diagonal, should be 0
      expect(data[1]).toBe(1) // [0,1] = above diagonal, should be 1
    })
  })

  // Note: maskedFill requires boolean tensors which need tensor.to(bool) or
  // creation with bool dtype (currently unsupported in tensor creation).
  // The maskedFill tests in the maskedFill describe block above use the
  // basic functionality tests instead.
})

describe('einsum (Einstein summation)', () => {
  test('matrix multiplication via einsum', () => {
    run(() => {
      // a: [2, 3], b: [3, 4] -> result: [2, 4]
      const a = cpu.tensor([1, 2, 3, 4, 5, 6], [2, 3] as const)
      const b = cpu.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [3, 4] as const)

      const result = einsum('ij,jk->ik', [a, b])

      expect(result.shape).toEqual([2, 4])

      // Verify against matmul result
      const matmulResult = a.matmul(b)
      const einsumData = result.toArray()
      const matmulData = matmulResult.toArray()

      for (let i = 0; i < einsumData.length; i++) {
        expect(einsumData[i]).toBeCloseTo(matmulData[i]!)
      }
    })
  })

  test('batched matrix multiplication via einsum', () => {
    run(() => {
      // a: [2, 3, 4], b: [2, 4, 5] -> result: [2, 3, 5]
      const a = cpu.randn([2, 3, 4] as const)
      const b = cpu.randn([2, 4, 5] as const)

      const result = einsum('bij,bjk->bik', [a, b])

      expect(result.shape).toEqual([2, 3, 5])

      // Compare with bmm
      const bmmResult = a.bmm(b)
      const einsumData = result.toArray()
      const bmmData = bmmResult.toArray()

      for (let i = 0; i < einsumData.length; i++) {
        expect(einsumData[i]).toBeCloseTo(bmmData[i]!, 4)
      }
    })
  })

  test('matrix trace via einsum', () => {
    run(() => {
      // Identity matrix trace should be n
      const identity = cpu.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1], [3, 3] as const)

      const trace = einsum('ii->', [identity])

      expect(trace.shape).toEqual([])
      expect(trace.item()).toBeCloseTo(3)
    })
  })

  test('transpose via einsum', () => {
    run(() => {
      const matrix = cpu.tensor([1, 2, 3, 4, 5, 6], [2, 3] as const)

      const transposed = einsum('ij->ji', [matrix])

      expect(transposed.shape).toEqual([3, 2])

      // Verify transpose is correct
      const data = transposed.toArray()
      expect(data[0]).toBe(1) // [0,0]
      expect(data[1]).toBe(4) // [0,1] = original [1,0]
      expect(data[2]).toBe(2) // [1,0] = original [0,1]
      expect(data[3]).toBe(5) // [1,1]
    })
  })

  test('dot product via einsum', () => {
    run(() => {
      const a = cpu.tensor([1, 2, 3], [3] as const)
      const b = cpu.tensor([4, 5, 6], [3] as const)

      const dot = einsum('i,i->', [a, b])

      expect(dot.shape).toEqual([])
      // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
      expect(dot.item()).toBeCloseTo(32)
    })
  })

  test('outer product via einsum', () => {
    run(() => {
      const a = cpu.tensor([1, 2], [2] as const)
      const b = cpu.tensor([3, 4, 5], [3] as const)

      const outer = einsum('i,j->ij', [a, b])

      expect(outer.shape).toEqual([2, 3])
      const data = outer.toArray()
      // [[1*3, 1*4, 1*5], [2*3, 2*4, 2*5]]
      expect(data[0]).toBeCloseTo(3)
      expect(data[1]).toBeCloseTo(4)
      expect(data[2]).toBeCloseTo(5)
      expect(data[3]).toBeCloseTo(6)
      expect(data[4]).toBeCloseTo(8)
      expect(data[5]).toBeCloseTo(10)
    })
  })

  test('sum over all elements via einsum', () => {
    run(() => {
      const tensor = cpu.tensor([1, 2, 3, 4, 5, 6], [2, 3] as const)

      const sum = einsum('ij->', [tensor])

      expect(sum.shape).toEqual([])
      expect(sum.item()).toBeCloseTo(21)
    })
  })

  test('diagonal extraction via einsum', () => {
    run(() => {
      const matrix = cpu.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 3] as const)

      const diagonal = einsum('ii->i', [matrix])

      expect(diagonal.shape).toEqual([3])
      const data = diagonal.toArray()
      expect(data[0]).toBeCloseTo(1) // [0,0]
      expect(data[1]).toBeCloseTo(5) // [1,1]
      expect(data[2]).toBeCloseTo(9) // [2,2]
    })
  })

  test('attention pattern via einsum', () => {
    run(() => {
      // Simulated attention: Q @ K^T
      const batchSize = 2
      const seqLen = 4
      const headDim = 8

      const Q = cpu.randn([batchSize, seqLen, headDim] as const)
      const K = cpu.randn([batchSize, seqLen, headDim] as const)

      // Q @ K^T: 'bqh,bkh->bqk' (batch, query, key)
      const scores = einsum('bqh,bkh->bqk', [Q, K])

      expect(scores.shape).toEqual([batchSize, seqLen, seqLen])
    })
  })
})
