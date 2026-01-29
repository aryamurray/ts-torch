/**
 * Tests for RNN, LSTM, and GRU modules
 */

import { describe, test, expect } from 'vitest'
import { RNN, LSTM, GRU } from '../rnn.js'
import { device, run } from '@ts-torch/core'

const cpu = device.cpu()

describe('RNN', () => {
  describe('constructor', () => {
    test('creates RNN with correct dimensions', () => {
      const rnn = new RNN(256, 512)

      expect(rnn.inputSize).toBe(256)
      expect(rnn.hiddenSize).toBe(512)
      expect(rnn.numLayers).toBe(1)
    })

    test('creates multi-layer RNN', () => {
      const rnn = new RNN(128, 256, { numLayers: 3 })

      expect(rnn.numLayers).toBe(3)
    })

    test('throws error for invalid inputSize', () => {
      expect(() => new RNN(0, 256)).toThrow('must be positive')
      expect(() => new RNN(-128, 256)).toThrow('must be positive')
    })

    test('throws error for invalid hiddenSize', () => {
      expect(() => new RNN(128, 0)).toThrow('must be positive')
      expect(() => new RNN(128, -256)).toThrow('must be positive')
    })

    test('default nonlinearity is tanh', () => {
      const rnn = new RNN(128, 256)

      expect(rnn.nonlinearity).toBe('tanh')
    })

    test('can use relu nonlinearity', () => {
      const rnn = new RNN(128, 256, { nonlinearity: 'relu' })

      expect(rnn.nonlinearity).toBe('relu')
    })

    test('default batchFirst is false', () => {
      const rnn = new RNN(128, 256)

      expect(rnn.batchFirst).toBe(false)
    })

    test('can set batchFirst', () => {
      const rnn = new RNN(128, 256, { batchFirst: true })

      expect(rnn.batchFirst).toBe(true)
    })

    test('default bias is true', () => {
      const rnn = new RNN(128, 256)

      expect(rnn.bias).toBe(true)
    })

    test('can disable bias', () => {
      const rnn = new RNN(128, 256, { bias: false })

      expect(rnn.bias).toBe(false)
    })
  })

  describe('forward pass', () => {
    test('processes sequence with seq-first format', () => {
      run(() => {
        const rnn = new RNN(64, 128)
        const input = cpu.randn([10, 2, 64] as const) // [seq, batch, input]

        const [output, hidden] = rnn.forward(input)

        expect(output.shape).toEqual([10, 2, 128])
        expect(hidden.shape).toEqual([1, 2, 128])
      })
    })

    test('processes sequence with batch-first format', () => {
      run(() => {
        const rnn = new RNN(64, 128, { batchFirst: true })
        const input = cpu.randn([2, 10, 64] as const) // [batch, seq, input]

        const [output, hidden] = rnn.forward(input)

        expect(output.shape).toEqual([2, 10, 128])
        expect(hidden.shape).toEqual([1, 2, 128])
      })
    })

    test('accepts initial hidden state', () => {
      run(() => {
        const rnn = new RNN(64, 128)
        const input = cpu.randn([10, 2, 64] as const)
        const h0 = cpu.randn([1, 2, 128] as const)

        const [output, hidden] = rnn.forward(input, h0)

        expect(output.shape).toEqual([10, 2, 128])
        expect(hidden.shape).toEqual([1, 2, 128])
      })
    })

    test('handles single timestep', () => {
      run(() => {
        const rnn = new RNN(64, 128)
        const input = cpu.randn([1, 2, 64] as const)

        const [output, hidden] = rnn.forward(input)

        expect(output.shape).toEqual([1, 2, 128])
      })
    })
  })

  describe('parameters', () => {
    test('has trainable parameters', () => {
      const rnn = new RNN(64, 128)
      const params = rnn.parameters()

      expect(params.length).toBeGreaterThan(0)
    })

    test('multi-layer has more parameters', () => {
      const rnn1 = new RNN(64, 128, { numLayers: 1 })
      const rnn2 = new RNN(64, 128, { numLayers: 2 })

      expect(rnn2.parameters().length).toBeGreaterThan(rnn1.parameters().length)
    })
  })

  describe('toString', () => {
    test('returns descriptive string', () => {
      const rnn = new RNN(256, 512, { numLayers: 2 })
      const str = rnn.toString()

      expect(str).toContain('RNN')
      expect(str).toContain('256')
      expect(str).toContain('512')
      expect(str).toContain('num_layers=2')
    })
  })
})

describe('LSTM', () => {
  describe('constructor', () => {
    test('creates LSTM with correct dimensions', () => {
      const lstm = new LSTM(256, 512)

      expect(lstm.inputSize).toBe(256)
      expect(lstm.hiddenSize).toBe(512)
      expect(lstm.numLayers).toBe(1)
    })

    test('creates multi-layer LSTM', () => {
      const lstm = new LSTM(128, 256, { numLayers: 3 })

      expect(lstm.numLayers).toBe(3)
    })

    test('throws error for invalid dimensions', () => {
      expect(() => new LSTM(0, 256)).toThrow('must be positive')
      expect(() => new LSTM(128, 0)).toThrow('must be positive')
    })
  })

  describe('forward pass', () => {
    test('processes sequence and returns output and states', () => {
      run(() => {
        const lstm = new LSTM(64, 128)
        const input = cpu.randn([10, 2, 64] as const) // [seq, batch, input]

        const [output, [hn, cn]] = lstm.forward(input)

        expect(output.shape).toEqual([10, 2, 128])
        expect(hn.shape).toEqual([1, 2, 128])
        expect(cn.shape).toEqual([1, 2, 128])
      })
    })

    test('processes with batch-first format', () => {
      run(() => {
        const lstm = new LSTM(64, 128, { batchFirst: true })
        const input = cpu.randn([2, 10, 64] as const) // [batch, seq, input]

        const [output, [hn, cn]] = lstm.forward(input)

        expect(output.shape).toEqual([2, 10, 128])
      })
    })

    test('accepts initial hidden and cell states', () => {
      run(() => {
        const lstm = new LSTM(64, 128)
        const input = cpu.randn([10, 2, 64] as const)
        const h0 = cpu.randn([1, 2, 128] as const)
        const c0 = cpu.randn([1, 2, 128] as const)

        const [output, [hn, cn]] = lstm.forward(input, [h0, c0])

        expect(output.shape).toEqual([10, 2, 128])
      })
    })
  })

  describe('parameters', () => {
    test('has trainable parameters (4x gates)', () => {
      const lstm = new LSTM(64, 128)
      const params = lstm.parameters()

      // LSTM has 4x parameters compared to RNN (i, f, g, o gates)
      expect(params.length).toBeGreaterThan(0)
    })
  })

  describe('toString', () => {
    test('returns descriptive string', () => {
      const lstm = new LSTM(256, 512, { numLayers: 2 })
      const str = lstm.toString()

      expect(str).toContain('LSTM')
      expect(str).toContain('256')
      expect(str).toContain('512')
    })
  })
})

describe('GRU', () => {
  describe('constructor', () => {
    test('creates GRU with correct dimensions', () => {
      const gru = new GRU(256, 512)

      expect(gru.inputSize).toBe(256)
      expect(gru.hiddenSize).toBe(512)
      expect(gru.numLayers).toBe(1)
    })

    test('creates multi-layer GRU', () => {
      const gru = new GRU(128, 256, { numLayers: 3 })

      expect(gru.numLayers).toBe(3)
    })

    test('throws error for invalid dimensions', () => {
      expect(() => new GRU(0, 256)).toThrow('must be positive')
      expect(() => new GRU(128, 0)).toThrow('must be positive')
    })
  })

  describe('forward pass', () => {
    test('processes sequence and returns output and hidden', () => {
      run(() => {
        const gru = new GRU(64, 128)
        const input = cpu.randn([10, 2, 64] as const) // [seq, batch, input]

        const [output, hn] = gru.forward(input)

        expect(output.shape).toEqual([10, 2, 128])
        expect(hn.shape).toEqual([1, 2, 128])
      })
    })

    test('processes with batch-first format', () => {
      run(() => {
        const gru = new GRU(64, 128, { batchFirst: true })
        const input = cpu.randn([2, 10, 64] as const) // [batch, seq, input]

        const [output, hn] = gru.forward(input)

        expect(output.shape).toEqual([2, 10, 128])
      })
    })

    test('accepts initial hidden state', () => {
      run(() => {
        const gru = new GRU(64, 128)
        const input = cpu.randn([10, 2, 64] as const)
        const h0 = cpu.randn([1, 2, 128] as const)

        const [output, hn] = gru.forward(input, h0)

        expect(output.shape).toEqual([10, 2, 128])
      })
    })
  })

  describe('parameters', () => {
    test('has trainable parameters (3x gates)', () => {
      const gru = new GRU(64, 128)
      const params = gru.parameters()

      // GRU has 3x parameters compared to RNN (r, z, n gates)
      expect(params.length).toBeGreaterThan(0)
    })
  })

  describe('toString', () => {
    test('returns descriptive string', () => {
      const gru = new GRU(256, 512, { numLayers: 2 })
      const str = gru.toString()

      expect(str).toContain('GRU')
      expect(str).toContain('256')
      expect(str).toContain('512')
    })
  })
})
