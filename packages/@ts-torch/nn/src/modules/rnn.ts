/**
 * Recurrent Neural Network modules
 *
 * Implements RNN, LSTM, and GRU layers for sequence modeling.
 */

import { Module, Parameter, type Tensor, type float32 } from '../module.js'
import { device, cat, type DType, type DeviceType, type Shape } from '@ts-torch/core'

// CPU device for weight initialization
const cpu = device.cpu()

/**
 * Base RNN options interface
 */
export interface RNNBaseOptions<D extends DType<string> = float32> {
  /**
   * Number of recurrent layers (default: 1)
   */
  numLayers?: number

  /**
   * Whether to include bias terms (default: true)
   */
  bias?: boolean

  /**
   * Whether inputs are (batch, seq, features) vs (seq, batch, features) (default: false)
   */
  batchFirst?: boolean

  /**
   * Dropout probability between layers (default: 0.0)
   * Only applied if numLayers > 1
   */
  dropout?: number

  /**
   * Whether to use bidirectional RNN (default: false)
   */
  bidirectional?: boolean

  /**
   * Data type for weights (default: float32)
   */
  dtype?: D
}

/**
 * RNN-specific options
 */
export interface RNNOptions<D extends DType<string> = float32> extends RNNBaseOptions<D> {
  /**
   * Non-linearity to use: 'tanh' or 'relu' (default: 'tanh')
   */
  nonlinearity?: 'tanh' | 'relu'
}

/**
 * Elman RNN layer
 *
 * Applies a multi-layer Elman RNN with tanh or ReLU non-linearity to an input sequence.
 *
 * h_t = tanh(W_ih * x_t + b_ih + W_hh * h_(t-1) + b_hh)
 *
 * @template InputSize - Size of input features
 * @template HiddenSize - Size of hidden state
 * @template D - Data type (default: float32)
 * @template Dev - Device type (default: 'cpu')
 *
 * @example
 * ```ts
 * const rnn = new RNN(256, 512, { numLayers: 2, batchFirst: true });
 *
 * // Input: [batch, seq, input_size]
 * const input = cpu.randn([32, 100, 256]);
 *
 * // Output: [batch, seq, hidden_size], hidden: [numLayers, batch, hidden_size]
 * const [output, hidden] = rnn.forward(input);
 * ```
 */
export class RNN<
  InputSize extends number = number,
  HiddenSize extends number = number,
  D extends DType<string> = float32,
  Dev extends DeviceType = 'cpu',
> extends Module<Shape, Shape, D, Dev> {
  readonly inputSize: InputSize
  readonly hiddenSize: HiddenSize
  readonly numLayers: number
  readonly bias: boolean
  readonly batchFirst: boolean
  readonly dropout: number
  readonly bidirectional: boolean
  readonly nonlinearity: 'tanh' | 'relu'

  private numDirections: number

  // Weight parameters for each layer and direction
  // Layer l has weight_ih_l[k] and weight_hh_l[k]
  // where k = 0 for forward, k = 1 for reverse (if bidirectional)

  /**
   * Create a new RNN layer
   *
   * @param inputSize - Size of input features
   * @param hiddenSize - Size of hidden state
   * @param options - Configuration options
   */
  constructor(inputSize: InputSize, hiddenSize: HiddenSize, options: RNNOptions<D> = {}) {
    super()

    if (inputSize <= 0) throw new Error(`inputSize must be positive, got ${inputSize}`)
    if (hiddenSize <= 0) throw new Error(`hiddenSize must be positive, got ${hiddenSize}`)

    this.inputSize = inputSize
    this.hiddenSize = hiddenSize
    this.numLayers = options.numLayers ?? 1
    this.bias = options.bias ?? true
    this.batchFirst = options.batchFirst ?? false
    this.dropout = options.dropout ?? 0.0
    this.bidirectional = options.bidirectional ?? false
    this.nonlinearity = options.nonlinearity ?? 'tanh'
    this.numDirections = this.bidirectional ? 2 : 1

    // Initialize weights for each layer
    for (let layer = 0; layer < this.numLayers; layer++) {
      for (let direction = 0; direction < this.numDirections; direction++) {
        const suffix = this.numDirections === 1 ? `l${layer}` : `l${layer}_${direction === 0 ? '' : 'reverse'}`

        // Input-hidden weights
        const realInputSize = layer === 0 ? inputSize : hiddenSize * this.numDirections
        const weightIh = cpu.randn([hiddenSize, realInputSize]) as unknown as Tensor<Shape, D, 'cpu'>
        const k = Math.sqrt(1 / hiddenSize)
        const scaledWeightIh = weightIh.mulScalar(k) as Tensor<Shape, D, 'cpu'>
        scaledWeightIh.escape()
        const paramIh = new Parameter(scaledWeightIh, true)
        this.registerParameter(`weight_ih_${suffix}`, paramIh as any)

        // Hidden-hidden weights
        const weightHh = cpu.randn([hiddenSize, hiddenSize]) as unknown as Tensor<Shape, D, 'cpu'>
        const scaledWeightHh = weightHh.mulScalar(k) as Tensor<Shape, D, 'cpu'>
        scaledWeightHh.escape()
        const paramHh = new Parameter(scaledWeightHh, true)
        this.registerParameter(`weight_hh_${suffix}`, paramHh as any)

        // Biases
        if (this.bias) {
          const biasIh = cpu.zeros([hiddenSize]) as unknown as Tensor<Shape, D, 'cpu'>
          biasIh.escape()
          const paramBiasIh = new Parameter(biasIh, true)
          this.registerParameter(`bias_ih_${suffix}`, paramBiasIh as any)

          const biasHh = cpu.zeros([hiddenSize]) as unknown as Tensor<Shape, D, 'cpu'>
          biasHh.escape()
          const paramBiasHh = new Parameter(biasHh, true)
          this.registerParameter(`bias_hh_${suffix}`, paramBiasHh as any)
        }
      }
    }
  }

  /**
   * Forward pass through RNN
   *
   * @param input - Input tensor [seq, batch, input_size] or [batch, seq, input_size] if batchFirst
   * @param h0 - Initial hidden state [numLayers * numDirections, batch, hiddenSize]
   * @returns Tuple of [output, hidden]
   *   - output: [seq, batch, hiddenSize * numDirections] or [batch, seq, ...] if batchFirst
   *   - hidden: [numLayers * numDirections, batch, hiddenSize]
   */
  forward(
    input: Tensor<Shape, D, Dev>,
    h0?: Tensor<Shape, D, Dev>,
  ): [Tensor<Shape, D, Dev>, Tensor<Shape, D, Dev>] {
    const inputShape = input.shape as readonly number[]
    let seqLen: number
    let batchSize: number

    if (this.batchFirst) {
      batchSize = inputShape[0]
      seqLen = inputShape[1]
    } else {
      seqLen = inputShape[0]
      batchSize = inputShape[1]
    }

    // Initialize hidden state if not provided
    let hidden: Tensor<Shape, D, Dev>
    if (h0) {
      hidden = h0
    } else {
      hidden = cpu.zeros([
        this.numLayers * this.numDirections,
        batchSize,
        this.hiddenSize,
      ]) as unknown as Tensor<Shape, D, Dev>
    }

    // Convert to [seq, batch, features] for processing
    let x = input
    if (this.batchFirst) {
      x = x.transpose(0, 1) as Tensor<Shape, D, Dev>
    }

    // Process through layers
    let output = x
    const newHiddenStates: Tensor<Shape, D, Dev>[] = []

    for (let layer = 0; layer < this.numLayers; layer++) {
      const layerOutputs: Tensor<Shape, D, Dev>[] = []

      // Forward direction
      let hForward = hidden.narrow(0, layer * this.numDirections, 1).squeeze(0) as Tensor<Shape, D, Dev>

      const suffix = this.numDirections === 1 ? `l${layer}` : `l${layer}_`
      const weightIhF = (this as any)._parameters.get(`weight_ih_${suffix}`)?.data as Tensor<
        Shape,
        D,
        Dev
      >
      const weightHhF = (this as any)._parameters.get(`weight_hh_${suffix}`)?.data as Tensor<
        Shape,
        D,
        Dev
      >
      const biasIhF = this.bias
        ? ((this as any)._parameters.get(`bias_ih_${suffix}`)?.data as Tensor<Shape, D, Dev>)
        : null
      const biasHhF = this.bias
        ? ((this as any)._parameters.get(`bias_hh_${suffix}`)?.data as Tensor<Shape, D, Dev>)
        : null

      for (let t = 0; t < seqLen; t++) {
        const xt = output.narrow(0, t, 1).squeeze(0) as Tensor<Shape, D, Dev>
        hForward = this.rnnCell(xt, hForward, weightIhF, weightHhF, biasIhF, biasHhF)
        layerOutputs.push(hForward.reshape([1, batchSize, this.hiddenSize]) as Tensor<Shape, D, Dev>)
      }

      newHiddenStates.push(hForward.reshape([1, batchSize, this.hiddenSize]) as Tensor<Shape, D, Dev>)

      // Concatenate outputs along sequence dimension
      // Concatenate layer outputs along sequence dimension
      output = cat(layerOutputs, 0) as Tensor<Shape, D, Dev>

      // Apply dropout between layers (except last)
      if (this.dropout > 0 && layer < this.numLayers - 1 && this._training) {
        output = (output as any).dropout(this.dropout, true) as Tensor<Shape, D, Dev>
      }
    }

    // Stack hidden states
    const newHidden = cat(newHiddenStates, 0) as Tensor<Shape, D, Dev>

    // Convert back to batchFirst if needed
    if (this.batchFirst) {
      output = output.transpose(0, 1) as Tensor<Shape, D, Dev>
    }

    return [output, newHidden]
  }

  /**
   * Single RNN cell computation
   */
  private rnnCell(
    input: Tensor<Shape, D, Dev>,
    hidden: Tensor<Shape, D, Dev>,
    weightIh: Tensor<Shape, D, Dev>,
    weightHh: Tensor<Shape, D, Dev>,
    biasIh: Tensor<Shape, D, Dev> | null,
    biasHh: Tensor<Shape, D, Dev> | null,
  ): Tensor<Shape, D, Dev> {
    // h_t = act(x_t @ W_ih^T + h_(t-1) @ W_hh^T + b_ih + b_hh)
    let gate = input.matmul(weightIh.transpose(0, 1) as any) as Tensor<Shape, D, Dev>
    gate = gate.add(hidden.matmul(weightHh.transpose(0, 1) as any) as any) as Tensor<Shape, D, Dev>

    if (biasIh) {
      gate = gate.add(biasIh as any) as Tensor<Shape, D, Dev>
    }
    if (biasHh) {
      gate = gate.add(biasHh as any) as Tensor<Shape, D, Dev>
    }

    // Apply nonlinearity
    if (this.nonlinearity === 'tanh') {
      return gate.tanh() as Tensor<Shape, D, Dev>
    } else {
      return gate.relu() as Tensor<Shape, D, Dev>
    }
  }

  protected override _outputShapeHint(): string {
    return `[*, *, ${this.hiddenSize * this.numDirections}]`
  }

  override toString(): string {
    return `RNN(${this.inputSize}, ${this.hiddenSize}, num_layers=${this.numLayers}, nonlinearity='${this.nonlinearity}', bidirectional=${this.bidirectional})`
  }
}

/**
 * LSTM (Long Short-Term Memory) layer
 *
 * Applies a multi-layer LSTM to an input sequence.
 *
 * LSTM equations:
 * i_t = σ(W_ii x_t + b_ii + W_hi h_(t-1) + b_hi)  // input gate
 * f_t = σ(W_if x_t + b_if + W_hf h_(t-1) + b_hf)  // forget gate
 * g_t = tanh(W_ig x_t + b_ig + W_hg h_(t-1) + b_hg) // cell gate
 * o_t = σ(W_io x_t + b_io + W_ho h_(t-1) + b_ho)  // output gate
 * c_t = f_t * c_(t-1) + i_t * g_t                  // cell state
 * h_t = o_t * tanh(c_t)                            // hidden state
 *
 * @example
 * ```ts
 * const lstm = new LSTM(256, 512, { numLayers: 2, batchFirst: true });
 *
 * const input = cpu.randn([32, 100, 256]);
 * const [output, [h_n, c_n]] = lstm.forward(input);
 * ```
 */
export class LSTM<
  InputSize extends number = number,
  HiddenSize extends number = number,
  D extends DType<string> = float32,
  Dev extends DeviceType = 'cpu',
> extends Module<Shape, Shape, D, Dev> {
  readonly inputSize: InputSize
  readonly hiddenSize: HiddenSize
  readonly numLayers: number
  readonly bias: boolean
  readonly batchFirst: boolean
  readonly dropout: number
  readonly bidirectional: boolean

  private numDirections: number

  constructor(inputSize: InputSize, hiddenSize: HiddenSize, options: RNNBaseOptions<D> = {}) {
    super()

    if (inputSize <= 0) throw new Error(`inputSize must be positive, got ${inputSize}`)
    if (hiddenSize <= 0) throw new Error(`hiddenSize must be positive, got ${hiddenSize}`)

    this.inputSize = inputSize
    this.hiddenSize = hiddenSize
    this.numLayers = options.numLayers ?? 1
    this.bias = options.bias ?? true
    this.batchFirst = options.batchFirst ?? false
    this.dropout = options.dropout ?? 0.0
    this.bidirectional = options.bidirectional ?? false
    this.numDirections = this.bidirectional ? 2 : 1

    // Initialize weights for each layer
    // LSTM has 4x the parameters (i, f, g, o gates)
    for (let layer = 0; layer < this.numLayers; layer++) {
      for (let direction = 0; direction < this.numDirections; direction++) {
        const suffix =
          this.numDirections === 1 ? `l${layer}` : `l${layer}_${direction === 0 ? '' : 'reverse'}`

        const realInputSize = layer === 0 ? inputSize : hiddenSize * this.numDirections
        const gateSize = 4 * hiddenSize // i, f, g, o gates

        // Input-hidden weights: [4*hidden, input]
        const weightIh = cpu.randn([gateSize, realInputSize]) as unknown as Tensor<Shape, D, 'cpu'>
        const k = Math.sqrt(1 / hiddenSize)
        const scaledWeightIh = weightIh.mulScalar(k) as Tensor<Shape, D, 'cpu'>
        scaledWeightIh.escape()
        this.registerParameter(`weight_ih_${suffix}`, new Parameter(scaledWeightIh, true) as any)

        // Hidden-hidden weights: [4*hidden, hidden]
        const weightHh = cpu.randn([gateSize, hiddenSize]) as unknown as Tensor<Shape, D, 'cpu'>
        const scaledWeightHh = weightHh.mulScalar(k) as Tensor<Shape, D, 'cpu'>
        scaledWeightHh.escape()
        this.registerParameter(`weight_hh_${suffix}`, new Parameter(scaledWeightHh, true) as any)

        // Biases
        if (this.bias) {
          const biasIh = cpu.zeros([gateSize]) as unknown as Tensor<Shape, D, 'cpu'>
          biasIh.escape()
          this.registerParameter(`bias_ih_${suffix}`, new Parameter(biasIh, true) as any)

          // Initialize forget gate bias to 1 (helps with gradient flow)
          const biasHh = cpu.zeros([gateSize]) as unknown as Tensor<Shape, D, 'cpu'>
          biasHh.escape()
          this.registerParameter(`bias_hh_${suffix}`, new Parameter(biasHh, true) as any)
        }
      }
    }
  }

  /**
   * Forward pass through LSTM
   *
   * @param input - Input tensor [seq, batch, input_size] or [batch, seq, input_size] if batchFirst
   * @param hx - Optional tuple of (h_0, c_0) initial states
   * @returns Tuple of [output, (h_n, c_n)]
   */
  forward(
    input: Tensor<Shape, D, Dev>,
    hx?: [Tensor<Shape, D, Dev>, Tensor<Shape, D, Dev>],
  ): [Tensor<Shape, D, Dev>, [Tensor<Shape, D, Dev>, Tensor<Shape, D, Dev>]] {
    const inputShape = input.shape as readonly number[]
    let seqLen: number
    let batchSize: number

    if (this.batchFirst) {
      batchSize = inputShape[0]
      seqLen = inputShape[1]
    } else {
      seqLen = inputShape[0]
      batchSize = inputShape[1]
    }

    // Initialize hidden and cell states
    let h: Tensor<Shape, D, Dev>
    let c: Tensor<Shape, D, Dev>

    if (hx) {
      ;[h, c] = hx
    } else {
      h = cpu.zeros([
        this.numLayers * this.numDirections,
        batchSize,
        this.hiddenSize,
      ]) as unknown as Tensor<Shape, D, Dev>
      c = cpu.zeros([
        this.numLayers * this.numDirections,
        batchSize,
        this.hiddenSize,
      ]) as unknown as Tensor<Shape, D, Dev>
    }

    // Convert to [seq, batch, features]
    let x = input
    if (this.batchFirst) {
      x = x.transpose(0, 1) as Tensor<Shape, D, Dev>
    }

    let output = x
    const newHiddens: Tensor<Shape, D, Dev>[] = []
    const newCells: Tensor<Shape, D, Dev>[] = []

    for (let layer = 0; layer < this.numLayers; layer++) {
      const layerOutputs: Tensor<Shape, D, Dev>[] = []

      let hLayer = h.narrow(0, layer * this.numDirections, 1).squeeze(0) as Tensor<Shape, D, Dev>
      let cLayer = c.narrow(0, layer * this.numDirections, 1).squeeze(0) as Tensor<Shape, D, Dev>

      const suffix = this.numDirections === 1 ? `l${layer}` : `l${layer}_`
      const weightIh = (this as any)._parameters.get(`weight_ih_${suffix}`)?.data as Tensor<
        Shape,
        D,
        Dev
      >
      const weightHh = (this as any)._parameters.get(`weight_hh_${suffix}`)?.data as Tensor<
        Shape,
        D,
        Dev
      >
      const biasIh = this.bias
        ? ((this as any)._parameters.get(`bias_ih_${suffix}`)?.data as Tensor<Shape, D, Dev>)
        : null
      const biasHh = this.bias
        ? ((this as any)._parameters.get(`bias_hh_${suffix}`)?.data as Tensor<Shape, D, Dev>)
        : null

      for (let t = 0; t < seqLen; t++) {
        const xt = output.narrow(0, t, 1).squeeze(0) as Tensor<Shape, D, Dev>
        ;[hLayer, cLayer] = this.lstmCell(xt, hLayer, cLayer, weightIh, weightHh, biasIh, biasHh)
        layerOutputs.push(hLayer.reshape([1, batchSize, this.hiddenSize]) as Tensor<Shape, D, Dev>)
      }

      newHiddens.push(hLayer.reshape([1, batchSize, this.hiddenSize]) as Tensor<Shape, D, Dev>)
      newCells.push(cLayer.reshape([1, batchSize, this.hiddenSize]) as Tensor<Shape, D, Dev>)

      // Concatenate outputs
      output = cat(layerOutputs, 0) as Tensor<Shape, D, Dev>

      // Dropout between layers
      if (this.dropout > 0 && layer < this.numLayers - 1 && this._training) {
        output = (output as any).dropout(this.dropout, true) as Tensor<Shape, D, Dev>
      }
    }

    // Stack hidden states
    const hn = cat(newHiddens, 0) as Tensor<Shape, D, Dev>
    const cn = cat(newCells, 0) as Tensor<Shape, D, Dev>

    if (this.batchFirst) {
      output = output.transpose(0, 1) as Tensor<Shape, D, Dev>
    }

    return [output, [hn, cn]]
  }

  /**
   * Single LSTM cell computation
   */
  private lstmCell(
    input: Tensor<Shape, D, Dev>,
    hidden: Tensor<Shape, D, Dev>,
    cell: Tensor<Shape, D, Dev>,
    weightIh: Tensor<Shape, D, Dev>,
    weightHh: Tensor<Shape, D, Dev>,
    biasIh: Tensor<Shape, D, Dev> | null,
    biasHh: Tensor<Shape, D, Dev> | null,
  ): [Tensor<Shape, D, Dev>, Tensor<Shape, D, Dev>] {
    // Compute all gates at once
    let gates = input.matmul(weightIh.transpose(0, 1) as any) as Tensor<Shape, D, Dev>
    gates = gates.add(hidden.matmul(weightHh.transpose(0, 1) as any) as any) as Tensor<Shape, D, Dev>

    if (biasIh) gates = gates.add(biasIh as any) as Tensor<Shape, D, Dev>
    if (biasHh) gates = gates.add(biasHh as any) as Tensor<Shape, D, Dev>

    // Split into i, f, g, o gates
    const gatesShape = gates.shape as readonly number[]
    const chunkSize = gatesShape[1] / 4

    const i = gates.narrow(1, 0, chunkSize).sigmoid() as Tensor<Shape, D, Dev> // input gate
    const f = gates.narrow(1, chunkSize, chunkSize).sigmoid() as Tensor<Shape, D, Dev> // forget gate
    const g = gates.narrow(1, chunkSize * 2, chunkSize).tanh() as Tensor<Shape, D, Dev> // cell gate
    const o = gates.narrow(1, chunkSize * 3, chunkSize).sigmoid() as Tensor<Shape, D, Dev> // output gate

    // Update cell and hidden state
    const newCell = f.mul(cell as any).add(i.mul(g as any) as any) as Tensor<Shape, D, Dev>
    const newHidden = o.mul(newCell.tanh() as any) as Tensor<Shape, D, Dev>

    return [newHidden, newCell]
  }

  protected override _outputShapeHint(): string {
    return `[*, *, ${this.hiddenSize * this.numDirections}]`
  }

  override toString(): string {
    return `LSTM(${this.inputSize}, ${this.hiddenSize}, num_layers=${this.numLayers}, bidirectional=${this.bidirectional})`
  }
}

/**
 * GRU (Gated Recurrent Unit) layer
 *
 * Applies a multi-layer GRU to an input sequence.
 *
 * GRU equations:
 * r_t = σ(W_ir x_t + b_ir + W_hr h_(t-1) + b_hr)  // reset gate
 * z_t = σ(W_iz x_t + b_iz + W_hz h_(t-1) + b_hz)  // update gate
 * n_t = tanh(W_in x_t + b_in + r_t * (W_hn h_(t-1) + b_hn)) // new gate
 * h_t = (1 - z_t) * n_t + z_t * h_(t-1)           // hidden state
 *
 * @example
 * ```ts
 * const gru = new GRU(256, 512, { numLayers: 2, batchFirst: true });
 *
 * const input = cpu.randn([32, 100, 256]);
 * const [output, h_n] = gru.forward(input);
 * ```
 */
export class GRU<
  InputSize extends number = number,
  HiddenSize extends number = number,
  D extends DType<string> = float32,
  Dev extends DeviceType = 'cpu',
> extends Module<Shape, Shape, D, Dev> {
  readonly inputSize: InputSize
  readonly hiddenSize: HiddenSize
  readonly numLayers: number
  readonly bias: boolean
  readonly batchFirst: boolean
  readonly dropout: number
  readonly bidirectional: boolean

  private numDirections: number

  constructor(inputSize: InputSize, hiddenSize: HiddenSize, options: RNNBaseOptions<D> = {}) {
    super()

    if (inputSize <= 0) throw new Error(`inputSize must be positive, got ${inputSize}`)
    if (hiddenSize <= 0) throw new Error(`hiddenSize must be positive, got ${hiddenSize}`)

    this.inputSize = inputSize
    this.hiddenSize = hiddenSize
    this.numLayers = options.numLayers ?? 1
    this.bias = options.bias ?? true
    this.batchFirst = options.batchFirst ?? false
    this.dropout = options.dropout ?? 0.0
    this.bidirectional = options.bidirectional ?? false
    this.numDirections = this.bidirectional ? 2 : 1

    // Initialize weights for each layer
    // GRU has 3x the parameters (r, z, n gates)
    for (let layer = 0; layer < this.numLayers; layer++) {
      for (let direction = 0; direction < this.numDirections; direction++) {
        const suffix =
          this.numDirections === 1 ? `l${layer}` : `l${layer}_${direction === 0 ? '' : 'reverse'}`

        const realInputSize = layer === 0 ? inputSize : hiddenSize * this.numDirections
        const gateSize = 3 * hiddenSize // r, z, n gates

        // Input-hidden weights: [3*hidden, input]
        const weightIh = cpu.randn([gateSize, realInputSize]) as unknown as Tensor<Shape, D, 'cpu'>
        const k = Math.sqrt(1 / hiddenSize)
        const scaledWeightIh = weightIh.mulScalar(k) as Tensor<Shape, D, 'cpu'>
        scaledWeightIh.escape()
        this.registerParameter(`weight_ih_${suffix}`, new Parameter(scaledWeightIh, true) as any)

        // Hidden-hidden weights: [3*hidden, hidden]
        const weightHh = cpu.randn([gateSize, hiddenSize]) as unknown as Tensor<Shape, D, 'cpu'>
        const scaledWeightHh = weightHh.mulScalar(k) as Tensor<Shape, D, 'cpu'>
        scaledWeightHh.escape()
        this.registerParameter(`weight_hh_${suffix}`, new Parameter(scaledWeightHh, true) as any)

        // Biases
        if (this.bias) {
          const biasIh = cpu.zeros([gateSize]) as unknown as Tensor<Shape, D, 'cpu'>
          biasIh.escape()
          this.registerParameter(`bias_ih_${suffix}`, new Parameter(biasIh, true) as any)

          const biasHh = cpu.zeros([gateSize]) as unknown as Tensor<Shape, D, 'cpu'>
          biasHh.escape()
          this.registerParameter(`bias_hh_${suffix}`, new Parameter(biasHh, true) as any)
        }
      }
    }
  }

  /**
   * Forward pass through GRU
   *
   * @param input - Input tensor
   * @param h0 - Initial hidden state
   * @returns Tuple of [output, h_n]
   */
  forward(
    input: Tensor<Shape, D, Dev>,
    h0?: Tensor<Shape, D, Dev>,
  ): [Tensor<Shape, D, Dev>, Tensor<Shape, D, Dev>] {
    const inputShape = input.shape as readonly number[]
    let seqLen: number
    let batchSize: number

    if (this.batchFirst) {
      batchSize = inputShape[0]
      seqLen = inputShape[1]
    } else {
      seqLen = inputShape[0]
      batchSize = inputShape[1]
    }

    // Initialize hidden state
    let h: Tensor<Shape, D, Dev>
    if (h0) {
      h = h0
    } else {
      h = cpu.zeros([
        this.numLayers * this.numDirections,
        batchSize,
        this.hiddenSize,
      ]) as unknown as Tensor<Shape, D, Dev>
    }

    let x = input
    if (this.batchFirst) {
      x = x.transpose(0, 1) as Tensor<Shape, D, Dev>
    }

    let output = x
    const newHiddens: Tensor<Shape, D, Dev>[] = []

    for (let layer = 0; layer < this.numLayers; layer++) {
      const layerOutputs: Tensor<Shape, D, Dev>[] = []

      let hLayer = h.narrow(0, layer * this.numDirections, 1).squeeze(0) as Tensor<Shape, D, Dev>

      const suffix = this.numDirections === 1 ? `l${layer}` : `l${layer}_`
      const weightIh = (this as any)._parameters.get(`weight_ih_${suffix}`)?.data as Tensor<
        Shape,
        D,
        Dev
      >
      const weightHh = (this as any)._parameters.get(`weight_hh_${suffix}`)?.data as Tensor<
        Shape,
        D,
        Dev
      >
      const biasIh = this.bias
        ? ((this as any)._parameters.get(`bias_ih_${suffix}`)?.data as Tensor<Shape, D, Dev>)
        : null
      const biasHh = this.bias
        ? ((this as any)._parameters.get(`bias_hh_${suffix}`)?.data as Tensor<Shape, D, Dev>)
        : null

      for (let t = 0; t < seqLen; t++) {
        const xt = output.narrow(0, t, 1).squeeze(0) as Tensor<Shape, D, Dev>
        hLayer = this.gruCell(xt, hLayer, weightIh, weightHh, biasIh, biasHh)
        layerOutputs.push(hLayer.reshape([1, batchSize, this.hiddenSize]) as Tensor<Shape, D, Dev>)
      }

      newHiddens.push(hLayer.reshape([1, batchSize, this.hiddenSize]) as Tensor<Shape, D, Dev>)

      output = cat(layerOutputs, 0) as Tensor<Shape, D, Dev>

      if (this.dropout > 0 && layer < this.numLayers - 1 && this._training) {
        output = (output as any).dropout(this.dropout, true) as Tensor<Shape, D, Dev>
      }
    }

    const hn = cat(newHiddens, 0) as Tensor<Shape, D, Dev>

    if (this.batchFirst) {
      output = output.transpose(0, 1) as Tensor<Shape, D, Dev>
    }

    return [output, hn]
  }

  /**
   * Single GRU cell computation
   */
  private gruCell(
    input: Tensor<Shape, D, Dev>,
    hidden: Tensor<Shape, D, Dev>,
    weightIh: Tensor<Shape, D, Dev>,
    weightHh: Tensor<Shape, D, Dev>,
    biasIh: Tensor<Shape, D, Dev> | null,
    biasHh: Tensor<Shape, D, Dev> | null,
  ): Tensor<Shape, D, Dev> {
    // Compute gates
    let gatesIh = input.matmul(weightIh.transpose(0, 1) as any) as Tensor<Shape, D, Dev>
    let gatesHh = hidden.matmul(weightHh.transpose(0, 1) as any) as Tensor<Shape, D, Dev>

    if (biasIh) gatesIh = gatesIh.add(biasIh as any) as Tensor<Shape, D, Dev>
    if (biasHh) gatesHh = gatesHh.add(biasHh as any) as Tensor<Shape, D, Dev>

    const gatesShape = gatesIh.shape as readonly number[]
    const chunkSize = gatesShape[1] / 3

    // Reset and update gates
    const rIh = gatesIh.narrow(1, 0, chunkSize) as Tensor<Shape, D, Dev>
    const zIh = gatesIh.narrow(1, chunkSize, chunkSize) as Tensor<Shape, D, Dev>
    const nIh = gatesIh.narrow(1, chunkSize * 2, chunkSize) as Tensor<Shape, D, Dev>

    const rHh = gatesHh.narrow(1, 0, chunkSize) as Tensor<Shape, D, Dev>
    const zHh = gatesHh.narrow(1, chunkSize, chunkSize) as Tensor<Shape, D, Dev>
    const nHh = gatesHh.narrow(1, chunkSize * 2, chunkSize) as Tensor<Shape, D, Dev>

    const r = rIh.add(rHh as any).sigmoid() as Tensor<Shape, D, Dev> // reset gate
    const z = zIh.add(zHh as any).sigmoid() as Tensor<Shape, D, Dev> // update gate
    const n = nIh.add(r.mul(nHh as any) as any).tanh() as Tensor<Shape, D, Dev> // new gate

    // h_t = (1 - z) * n + z * h_(t-1)
    const oneMinusZ = z.mulScalar(-1).addScalar(1) as Tensor<Shape, D, Dev>
    const newHidden = oneMinusZ.mul(n as any).add(z.mul(hidden as any) as any) as Tensor<Shape, D, Dev>

    return newHidden
  }

  protected override _outputShapeHint(): string {
    return `[*, *, ${this.hiddenSize * this.numDirections}]`
  }

  override toString(): string {
    return `GRU(${this.inputSize}, ${this.hiddenSize}, num_layers=${this.numLayers}, bidirectional=${this.bidirectional})`
  }
}
