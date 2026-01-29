/**
 * Mock tensor implementation for testing without FFI dependencies
 */

export interface MockDtype {
  name: string;
  size: number;
}

/**
 * Helper to safely get array element with bounds checking
 */
function getAt<T>(arr: ArrayLike<T>, index: number): T {
  if (index < 0 || index >= arr.length) {
    throw new Error(`Index ${index} out of bounds for array of length ${arr.length}`);
  }
  return arr[index] as T;
}

export class MockTensor {
  readonly shape: readonly number[];
  readonly dtype: MockDtype;
  private _data: Float32Array;
  private _requiresGrad: boolean;
  private _grad: MockTensor | null;
  private _freed: boolean;
  private _escaped: boolean;

  constructor(
    data: Float32Array,
    shape: readonly number[],
    requiresGrad: boolean = false
  ) {
    this._data = data;
    this.shape = shape;
    this.dtype = { name: 'float32', size: 4 };
    this._requiresGrad = requiresGrad;
    this._grad = null;
    this._freed = false;
    this._escaped = false;
  }

  get requiresGrad(): boolean {
    return this._requiresGrad;
  }

  set requiresGrad(value: boolean) {
    this._requiresGrad = value;
  }

  get grad(): MockTensor | null {
    return this._grad;
  }

  get isFreed(): boolean {
    return this._freed;
  }

  get isEscaped(): boolean {
    return this._escaped;
  }

  numel(): number {
    return this.shape.reduce((acc, dim) => acc * dim, 1);
  }

  ndim(): number {
    return this.shape.length;
  }

  toArray(): number[] {
    this._checkNotFreed();
    return Array.from(this._data);
  }

  item(): number {
    this._checkNotFreed();
    if (this.numel() !== 1) {
      throw new Error('item() can only be called on tensors with one element');
    }
    return getAt(this._data, 0);
  }

  clone(): MockTensor {
    this._checkNotFreed();
    return new MockTensor(
      new Float32Array(this._data),
      [...this.shape],
      this._requiresGrad
    );
  }

  detach(): MockTensor {
    this._checkNotFreed();
    return new MockTensor(
      new Float32Array(this._data),
      [...this.shape],
      false
    );
  }

  add(other: MockTensor): MockTensor {
    this._checkNotFreed();
    this._checkShapeMatch(other);

    const result = new Float32Array(this._data.length);
    for (let i = 0; i < this._data.length; i++) {
      result[i] = getAt(this._data, i) + getAt(other._data, i);
    }

    return new MockTensor(
      result,
      [...this.shape],
      this._requiresGrad || other._requiresGrad
    );
  }

  sub(other: MockTensor): MockTensor {
    this._checkNotFreed();
    this._checkShapeMatch(other);

    const result = new Float32Array(this._data.length);
    for (let i = 0; i < this._data.length; i++) {
      result[i] = getAt(this._data, i) - getAt(other._data, i);
    }

    return new MockTensor(
      result,
      [...this.shape],
      this._requiresGrad || other._requiresGrad
    );
  }

  mul(other: MockTensor): MockTensor {
    this._checkNotFreed();
    this._checkShapeMatch(other);

    const result = new Float32Array(this._data.length);
    for (let i = 0; i < this._data.length; i++) {
      result[i] = getAt(this._data, i) * getAt(other._data, i);
    }

    return new MockTensor(
      result,
      [...this.shape],
      this._requiresGrad || other._requiresGrad
    );
  }

  mulScalar(scalar: number): MockTensor {
    this._checkNotFreed();

    const result = new Float32Array(this._data.length);
    for (let i = 0; i < this._data.length; i++) {
      result[i] = getAt(this._data, i) * scalar;
    }

    return new MockTensor(result, [...this.shape], this._requiresGrad);
  }

  relu(): MockTensor {
    this._checkNotFreed();

    const result = new Float32Array(this._data.length);
    for (let i = 0; i < this._data.length; i++) {
      result[i] = Math.max(0, getAt(this._data, i));
    }

    return new MockTensor(result, [...this.shape], this._requiresGrad);
  }

  sigmoid(): MockTensor {
    this._checkNotFreed();

    const result = new Float32Array(this._data.length);
    for (let i = 0; i < this._data.length; i++) {
      result[i] = 1 / (1 + Math.exp(-getAt(this._data, i)));
    }

    return new MockTensor(result, [...this.shape], this._requiresGrad);
  }

  tanh(): MockTensor {
    this._checkNotFreed();

    const result = new Float32Array(this._data.length);
    for (let i = 0; i < this._data.length; i++) {
      result[i] = Math.tanh(getAt(this._data, i));
    }

    return new MockTensor(result, [...this.shape], this._requiresGrad);
  }

  exp(): MockTensor {
    this._checkNotFreed();

    const result = new Float32Array(this._data.length);
    for (let i = 0; i < this._data.length; i++) {
      result[i] = Math.exp(getAt(this._data, i));
    }

    return new MockTensor(result, [...this.shape], this._requiresGrad);
  }

  log(): MockTensor {
    this._checkNotFreed();

    const result = new Float32Array(this._data.length);
    for (let i = 0; i < this._data.length; i++) {
      result[i] = Math.log(getAt(this._data, i));
    }

    return new MockTensor(result, [...this.shape], this._requiresGrad);
  }

  sqrt(): MockTensor {
    this._checkNotFreed();

    const result = new Float32Array(this._data.length);
    for (let i = 0; i < this._data.length; i++) {
      result[i] = Math.sqrt(getAt(this._data, i));
    }

    return new MockTensor(result, [...this.shape], this._requiresGrad);
  }

  abs(): MockTensor {
    this._checkNotFreed();

    const result = new Float32Array(this._data.length);
    for (let i = 0; i < this._data.length; i++) {
      result[i] = Math.abs(getAt(this._data, i));
    }

    return new MockTensor(result, [...this.shape], this._requiresGrad);
  }

  neg(): MockTensor {
    this._checkNotFreed();

    const result = new Float32Array(this._data.length);
    for (let i = 0; i < this._data.length; i++) {
      result[i] = -getAt(this._data, i);
    }

    return new MockTensor(result, [...this.shape], this._requiresGrad);
  }

  pow(exponent: number): MockTensor {
    this._checkNotFreed();

    const result = new Float32Array(this._data.length);
    for (let i = 0; i < this._data.length; i++) {
      result[i] = Math.pow(getAt(this._data, i), exponent);
    }

    return new MockTensor(result, [...this.shape], this._requiresGrad);
  }

  divScalar(scalar: number): MockTensor {
    this._checkNotFreed();

    const result = new Float32Array(this._data.length);
    for (let i = 0; i < this._data.length; i++) {
      result[i] = getAt(this._data, i) / scalar;
    }

    return new MockTensor(result, [...this.shape], this._requiresGrad);
  }

  softmax(dim: number = -1): MockTensor {
    this._checkNotFreed();

    if (this.shape.length === 1) {
      const data = this._data;
      let maxVal = -Infinity;
      for (let i = 0; i < data.length; i++) {
        maxVal = Math.max(maxVal, getAt(data, i));
      }
      const expVals = new Float32Array(data.length);
      let sum = 0;
      for (let i = 0; i < data.length; i++) {
        const val = Math.exp(getAt(data, i) - maxVal);
        expVals[i] = val;
        sum += val;
      }
      for (let i = 0; i < expVals.length; i++) {
        expVals[i] = expVals[i] / sum;
      }
      return new MockTensor(expVals, [...this.shape], this._requiresGrad);
    }

    if (this.shape.length === 2) {
      const rows = getAt(this.shape, 0);
      const cols = getAt(this.shape, 1);
      const axis = dim < 0 ? dim + 2 : dim;
      if (axis !== 0 && axis !== 1) {
        throw new Error(`softmax only supports dim 0 or 1 for 2D tensors in mock`);
      }
      const result = new Float32Array(this._data.length);
      if (axis === 1) {
        for (let row = 0; row < rows; row++) {
          let maxVal = -Infinity;
          for (let col = 0; col < cols; col++) {
            maxVal = Math.max(maxVal, getAt(this._data, row * cols + col));
          }
          let sum = 0;
          for (let col = 0; col < cols; col++) {
            const val = Math.exp(getAt(this._data, row * cols + col) - maxVal);
            result[row * cols + col] = val;
            sum += val;
          }
          for (let col = 0; col < cols; col++) {
            result[row * cols + col] = result[row * cols + col] / sum;
          }
        }
      } else {
        for (let col = 0; col < cols; col++) {
          let maxVal = -Infinity;
          for (let row = 0; row < rows; row++) {
            maxVal = Math.max(maxVal, getAt(this._data, row * cols + col));
          }
          let sum = 0;
          for (let row = 0; row < rows; row++) {
            const val = Math.exp(getAt(this._data, row * cols + col) - maxVal);
            result[row * cols + col] = val;
            sum += val;
          }
          for (let row = 0; row < rows; row++) {
            result[row * cols + col] = result[row * cols + col] / sum;
          }
        }
      }

      return new MockTensor(result, [...this.shape], this._requiresGrad);
    }

    throw new Error(`softmax not implemented for shape [${this.shape}]`);
  }

  logSoftmax(dim: number = -1): MockTensor {
    return this.softmax(dim).log();
  }

  maximum(other: MockTensor): MockTensor {
    this._checkNotFreed();
    this._checkShapeMatch(other);

    const result = new Float32Array(this._data.length);
    for (let i = 0; i < this._data.length; i++) {
      result[i] = Math.max(getAt(this._data, i), getAt(other._data, i));
    }

    return new MockTensor(result, [...this.shape], this._requiresGrad || other._requiresGrad);
  }

  sum(dim?: number): MockTensor {
    this._checkNotFreed();

    if (dim === undefined) {
      // Sum all elements
      let total = 0;
      for (let i = 0; i < this._data.length; i++) {
        total += getAt(this._data, i);
      }
      return new MockTensor(
        new Float32Array([total]),
        [],
        this._requiresGrad
      );
    }

    // Simple implementation for 2D tensors summing along a dimension
    if (this.shape.length === 2 && dim === 0) {
      const rows = getAt(this.shape, 0);
      const cols = getAt(this.shape, 1);
      const result = new Float32Array(cols);
      for (let col = 0; col < cols; col++) {
        let sum = 0;
        for (let row = 0; row < rows; row++) {
          sum += getAt(this._data, row * cols + col);
        }
        result[col] = sum;
      }
      return new MockTensor(result, [cols], this._requiresGrad);
    }

    if (this.shape.length === 2 && dim === 1) {
      const rows = getAt(this.shape, 0);
      const cols = getAt(this.shape, 1);
      const result = new Float32Array(rows);
      for (let row = 0; row < rows; row++) {
        let sum = 0;
        for (let col = 0; col < cols; col++) {
          sum += getAt(this._data, row * cols + col);
        }
        result[row] = sum;
      }
      return new MockTensor(result, [rows], this._requiresGrad);
    }

    throw new Error(`sum(dim=${dim}) not implemented for shape [${this.shape}]`);
  }

  mean(dim?: number): MockTensor {
    const sumTensor = this.sum(dim);
    const count = dim === undefined ? this.numel() : getAt(this.shape, dim);
    return sumTensor.mulScalar(1 / count);
  }

  addScalar(scalar: number): MockTensor {
    this._checkNotFreed();

    const result = new Float32Array(this._data.length);
    for (let i = 0; i < this._data.length; i++) {
      result[i] = getAt(this._data, i) + scalar;
    }

    return new MockTensor(result, [...this.shape], this._requiresGrad);
  }

  div(other: MockTensor): MockTensor {
    this._checkNotFreed();
    this._checkShapeMatch(other);

    const result = new Float32Array(this._data.length);
    for (let i = 0; i < this._data.length; i++) {
      result[i] = getAt(this._data, i) / getAt(other._data, i);
    }

    return new MockTensor(result, [...this.shape], this._requiresGrad || other._requiresGrad);
  }

  clamp(min: number, max: number): MockTensor {
    this._checkNotFreed();
    if (min > max) {
      throw new Error('min must be <= max');
    }

    const result = new Float32Array(this._data.length);
    for (let i = 0; i < this._data.length; i++) {
      const value = getAt(this._data, i);
      result[i] = Math.min(Math.max(value, min), max);
    }

    return new MockTensor(result, [...this.shape], this._requiresGrad);
  }

  clampMin(min: number): MockTensor {
    this._checkNotFreed();

    const result = new Float32Array(this._data.length);
    for (let i = 0; i < this._data.length; i++) {
      const value = getAt(this._data, i);
      result[i] = Math.max(value, min);
    }

    return new MockTensor(result, [...this.shape], this._requiresGrad);
  }

  clampMax(max: number): MockTensor {
    this._checkNotFreed();

    const result = new Float32Array(this._data.length);
    for (let i = 0; i < this._data.length; i++) {
      const value = getAt(this._data, i);
      result[i] = Math.min(value, max);
    }

    return new MockTensor(result, [...this.shape], this._requiresGrad);
  }

  sumDim(dim: number, keepdim: boolean = false): MockTensor {
    const summed = this.sum(dim);
    if (!keepdim) {
      return summed;
    }
    const axis = dim < 0 ? dim + this.shape.length : dim;
    const newShape = [...summed.shape];
    newShape.splice(axis, 0, 1);
    return new MockTensor(new Float32Array(summed.toArray()), newShape, this._requiresGrad);
  }

  dropout(p: number = 0.5, training: boolean = true): MockTensor {
    this._checkNotFreed();
    if (!training || p === 0) {
      return this.clone();
    }
    const result = new Float32Array(this._data.length);
    for (let i = 0; i < this._data.length; i++) {
      const keep = Math.random() >= p;
      result[i] = keep ? getAt(this._data, i) / (1 - p) : 0;
    }
    return new MockTensor(result, [...this.shape], this._requiresGrad);
  }

  transpose(_dim0: number, _dim1: number): MockTensor {
    this._checkNotFreed();

    if (this.shape.length !== 2) {
      throw new Error('transpose only implemented for 2D tensors in mock');
    }

    const rows = getAt(this.shape, 0);
    const cols = getAt(this.shape, 1);
    const result = new Float32Array(this._data.length);

    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        result[j * rows + i] = getAt(this._data, i * cols + j);
      }
    }

    return new MockTensor(result, [cols, rows], this._requiresGrad);
  }

  matmul(other: MockTensor): MockTensor {
    this._checkNotFreed();

    if (this.shape.length !== 2 || other.shape.length !== 2) {
      throw new Error('matmul only implemented for 2D tensors in mock');
    }

    const m = getAt(this.shape, 0);
    const k1 = getAt(this.shape, 1);
    const k2 = getAt(other.shape, 0);
    const n = getAt(other.shape, 1);

    if (k1 !== k2) {
      throw new Error(`Matrix dimensions don't match for matmul: [${m}, ${k1}] x [${k2}, ${n}]`);
    }

    const result = new Float32Array(m * n);

    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        let sum = 0;
        for (let k = 0; k < k1; k++) {
          const aIdx = i * k1 + k;
          const bIdx = k * n + j;
          sum += getAt(this._data, aIdx) * getAt(other._data, bIdx);
        }
        result[i * n + j] = sum;
      }
    }

    return new MockTensor(result, [m, n], this._requiresGrad || other._requiresGrad);
  }

  backward(): void {
    this._checkNotFreed();

    if (!this._requiresGrad) {
      throw new Error('Cannot call backward() on tensor that does not require gradients');
    }

    if (this.numel() !== 1) {
      throw new Error('backward() can only be called on scalar tensors');
    }

    // Simple mock: create gradient of ones with same shape
    this._grad = new MockTensor(
      new Float32Array(this.numel()).fill(1),
      [...this.shape],
      false
    );
  }

  zeroGrad(): void {
    this._checkNotFreed();

    if (this._grad) {
      this._grad = null;
    }
  }

  escape(): MockTensor {
    this._escaped = true;
    return this;
  }

  free(): void {
    if (!this._freed) {
      this._freed = true;
      this._grad = null;
    }
  }

  private _checkNotFreed(): void {
    if (this._freed) {
      throw new Error('Cannot use freed tensor');
    }
  }

  private _checkShapeMatch(other: MockTensor): void {
    if (this.shape.length !== other.shape.length) {
      throw new Error(
        `Shape mismatch: [${this.shape}] vs [${other.shape}]`
      );
    }
    for (let i = 0; i < this.shape.length; i++) {
      if (this.shape[i] !== other.shape[i]) {
        throw new Error(
          `Shape mismatch: [${this.shape}] vs [${other.shape}]`
        );
      }
    }
  }
}

/**
 * Mock tensor factory functions
 */
export const mockTensorFactories = {
  zeros(shape: readonly number[], requiresGrad: boolean = false): MockTensor {
    const numel = shape.reduce((acc, dim) => acc * dim, 1);
    return new MockTensor(new Float32Array(numel), shape, requiresGrad);
  },

  ones(shape: readonly number[], requiresGrad: boolean = false): MockTensor {
    const numel = shape.reduce((acc, dim) => acc * dim, 1);
    const data = new Float32Array(numel).fill(1);
    return new MockTensor(data, shape, requiresGrad);
  },

  randn(shape: readonly number[], requiresGrad: boolean = false): MockTensor {
    const numel = shape.reduce((acc, dim) => acc * dim, 1);
    const data = new Float32Array(numel);

    // Box-Muller transform for normal distribution
    for (let i = 0; i < numel; i += 2) {
      const u1 = Math.random();
      const u2 = Math.random();
      const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      const z1 = Math.sqrt(-2 * Math.log(u1)) * Math.sin(2 * Math.PI * u2);
      data[i] = z0;
      if (i + 1 < numel) {
        data[i + 1] = z1;
      }
    }

    return new MockTensor(data, shape, requiresGrad);
  },

  fromArray(
    data: number[],
    shape: readonly number[],
    requiresGrad: boolean = false
  ): MockTensor {
    const expectedNumel = shape.reduce((acc, dim) => acc * dim, 1);
    if (data.length !== expectedNumel) {
      throw new Error(
        `Data length ${data.length} does not match shape [${shape}] (expected ${expectedNumel} elements)`
      );
    }
    return new MockTensor(new Float32Array(data), shape, requiresGrad);
  },
};
