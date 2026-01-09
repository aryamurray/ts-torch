/**
 * Mock tensor implementation for testing without FFI dependencies
 */

export interface MockDtype {
  name: string;
  size: number;
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
    return this._data[0];
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
      result[i] = this._data[i] + other._data[i];
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
      result[i] = this._data[i] - other._data[i];
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
      result[i] = this._data[i] * other._data[i];
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
      result[i] = this._data[i] * scalar;
    }

    return new MockTensor(result, [...this.shape], this._requiresGrad);
  }

  relu(): MockTensor {
    this._checkNotFreed();

    const result = new Float32Array(this._data.length);
    for (let i = 0; i < this._data.length; i++) {
      result[i] = Math.max(0, this._data[i]);
    }

    return new MockTensor(result, [...this.shape], this._requiresGrad);
  }

  sigmoid(): MockTensor {
    this._checkNotFreed();

    const result = new Float32Array(this._data.length);
    for (let i = 0; i < this._data.length; i++) {
      result[i] = 1 / (1 + Math.exp(-this._data[i]));
    }

    return new MockTensor(result, [...this.shape], this._requiresGrad);
  }

  sum(dim?: number): MockTensor {
    this._checkNotFreed();

    if (dim === undefined) {
      // Sum all elements
      let total = 0;
      for (let i = 0; i < this._data.length; i++) {
        total += this._data[i];
      }
      return new MockTensor(
        new Float32Array([total]),
        [],
        this._requiresGrad
      );
    }

    // Simple implementation for 2D tensors summing along a dimension
    if (this.shape.length === 2 && dim === 0) {
      const [rows, cols] = this.shape;
      const result = new Float32Array(cols);
      for (let col = 0; col < cols; col++) {
        let sum = 0;
        for (let row = 0; row < rows; row++) {
          sum += this._data[row * cols + col];
        }
        result[col] = sum;
      }
      return new MockTensor(result, [cols], this._requiresGrad);
    }

    if (this.shape.length === 2 && dim === 1) {
      const [rows, cols] = this.shape;
      const result = new Float32Array(rows);
      for (let row = 0; row < rows; row++) {
        let sum = 0;
        for (let col = 0; col < cols; col++) {
          sum += this._data[row * cols + col];
        }
        result[row] = sum;
      }
      return new MockTensor(result, [rows], this._requiresGrad);
    }

    throw new Error(`sum(dim=${dim}) not implemented for shape [${this.shape}]`);
  }

  mean(dim?: number): MockTensor {
    const sumTensor = this.sum(dim);
    const count = dim === undefined ? this.numel() : this.shape[dim];
    return sumTensor.mulScalar(1 / count);
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
