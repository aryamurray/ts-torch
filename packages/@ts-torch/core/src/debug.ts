/**
 * Debug mode for runtime shape validation.
 * Enabled via TS_TORCH_DEBUG=1 or NODE_ENV=development
 */
export const DebugMode = {
  enabled: process.env.TS_TORCH_DEBUG === "1" || process.env.NODE_ENV === "development",

  enable(): void {
    this.enabled = true;
  },

  disable(): void {
    this.enabled = false;
  },
};

/**
 * Validate two shapes match (for element-wise ops).
 * Only runs if DebugMode.enabled.
 */
export function validateShapesMatch(
  a: readonly number[],
  b: readonly number[],
  operation: string,
): void {
  if (!DebugMode.enabled) return;

  if (a.length !== b.length) {
    throw new TypeError(
      `Shape mismatch in ${operation}: tensors have different ranks (${a.length} vs ${b.length})`,
    );
  }
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) {
      throw new TypeError(`Shape mismatch in ${operation} at dimension ${i}: ${a[i]} vs ${b[i]}`);
    }
  }
}

/**
 * Validate matmul shapes (inner dims match).
 */
export function validateMatmulShapes(a: readonly number[], b: readonly number[]): void {
  if (!DebugMode.enabled) return;

  if (a.length < 1 || b.length < 1) {
    throw new TypeError("matmul requires tensors with at least 1 dimension");
  }
  const k1 = a[a.length - 1];
  const k2 = b.length === 1 ? b[0] : b[b.length - 2];
  if (k1 !== k2) {
    throw new TypeError(
      `matmul dimension mismatch: inner dimensions must match (${k1} vs ${k2}). ` +
        `Got shapes [${a.join(", ")}] @ [${b.join(", ")}]`,
    );
  }
}

/**
 * Validate reshape preserves element count.
 */
export function validateReshape(from: readonly number[], to: readonly number[]): void {
  if (!DebugMode.enabled) return;

  const fromElements = from.reduce((a, b) => a * b, 1);
  const toElements = to.reduce((a, b) => a * b, 1);
  if (fromElements !== toElements) {
    throw new TypeError(
      `Cannot reshape tensor of ${fromElements} elements to shape [${to.join(", ")}] (${toElements} elements)`,
    );
  }
}

/**
 * Validate dimension index is in range.
 */
export function validateDim(shape: readonly number[], dim: number, operation: string): void {
  if (!DebugMode.enabled) return;

  const ndim = shape.length;
  if (dim < -ndim || dim >= ndim) {
    throw new RangeError(
      `Invalid dimension ${dim} for ${operation} on tensor with ${ndim} dimensions`,
    );
  }
}

/**
 * Validate tensor is scalar for .item() call.
 */
export function validateScalar(shape: readonly number[]): void {
  if (!DebugMode.enabled) return;

  const numel = shape.reduce((a, b) => a * b, 1);
  if (numel !== 1) {
    throw new TypeError(`item() only works on scalar tensors. Got tensor with ${numel} elements`);
  }
}

/**
 * Validate broadcast shapes are compatible.
 */
export function validateBroadcast(a: readonly number[], b: readonly number[]): void {
  if (!DebugMode.enabled) return;

  const maxRank = Math.max(a.length, b.length);
  for (let i = 0; i < maxRank; i++) {
    const dimA = a[a.length - 1 - i] ?? 1;
    const dimB = b[b.length - 1 - i] ?? 1;
    if (dimA !== dimB && dimA !== 1 && dimB !== 1) {
      throw new TypeError(`Cannot broadcast shapes [${a.join(", ")}] and [${b.join(", ")}]`);
    }
  }
}

/**
 * Validate tensor is not null/freed.
 */
export function validateTensor(
  tensor: { _freed?: boolean } | null | undefined,
  operation: string,
): void {
  if (!DebugMode.enabled) return;

  if (!tensor) {
    throw new TypeError(`${operation}: tensor is null or undefined`);
  }
  if (tensor._freed) {
    throw new TypeError(`${operation}: tensor has already been freed`);
  }
}
