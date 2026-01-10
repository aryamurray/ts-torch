/**
 * Test fixtures for tensor testing
 */

export interface TensorFixture {
  description: string;
  data: number[];
  shape: readonly number[];
}

/**
 * Collection of tensor test fixtures organized by category
 */
export const TensorFixtures = {
  /**
   * Small tensors for basic tests
   */
  small: {
    scalar: {
      description: 'Single value tensor',
      data: [42],
      shape: [] as const,
    },
    vector3: {
      description: '3-element vector',
      data: [1, 2, 3],
      shape: [3] as const,
    },
    matrix2x2: {
      description: '2x2 matrix',
      data: [1, 2, 3, 4],
      shape: [2, 2] as const,
    },
    matrix3x3: {
      description: '3x3 matrix',
      data: [1, 2, 3, 4, 5, 6, 7, 8, 9],
      shape: [3, 3] as const,
    },
  },

  /**
   * Vector fixtures
   */
  vectors: {
    zeros: {
      description: 'Zero vector',
      data: [0, 0, 0, 0, 0],
      shape: [5] as const,
    },
    ones: {
      description: 'Ones vector',
      data: [1, 1, 1, 1, 1],
      shape: [5] as const,
    },
    sequential: {
      description: 'Sequential values',
      data: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
      shape: [10] as const,
    },
    negative: {
      description: 'Negative values',
      data: [-1, -2, -3, -4, -5],
      shape: [5] as const,
    },
    mixed: {
      description: 'Mixed positive and negative',
      data: [1, -2, 3, -4, 5, -6],
      shape: [6] as const,
    },
  },

  /**
   * Matrix fixtures
   */
  matrices: {
    identity3x3: {
      description: '3x3 identity matrix',
      data: [1, 0, 0, 0, 1, 0, 0, 0, 1],
      shape: [3, 3] as const,
    },
    identity4x4: {
      description: '4x4 identity matrix',
      data: [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
      ],
      shape: [4, 4] as const,
    },
    zeros2x3: {
      description: '2x3 zero matrix',
      data: [0, 0, 0, 0, 0, 0],
      shape: [2, 3] as const,
    },
    ones3x2: {
      description: '3x2 ones matrix',
      data: [1, 1, 1, 1, 1, 1],
      shape: [3, 2] as const,
    },
    rectangular: {
      description: '3x4 rectangular matrix',
      data: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
      shape: [3, 4] as const,
    },
  },

  /**
   * Batched tensor fixtures
   */
  batched: {
    batchVectors: {
      description: 'Batch of 4 vectors of length 3',
      data: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
      shape: [4, 3] as const,
    },
    batchMatrices: {
      description: 'Batch of 2 2x2 matrices',
      data: [1, 2, 3, 4, 5, 6, 7, 8],
      shape: [2, 2, 2] as const,
    },
    batchImages: {
      description: 'Batch of 2 grayscale 2x2 images',
      data: [
        1, 2, 3, 4,
        5, 6, 7, 8,
      ],
      shape: [2, 1, 2, 2] as const,
    },
  },

  /**
   * Special value fixtures
   */
  special: {
    smallValues: {
      description: 'Very small values',
      data: [1e-6, 1e-7, 1e-8, 1e-9],
      shape: [4] as const,
    },
    largeValues: {
      description: 'Large values',
      data: [1e6, 1e7, 1e8, 1e9],
      shape: [4] as const,
    },
    fractional: {
      description: 'Fractional values',
      data: [0.1, 0.2, 0.3, 0.4, 0.5],
      shape: [5] as const,
    },
    normalized: {
      description: 'Values between 0 and 1',
      data: [0.0, 0.25, 0.5, 0.75, 1.0],
      shape: [5] as const,
    },
  },

  /**
   * Fixtures for gradient-enabled tensors
   */
  gradEnabled: {
    simpleScalar: {
      description: 'Scalar for backward pass',
      data: [5.0],
      shape: [] as const,
    },
    weights2x3: {
      description: 'Weight matrix for linear layer',
      data: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
      shape: [2, 3] as const,
    },
    biases: {
      description: 'Bias vector',
      data: [0.1, 0.2],
      shape: [2] as const,
    },
  },
};

/**
 * Test pattern types for generating tensors
 */
export type TensorPattern =
  | 'zeros'
  | 'ones'
  | 'sequential'
  | 'random'
  | 'identity';

/**
 * Create a test tensor with a specific pattern and shape
 */
export function createTestTensor(
  pattern: TensorPattern,
  shape: readonly number[]
): TensorFixture {
  const numel = shape.reduce((acc, dim) => acc * dim, 1);
  let data: number[];

  switch (pattern) {
    case 'zeros':
      data = Array(numel).fill(0);
      break;

    case 'ones':
      data = Array(numel).fill(1);
      break;

    case 'sequential':
      data = Array.from({ length: numel }, (_, i) => i);
      break;

    case 'random':
      data = Array.from({ length: numel }, () => Math.random());
      break;

    case 'identity':
      if (shape.length !== 2 || shape[0] !== shape[1]) {
        throw new Error('Identity pattern requires square matrix shape');
      }
      data = Array(numel).fill(0);
      const size = shape[0];
      if (size === undefined) {
        throw new Error('Shape dimension is undefined');
      }
      for (let i = 0; i < size; i++) {
        data[i * size + i] = 1;
      }
      break;

    default:
      throw new Error(`Unknown pattern: ${pattern}`);
  }

  return {
    description: `${pattern} tensor with shape [${shape.join(', ')}]`,
    data,
    shape,
  };
}
