/**
 * Tests for MNIST dataset loader
 *
 * Tests MNIST data loading with mocked file system operations.
 */

import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { MNIST } from '../mnist.js';

// Import test utils from source
const utilsModule = await import('../../../../core/src/test/utils.js');
const { scopedTest } = utilsModule;

// Mock fs module
const readFileSyncMock = vi.fn();
vi.mock('fs', () => ({
  readFileSync: readFileSyncMock,
}));

/**
 * Create mock IDX3 image file buffer
 */
function createMockImagesBuffer(numImages: number, rows = 28, cols = 28): Buffer {
  const headerSize = 16;
  const imageSize = rows * cols;
  const totalSize = headerSize + numImages * imageSize;
  const buffer = Buffer.alloc(totalSize);

  // Write header
  const view = new DataView(buffer.buffer, buffer.byteOffset);
  view.setUint32(0, 0x00000803, false); // magic number
  view.setUint32(4, numImages, false);
  view.setUint32(8, rows, false);
  view.setUint32(12, cols, false);

  // Write pixel data (simple pattern)
  for (let i = 0; i < numImages; i++) {
    for (let j = 0; j < imageSize; j++) {
      buffer[headerSize + i * imageSize + j] = (i + j) % 256;
    }
  }

  return buffer;
}

/**
 * Create mock IDX1 label file buffer
 */
function createMockLabelsBuffer(numLabels: number): Buffer {
  const headerSize = 8;
  const totalSize = headerSize + numLabels;
  const buffer = Buffer.alloc(totalSize);

  // Write header
  const view = new DataView(buffer.buffer, buffer.byteOffset);
  view.setUint32(0, 0x00000801, false); // magic number
  view.setUint32(4, numLabels, false);

  // Write labels (0-9 pattern)
  for (let i = 0; i < numLabels; i++) {
    buffer[headerSize + i] = i % 10;
  }

  return buffer;
}

describe('MNIST', () => {
  let consoleLogSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    vi.clearAllMocks();
    consoleLogSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
  });

  afterEach(() => {
    consoleLogSpy.mockRestore();
  });

  describe('Constructor', () => {
    it('should create MNIST dataset for training', () => {
      const mnist = new MNIST('./data', true);
      expect(mnist).toBeDefined();
    });

    it('should create MNIST dataset for testing', () => {
      const mnist = new MNIST('./data', false);
      expect(mnist).toBeDefined();
    });

    it('should default to training mode', () => {
      const mnist = new MNIST('./data');
      expect(mnist).toBeDefined();
    });

    it('should have length of 0 before loading', () => {
      const mnist = new MNIST('./data', true);
      expect(mnist.length).toBe(0);
    });
  });

  describe('load()', () => {
    it('should load training data', async () => {
      const numSamples = 100;
      const imagesBuffer = createMockImagesBuffer(numSamples);
      const labelsBuffer = createMockLabelsBuffer(numSamples);

      readFileSyncMock.mockImplementation((path) => {
        if (typeof path === 'string' && path.includes('train-images')) {
          return imagesBuffer;
        }
        if (typeof path === 'string' && path.includes('train-labels')) {
          return labelsBuffer;
        }
        throw new Error('Unexpected file path');
      });

      const mnist = new MNIST('./data', true);
      await mnist.load();

      expect(mnist.length).toBe(numSamples);
      expect(readFileSyncMock).toHaveBeenCalledTimes(2);
      expect(consoleLogSpy).toHaveBeenCalledWith(`Loaded MNIST train: ${numSamples} samples`);
    });

    it('should load test data', async () => {
      const numSamples = 50;
      const imagesBuffer = createMockImagesBuffer(numSamples);
      const labelsBuffer = createMockLabelsBuffer(numSamples);

      readFileSyncMock.mockImplementation((path) => {
        if (typeof path === 'string' && path.includes('t10k-images')) {
          return imagesBuffer;
        }
        if (typeof path === 'string' && path.includes('t10k-labels')) {
          return labelsBuffer;
        }
        throw new Error('Unexpected file path');
      });

      const mnist = new MNIST('./data', false);
      await mnist.load();

      expect(mnist.length).toBe(numSamples);
      expect(consoleLogSpy).toHaveBeenCalledWith(`Loaded MNIST test: ${numSamples} samples`);
    });

    it('should throw error for invalid images magic number', async () => {
      const buffer = Buffer.alloc(16);
      const view = new DataView(buffer.buffer);
      view.setUint32(0, 0x12345678, false); // Invalid magic

      readFileSyncMock.mockReturnValue(buffer);

      const mnist = new MNIST('./data', true);
      await expect(mnist.load()).rejects.toThrow('Invalid MNIST images magic number');
    });

    it('should throw error for invalid labels magic number', async () => {
      const imagesBuffer = createMockImagesBuffer(10);
      const labelsBuffer = Buffer.alloc(8);
      const view = new DataView(labelsBuffer.buffer);
      view.setUint32(0, 0x12345678, false); // Invalid magic

      readFileSyncMock.mockImplementation((path) => {
        if (typeof path === 'string' && path.includes('images')) {
          return imagesBuffer;
        }
        return labelsBuffer;
      });

      const mnist = new MNIST('./data', true);
      await expect(mnist.load()).rejects.toThrow('Invalid MNIST labels magic number');
    });

    it('should throw error for unexpected image dimensions', async () => {
      const imagesBuffer = createMockImagesBuffer(10, 32, 32); // Wrong size
      const labelsBuffer = createMockLabelsBuffer(10);

      readFileSyncMock.mockImplementation((path) => {
        if (typeof path === 'string' && path.includes('images')) {
          return imagesBuffer;
        }
        return labelsBuffer;
      });

      const mnist = new MNIST('./data', true);
      await expect(mnist.load()).rejects.toThrow('Unexpected MNIST image size: 32x32');
    });

    it('should normalize pixel values to [0, 1]', async () => {
      const imagesBuffer = createMockImagesBuffer(1);
      const labelsBuffer = createMockLabelsBuffer(1);

      readFileSyncMock.mockImplementation((path) => {
        if (typeof path === 'string' && path.includes('images')) {
          return imagesBuffer;
        }
        return labelsBuffer;
      });

      const mnist = new MNIST('./data', true);
      await mnist.load();

      // Verify data is loaded (tested via get() in next section)
      expect(mnist.length).toBe(1);
    });
  });

  describe('get()', () => {
    beforeEach(async () => {
      const numSamples = 10;
      const imagesBuffer = createMockImagesBuffer(numSamples);
      const labelsBuffer = createMockLabelsBuffer(numSamples);

      readFileSyncMock.mockImplementation((path) => {
        if (typeof path === 'string' && path.includes('images')) {
          return imagesBuffer;
        }
        return labelsBuffer;
      });
    });

    it('should get a single sample', async () => {
      await scopedTest(async () => {
        const mnist = new MNIST('./data', true);
        await mnist.load();

        const sample = mnist.get(0);

        expect(sample).toBeDefined();
        expect(sample.image).toBeDefined();
        expect(sample.label).toBeDefined();
        expect(sample.image).toHaveShape([784]);
        expect(sample.label).toBeGreaterThanOrEqual(0);
        expect(sample.label).toBeLessThanOrEqual(9);
      });
    });

    it('should throw error if not loaded', () =>
      scopedTest(() => {
        const mnist = new MNIST('./data', true);
        expect(() => mnist.get(0)).toThrow('MNIST not loaded. Call load() first.');
      }));

    it('should throw error for negative index', async () => {
      await scopedTest(async () => {
        const mnist = new MNIST('./data', true);
        await mnist.load();

        expect(() => mnist.get(-1)).toThrow('Index -1 out of bounds');
      });
    });

    it('should throw error for index >= length', async () => {
      await scopedTest(async () => {
        const mnist = new MNIST('./data', true);
        await mnist.load();

        expect(() => mnist.get(10)).toThrow('Index 10 out of bounds');
      });
    });

    it('should get correct label for each sample', async () => {
      await scopedTest(async () => {
        const mnist = new MNIST('./data', true);
        await mnist.load();

        for (let i = 0; i < mnist.length; i++) {
          const sample = mnist.get(i);
          expect(sample.label).toBe(i % 10); // Pattern from mock
        }
      });
    });

    it('should return normalized image data', async () => {
      await scopedTest(async () => {
        const mnist = new MNIST('./data', true);
        await mnist.load();

        const sample = mnist.get(0);
        const data = Array.from(sample.image.toArray() as Iterable<number | bigint>).map(Number);

        // All values should be in [0, 1]
        data.forEach((val) => {
          expect(val).toBeGreaterThanOrEqual(0);
          expect(val).toBeLessThanOrEqual(1);
        });
      });
    });
  });

  describe('getBatch()', () => {
    beforeEach(async () => {
      const numSamples = 20;
      const imagesBuffer = createMockImagesBuffer(numSamples);
      const labelsBuffer = createMockLabelsBuffer(numSamples);

      readFileSyncMock.mockImplementation((path) => {
        if (typeof path === 'string' && path.includes('images')) {
          return imagesBuffer;
        }
        return labelsBuffer;
      });
    });

    it('should get a batch of samples', async () => {
      await scopedTest(async () => {
        const mnist = new MNIST('./data', true);
        await mnist.load();

        const batch = mnist.getBatch(0, 5);

        expect(batch.images).toBeDefined();
        expect(batch.labels).toBeDefined();
        expect(batch.images).toHaveShape([5, 784]);
        expect(batch.labels).toHaveLength(5);
      });
    });

    it('should throw error if not loaded', () =>
      scopedTest(() => {
        const mnist = new MNIST('./data', true);
        expect(() => mnist.getBatch(0, 5)).toThrow('MNIST not loaded. Call load() first.');
      }));

    it('should handle batch at end of dataset', async () => {
      await scopedTest(async () => {
        const mnist = new MNIST('./data', true);
        await mnist.load();

        const batch = mnist.getBatch(15, 10);

        expect(batch.images).toHaveShape([5, 784]); // Only 5 samples left
        expect(batch.labels).toHaveLength(5);
      });
    });

    it('should return correct labels in batch', async () => {
      await scopedTest(async () => {
        const mnist = new MNIST('./data', true);
        await mnist.load();

        const batch = mnist.getBatch(0, 10);

        expect(batch.labels).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
      });
    });
  });

  describe('batches()', () => {
    beforeEach(async () => {
      const numSamples = 25;
      const imagesBuffer = createMockImagesBuffer(numSamples);
      const labelsBuffer = createMockLabelsBuffer(numSamples);

      readFileSyncMock.mockImplementation((path) => {
        if (typeof path === 'string' && path.includes('images')) {
          return imagesBuffer;
        }
        return labelsBuffer;
      });
    });

    it('should iterate over batches', async () => {
      await scopedTest(async () => {
        const mnist = new MNIST('./data', true);
        await mnist.load();

        const batches = Array.from(mnist.batches(5));

        expect(batches).toHaveLength(5);
        batches.forEach((batch) => {
          expect(batch.images).toHaveShape([5, 784]);
          expect(batch.labels).toHaveLength(5);
          expect(batch.labelsTensor).toHaveShape([5]);
        });
      });
    });

    it('should throw error if not loaded', () =>
      scopedTest(() => {
        const mnist = new MNIST('./data', true);
        expect(() => Array.from(mnist.batches(5))).toThrow('MNIST not loaded. Call load() first.');
      }));

    it('should handle last incomplete batch', async () => {
      await scopedTest(async () => {
        const mnist = new MNIST('./data', true);
        await mnist.load();

        const batches = Array.from(mnist.batches(10));

        expect(batches).toHaveLength(3);
        expect(batches[0]!.images).toHaveShape([10, 784]);
        expect(batches[1]!.images).toHaveShape([10, 784]);
        expect(batches[2]!.images).toHaveShape([5, 784]); // Last batch
      });
    });

    it('should shuffle when requested', async () => {
      await scopedTest(async () => {
        const mnist = new MNIST('./data', true);
        await mnist.load();

        const batches1 = Array.from(mnist.batches(25, true));
        const batches2 = Array.from(mnist.batches(25, true));

        // Both should have same total labels
        const labels1 = batches1[0]!.labels.sort();
        const labels2 = batches2[0]!.labels.sort();

        expect(labels1).toEqual(Array.from({ length: 25 }, (_, i) => i % 10));
        expect(labels2).toEqual(Array.from({ length: 25 }, (_, i) => i % 10));
      });
    });

    it('should not shuffle by default', async () => {
      await scopedTest(async () => {
        const mnist = new MNIST('./data', true);
        await mnist.load();

        const batches = Array.from(mnist.batches(10));

        expect(batches[0]!.labels).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        expect(batches[1]!.labels).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
      });
    });

    it('should create labelsTensor with int64 dtype', async () => {
      await scopedTest(async () => {
        const mnist = new MNIST('./data', true);
        await mnist.load();

        const batches = Array.from(mnist.batches(5));

        expect(batches[0]!.labelsTensor.dtype.name).toBe('int64');
      });
    });

    it('should handle single batch', async () => {
      await scopedTest(async () => {
        const mnist = new MNIST('./data', true);
        await mnist.load();

        const batches = Array.from(mnist.batches(100));

        expect(batches).toHaveLength(1);
        expect(batches[0]!.images).toHaveShape([25, 784]);
      });
    });

    it('should handle batch size of 1', async () => {
      await scopedTest(async () => {
        const mnist = new MNIST('./data', true);
        await mnist.load();

        const batches = Array.from(mnist.batches(1));

        expect(batches).toHaveLength(25);
        batches.forEach((batch) => {
          expect(batch.images).toHaveShape([1, 784]);
          expect(batch.labels).toHaveLength(1);
        });
      });
    });
  });

  describe('classes property', () => {
    it('should return array of digit strings', () => {
      const mnist = new MNIST('./data', true);
      expect(mnist.classes).toEqual(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']);
    });

    it('should have 10 classes', () => {
      const mnist = new MNIST('./data', true);
      expect(mnist.classes).toHaveLength(10);
    });
  });
});
