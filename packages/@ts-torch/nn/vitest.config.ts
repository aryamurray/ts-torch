import { defineConfig } from 'vitest/config';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  test: {
    name: '@ts-torch/nn',
    root: __dirname,

    // Test file patterns
    include: ['src/**/*.test.ts', 'src/**/__tests__/**/*.test.ts'],

    // Test timeout
    testTimeout: 30000,

    // Pool configuration for FFI compatibility (Vitest 4.0+)
    pool: 'forks',

    // Setup files
    setupFiles: ['./vitest.setup.ts'],

    // Environment
    globals: true,
    environment: 'node',
  },

  // Vitest 4.0+ forks options at top level
  forks: {
    singleFork: true,
  },

  resolve: {
    alias: {
      '@ts-torch/test-utils': resolve(__dirname, '../../shared-test-utils/src/index.ts'),
      '@ts-torch/core': resolve(__dirname, '../core/src/index.ts'),
      // Needed by e2e tests that import train/datasets
      '@ts-torch/train': resolve(__dirname, '../train/src/index.ts'),
      '@ts-torch/datasets': resolve(__dirname, '../datasets/src/index.ts'),
      '@ts-torch/optim': resolve(__dirname, '../optim/src/index.ts'),
      '@ts-torch/nn': resolve(__dirname, './src/index.ts'),
    },
  },
});
