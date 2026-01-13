import { defineConfig } from 'vitest/config';
import { dirname, resolve } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  resolve: {
    // Use "bun" condition to resolve workspace packages to source files
    conditions: ['bun', 'import', 'module', 'default'],
    alias: {
      '@ts-torch/core': resolve(__dirname, '../core/src/index.ts'),
    },
  },
  test: {
    name: '@ts-torch/optim',
    root: __dirname,

    // Test file patterns
    include: ['src/**/*.test.ts'],

    // Global test settings
    globals: true,
    environment: 'node',
    testTimeout: 30000,

    // Pool configuration for FFI compatibility (Vitest 4.0+)
    pool: 'forks',

    // Setup files
    setupFiles: ['./vitest.setup.ts'],
  },

  // Vitest 4.0+ forks options at top level
  forks: {
    singleFork: true,
  },
});
