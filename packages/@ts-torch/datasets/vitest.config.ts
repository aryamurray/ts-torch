import { defineConfig } from 'vitest/config';
import { dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  test: {
    name: '@ts-torch/datasets',
    root: __dirname,

    // Test environment
    globals: true,
    environment: 'node',
    testTimeout: 60000, // Longer timeout for dataset loading

    // Pool configuration for FFI compatibility (Vitest 4.0+)
    pool: 'forks',

    // Test file patterns
    include: [
      'src/**/*.test.ts',
      '__tests__/**/*.test.ts',
    ],
    exclude: [
      '**/node_modules/**',
      '**/dist/**',
      '**/build/**',
    ],

    // Setup files
    setupFiles: ['./vitest.setup.ts'],

    // Coverage configuration
    coverage: {
      provider: 'v8',
      include: ['src/**/*.ts'],
      exclude: [
        'src/**/*.test.ts',
        'src/**/*.d.ts',
        'src/**/__tests__/**',
      ],
      reporter: ['text', 'json', 'html'],
    },
  },

  // Vitest 4.0+ forks options at top level
  forks: {
    singleFork: true,
  },
});
