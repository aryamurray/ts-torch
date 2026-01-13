import { defineConfig } from 'vitest/config';

export default defineConfig({
  resolve: {
    // Use "bun" condition to resolve to source files during development/testing
    conditions: ['bun', 'import', 'module', 'default'],
  },
  test: {
    // Project mode for monorepo (Vitest 4.0+ uses 'projects' instead of 'workspace')
    projects: ['packages/@ts-torch/*/vitest.config.ts'],

    // Global test settings
    globals: true,
    environment: 'node',
    testTimeout: 30000,

    // Pool configuration for FFI compatibility (Vitest 4.0+ - top-level)
    pool: 'forks',

    // Test file patterns
    include: ['**/*.{test,spec}.{ts,tsx}'],
    exclude: [
      '**/node_modules/**',
      '**/dist/**',
      '**/build/**',
      '**/.{idea,git,cache,output,temp}/**',
    ],

    // Coverage configuration
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html', 'lcov'],
      thresholds: {
        lines: 70,
        functions: 70,
        statements: 70,
        branches: 60,
      },
      exclude: [
        '**/node_modules/**',
        '**/dist/**',
        '**/build/**',
        '**/*.config.{ts,js}',
        '**/*.d.ts',
        '**/test/**',
        '**/__tests__/**',
      ],
    },

    // Setup files
    setupFiles: ['./vitest.setup.ts'],
  },

  // Vitest 4.0+ forks options at top level
  forks: {
    singleFork: true,
  },
});
