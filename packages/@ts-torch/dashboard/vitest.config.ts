import { defineConfig } from 'vitest/config'
import { dirname } from 'path'
import { fileURLToPath } from 'url'

const __dirname = dirname(fileURLToPath(import.meta.url))

export default defineConfig({
  test: {
    name: '@ts-torch/dashboard',
    root: __dirname,

    // Test file patterns
    include: ['src/**/*.test.ts'],

    // Global test settings
    globals: true,
    environment: 'node',
    testTimeout: 30000,
  },
})
