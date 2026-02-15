import { defineConfig } from 'vite'
import { resolve } from 'path'
import dts from 'vite-plugin-dts'

export default defineConfig({
  build: {
    lib: {
      entry: resolve(__dirname, 'src/index.ts'),
      formats: ['es', 'cjs'],
      fileName: (format) => `index.${format === 'es' ? 'mjs' : 'cjs'}`
    },
    rollupOptions: {
      // Externalize dependencies that shouldn't be bundled
      external: [
        // Workspace dependencies
        '@ts-torch/core',
        // Node built-ins (both with and without node: prefix)
        'path',
        'fs',
        'fs/promises',
        'url',
        'module',
        'os',
        'process',
        'node:path',
        'node:fs',
        'node:fs/promises',
        'node:url',
        'node:module',
        'node:os',
        'node:process',
        'crypto',
        'node:crypto',
      ],
    },
    sourcemap: true,
    outDir: 'dist'
  },
  plugins: [
    dts({
      rollupTypes: true,
      // Exclude test files
      exclude: ['**/*.test.ts', '**/*.spec.ts']
    })
  ]
})
