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
        '@ts-torch-platform/loader',
        // Native FFI library
        'koffi',
        // Node built-ins (both prefixed and non-prefixed for compatibility)
        'node:path', 'node:fs', 'node:url', 'node:module', 'node:os', 'node:process',
        'path', 'fs', 'url', 'module', 'os', 'process'
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
