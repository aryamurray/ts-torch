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
        '@ts-torch/nn',
        '@ts-torch/optim',
        // Node built-ins
        'path',
        'fs',
        'url',
        'module',
        'os',
        'process'
      ],
    },
    sourcemap: true,
    minify: false,
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
