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
      external: [
        '@ts-torch/core',
        '@ts-torch/nn',
        '@ts-torch/optim',
        '@ts-torch/datasets',
        '@ts-torch/train',
        '@ts-torch/rl',
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
      ],
    },
    sourcemap: true,
    minify: false,
    outDir: 'dist'
  },
  plugins: [
    dts({
      rollupTypes: false,
      exclude: ['**/*.test.ts', '**/*.spec.ts']
    })
  ]
})
