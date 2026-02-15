import { defineConfig } from 'vite'
import { resolve } from 'path'
import dts from 'vite-plugin-dts'

export default defineConfig({
  build: {
    lib: {
      entry: resolve(__dirname, 'src/index.ts'),
      formats: ['es', 'cjs'],
      fileName: (format) => `index.${format === 'es' ? 'mjs' : 'cjs'}`,
    },
    rollupOptions: {
      external: [/^node:/],
    },
    sourcemap: true,
    outDir: 'dist',
  },
  plugins: [
    dts({
      rollupTypes: true,
      exclude: ['**/*.test.ts', '**/*.spec.ts'],
    }),
  ],
})
