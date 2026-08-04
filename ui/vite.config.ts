import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
  },
  test: {
    // e2e/ holds Playwright specs (run via `npm run test:e2e`). Without this,
    // Vitest collects them too and fails on Playwright's fixture-based API.
    exclude: ['node_modules/**', 'dist/**', 'e2e/**'],
  },
})
