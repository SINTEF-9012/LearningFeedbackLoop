/**
 * Playwright configuration — Agent K smoke test scaffold.
 *
 * Per plan point 11 (Agent K refined): at least one end-to-end browser
 * smoke test covering the core session → upload → alert → confirm →
 * prior bar moves flow. This config keeps it hermetic-first: by
 * default it boots ``vite preview`` on port 4173 and runs against the
 * production build. The backend must be reachable at the configured
 * VITE_API_URL (default ``http://localhost:8000``); the test suite
 * marks tests as ``test.skip`` when the backend is unreachable so CI
 * without a live server still passes.
 *
 * Usage::
 *
 *   npm run test:e2e:install   # one-time
 *   npm run test:e2e
 */
import { defineConfig, devices } from '@playwright/test'

const PORT = Number(process.env.PLAYWRIGHT_UI_PORT || 4173)

export default defineConfig({
  testDir: './e2e',
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  reporter: 'list',
  use: {
    baseURL: `http://localhost:${PORT}`,
    trace: 'retain-on-failure',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
  webServer: {
    command: `npm run build && npx vite preview --port ${PORT} --strictPort`,
    port: PORT,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
})
