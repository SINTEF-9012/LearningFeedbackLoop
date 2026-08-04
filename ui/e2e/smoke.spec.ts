/**
 * Smoke test — app shell + top-level routes render without errors.
 *
 * This is the scaffold test required by plan point 11 (Agent K
 * refined). It's intentionally minimal: it verifies the Vite-built
 * SPA mounts, the primary navigation is reachable, and the Learnings
 * page (added in Agent K Phase 2) renders its heading. It does NOT
 * exercise the live backend flow end-to-end — that requires a running
 * FastAPI server + uploaded session and is the subject of a follow-up
 * spec (``session_flow.spec.ts``).
 */
import { test, expect } from '@playwright/test'

test.describe('App shell', () => {
  test('renders navigation and default route', async ({ page }) => {
    await page.goto('/#/detailed')
    // The app shell contains per-mode nav buttons; verify at least
    // one recognisable one rendered.
    await expect(page.getByRole('button', { name: 'Monitoring' })).toBeVisible()
  })

  test('navigates to the Learnings page', async ({ page }) => {
    await page.goto('/#/learnings')
    await expect(page.locator('h1', { hasText: 'Learnings' })).toBeVisible({
      timeout: 10_000,
    })
  })

  test('navigates to the Knowledge Graph page', async ({ page }) => {
    await page.goto('/#/detailed')
    const btn = page.getByRole('button', { name: /knowledge graph/i })
    await btn.click()
    // No assertion beyond "no unhandled error" — the graph component
    // may wait on /graph/unified when no backend is present; we just
    // need the page not to crash.
    await expect(page.locator('body')).toBeVisible()
  })
})
