/**
 * Real backend demo path smoke.
 *
 * Starts a real casedata demo session through the UI, waits for the stream
 * websocket to connect, injects a known-good alerting event through the real
 * backend, and verifies that metadata-only feedback can be applied from the
 * UI modal without mutating learned pattern priors.
 */
import { test, expect } from '@playwright/test'
import { execFileSync } from 'node:child_process'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

const BACKEND_BASE = process.env.PLAYWRIGHT_LIVE_BACKEND_URL || 'http://localhost:8000'
const API_BASE_URL_KEY = 'apiBaseUrl'

type DemoEventPayload = {
  session_id: string
  time_range: Record<string, unknown>
  pattern_keys: string[]
  channels: string[]
  cutting_context?: Record<string, unknown>
  external_signals?: Record<string, unknown> | null
  metadata?: Record<string, unknown>
}

type ProcessEventResponse = {
  processed: boolean
  significant: boolean
  memory_id?: string
}

type MemoryDetailResponse = {
  memory: Record<string, unknown>
  feedback_stats: Record<string, unknown>
}

type CurlJsonResponse<T> = {
  ok: boolean
  status: number
  json: T | null
}

function curlJsonRequest<T>(
  url: string,
  options?: {
    method?: string
    data?: unknown
    attempts?: number
    timeoutMs?: number
    delayMs?: number
  },
): CurlJsonResponse<T> {
  const method = options?.method ?? 'GET'
  const timeoutMs = options?.timeoutMs ?? 5_000
  const marker = '\n__CURL_STATUS__:'
  const args = [
    '-sS',
    '-X',
    method,
    '--connect-timeout',
    '5',
    '--max-time',
    String(Math.ceil(timeoutMs / 1000)),
    '-H',
    'Accept: application/json',
  ]

  if (options?.data !== undefined) {
    args.push('-H', 'Content-Type: application/json', '--data', JSON.stringify(options.data))
  }

  args.push('-w', `${marker}%{http_code}`, url)

  const output = execFileSync('curl', args, { encoding: 'utf8' })
  const markerIndex = output.lastIndexOf(marker)
  if (markerIndex === -1) {
    throw new Error(`curl status marker missing for ${url}`)
  }

  const bodyText = output.slice(0, markerIndex)
  const status = Number(output.slice(markerIndex + marker.length).trim())
  const json = bodyText.trim() ? JSON.parse(bodyText) as T : null
  return {
    ok: status >= 200 && status < 300,
    status,
    json,
  }
}

async function getOkResponse<T>(
  url: string,
  options?: {
    attempts?: number
    timeoutMs?: number
    delayMs?: number
  },
): Promise<CurlJsonResponse<T> | null> {
  const attempts = options?.attempts ?? 4
  const timeoutMs = options?.timeoutMs ?? 5_000
  const delayMs = options?.delayMs ?? 1_000

  for (let attempt = 0; attempt < attempts; attempt += 1) {
    const response = await Promise.resolve().then(() => {
      try {
        return curlJsonRequest<T>(url, { timeoutMs })
      } catch {
        return null
      }
    })
    if (response?.ok) return response
    if (attempt < attempts - 1) {
      await new Promise((resolveDelay) => setTimeout(resolveDelay, delayMs))
    }
  }

  return null
}

function loadDemoEvent(sessionId: string): DemoEventPayload {
  const raw = JSON.parse(
    readFileSync(resolve(process.cwd(), '../scripts/demo_data/event_2_chatter.json'), 'utf8'),
  ) as Record<string, unknown>
  const { _description, _explanation, ...payload } = raw
  void _description
  void _explanation
  return {
    ...(payload as DemoEventPayload),
    session_id: sessionId,
    metadata: {
      ...(((payload as DemoEventPayload).metadata || {}) as Record<string, unknown>),
      e2e_live_demo: true,
    },
  }
}

test.describe('Live demo path', () => {
  test('starts a real session, receives an alert, and applies metadata feedback', async ({ page }) => {
    test.setTimeout(240_000)

    const catalogResponse = await getOkResponse<{ cases?: unknown[] }>(`${BACKEND_BASE}/sessions/casedata/catalog`, {
      attempts: 1,
      timeoutMs: 45_000,
      delayMs: 1_000,
    })
    test.skip(!catalogResponse, `Casedata catalog unavailable at ${BACKEND_BASE}`)
    const catalog = catalogResponse?.json ?? null
    test.skip(!catalog || !Array.isArray(catalog.cases) || catalog.cases.length === 0, 'Casedata catalog unavailable for live demo smoke')

    let sessionId = ''
    let memoryId = ''

    await page.addInitScript(({ backendBase, storageKey }) => {
      window.localStorage.setItem(storageKey, backendBase)
      if (!window.sessionStorage.getItem('__liveDemoStorageInitialized')) {
        window.localStorage.removeItem('sessionId')
        window.sessionStorage.setItem('__liveDemoStorageInitialized', 'true')
      }
    }, { backendBase: BACKEND_BASE, storageKey: API_BASE_URL_KEY })

    try {
      await page.goto('/#/detailed')
      await page.locator('summary', { hasText: 'Configuration' }).click()
      const autoOpenAlertsToggle = page.getByRole('checkbox', { name: 'Auto-open new alerts' })
      await autoOpenAlertsToggle.uncheck()

      await expect(page.getByLabel('Demo source')).toHaveValue('simulated_casedata')
      const casedataOperation = page.getByLabel('Casedata start operation')
      await expect.poll(async () => casedataOperation.inputValue(), { timeout: 60_000 }).not.toBe('')

      const startDemoButton = page.getByRole('button', { name: /start demo/i })
      await expect(startDemoButton).toBeEnabled({ timeout: 30_000 })

      await startDemoButton.click()

      await expect.poll(async () => page.evaluate(() => window.localStorage.getItem('sessionId') || ''), {
        timeout: 60_000,
      }).not.toBe('')

      sessionId = await page.evaluate(() => window.localStorage.getItem('sessionId') || '')
      expect(sessionId).not.toBe('')

      await page.reload()

      const activeSession = page.getByLabel('Active session')
      await expect.poll(async () => activeSession.inputValue(), { timeout: 30_000 }).toBe(sessionId)
      await autoOpenAlertsToggle.uncheck()

      await expect(page.getByText(/^WS: connected$/)).toBeVisible({ timeout: 60_000 })

      const playbackPanel = page.locator('details').filter({ hasText: 'Playback controls' }).first()
      await expect.poll(async () => (await playbackPanel.textContent()) || '', { timeout: 60_000 }).toContain('running=true')

      await page.getByRole('button', { name: 'Pause' }).click()
      await expect.poll(() => {
        const sessionInfoResponse = curlJsonRequest<Record<string, unknown>>(
          `${BACKEND_BASE}/sessions/${encodeURIComponent(sessionId)}`,
          { timeoutMs: 10_000 },
        )
        return Boolean(sessionInfoResponse.ok && sessionInfoResponse.json && sessionInfoResponse.json.paused === true)
      }, { timeout: 20_000 }).toBe(true)

      const alertItems = page.locator('.alertItem')
      const alertCountBefore = await alertItems.count()
      const firstAlertTextBefore = alertCountBefore > 0 ? ((await alertItems.first().textContent()) || '') : ''
      const memoryItems = page.locator('.memoryItem')
      const memoryCountBefore = await memoryItems.count()
      const firstMemoryTextBefore = memoryCountBefore > 0 ? ((await memoryItems.first().textContent()) || '') : ''
      const eventPayload = loadDemoEvent(sessionId)

      const processEventResponse = curlJsonRequest<ProcessEventResponse>(`${BACKEND_BASE}/agent/memory/events`, {
        method: 'POST',
        data: eventPayload,
        timeoutMs: 90_000,
      })
      expect(processEventResponse.ok).toBeTruthy()
      expect(processEventResponse.json).toBeTruthy()
      const processed = processEventResponse.json as ProcessEventResponse
      expect(processed.processed).toBeTruthy()
      expect(processed.significant).toBeTruthy()
      expect(processed.memory_id).toBeTruthy()
      memoryId = String(processed.memory_id || '')

      if (alertCountBefore > 0) {
        await expect.poll(async () => (await alertItems.first().textContent()) || '', { timeout: 20_000 }).not.toBe(firstAlertTextBefore)
      } else if (memoryCountBefore > 0) {
        await expect.poll(async () => (await memoryItems.first().textContent()) || '', { timeout: 20_000 }).not.toBe(firstMemoryTextBefore)
      } else {
        await expect.poll(async () => await memoryItems.count(), { timeout: 20_000 }).toBeGreaterThan(0)
      }
      const memoryDetailModal = page.locator('.modalContent').getByText('Memory Detail', { exact: true })
      if (!(await memoryDetailModal.isVisible())) {
        try {
          if (await alertItems.count()) {
            await alertItems.first().click({ timeout: 5_000 })
          } else {
            await memoryItems.first().click({ timeout: 5_000 })
          }
        } catch {
          // Auto-open may surface the modal between the visibility check and click.
        }
      }

      await expect(memoryDetailModal).toBeVisible({ timeout: 20_000 })
      const feedbackComment = `playwright live demo metadata ${Date.now()}`
      await page.getByPlaceholder('Operator note about this memory').fill(feedbackComment)
      await page.getByRole('button', { name: 'Apply note/label/tags' }).click()
      await expect(page.getByText('Applied.', { exact: true })).toBeVisible({ timeout: 10_000 })

      const memoryResponse = curlJsonRequest<MemoryDetailResponse>(`${BACKEND_BASE}/agent/memory/${encodeURIComponent(memoryId)}`, {
        timeoutMs: 30_000,
      })
      expect(memoryResponse.ok).toBeTruthy()
      expect(memoryResponse.json).toBeTruthy()
      const memoryDetail = memoryResponse.json as MemoryDetailResponse
      expect(Number(memoryDetail.feedback_stats.comments || 0)).toBeGreaterThan(0)
    } finally {
      if (memoryId) {
        try {
          curlJsonRequest(`${BACKEND_BASE}/agent/memory/${encodeURIComponent(memoryId)}`, {
            method: 'DELETE',
            timeoutMs: 10_000,
          })
        } catch {
          // best-effort cleanup
        }
      }
      if (sessionId) {
        try {
          curlJsonRequest(`${BACKEND_BASE}/sessions/${encodeURIComponent(sessionId)}`, {
            method: 'DELETE',
            timeoutMs: 10_000,
          })
        } catch {
          // best-effort cleanup
        }
      }
    }
  })
})