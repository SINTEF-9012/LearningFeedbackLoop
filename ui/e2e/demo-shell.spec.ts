import { test, expect, type Page } from '@playwright/test'

type SessionSummary = {
  session_id: string
  status: string
  status_label: string
  running: boolean
  paused: boolean
  position: number
  total_samples: number
  progress: number | null
  last_error: string | null
  loading: boolean
  source: string
  source_label: string | null
  case_dir: string | null
  operation_id: string | null
  tool_id: string | null
  resolved_start_position: number | null
  requested_start_position: number | null
  start_at_first_cutting_row: boolean
}

function buildSessionSummary(overrides: Partial<SessionSummary> = {}): SessionSummary {
  return {
    session_id: 'demo-casedata-001',
    status: 'loading',
    status_label: 'Loading',
    running: false,
    paused: false,
    position: 0,
    total_samples: 120,
    progress: 0,
    last_error: null,
    loading: true,
    source: 'simulated_casedata',
    source_label: 'SITE_C - MACHINE_C1 - CASE_C1 / OF00001',
    case_dir: 'SITE_C - MACHINE_C1 - CASE_C1',
    operation_id: 'OF00001',
    tool_id: 'T01',
    resolved_start_position: 42,
    requested_start_position: 15,
    start_at_first_cutting_row: true,
    ...overrides,
  }
}

async function mockShellApi(page: Page, sessionSummaries: SessionSummary[]) {
  await page.route(/\/sessions(?:\?.*)?$/, async route => {
    await route.fulfill({
      json: {
        sessions: sessionSummaries.map((session) => session.session_id),
        session_summaries: sessionSummaries,
      },
    })
  })

  await page.route(/\/agent\/memory\/scorer\/priors(?:\?.*)?$/, async route => {
    await route.fulfill({ json: { priors: [] } })
  })

  await page.route(/\/agent\/memory\/experiment\/runs(?:\?.*)?$/, async route => {
    await route.fulfill({ json: { runs: [] } })
  })

  await page.route(/\/health$/, async route => {
    await route.fulfill({ json: { status: 'ok' } })
  })

  await page.route(/\/agent\/memory\/llm\/status$/, async route => {
    await route.fulfill({ json: { available: true, provider: 'mock-llm', model: 'demo' } })
  })

  await page.route(/\/health\/sindit$/, async route => {
    await route.fulfill({ json: { sindit: false, graphdb: false, sindit_enabled: false } })
  })
}

test.describe('Demo shell', () => {
  test('operator mode hides advanced tabs and redirects experiment route', async ({ page }) => {
    await mockShellApi(page, [])

    await page.goto('/#/settings')
    await expect(page.getByRole('button', { name: 'Experiment' })).toBeVisible()

    await page.getByLabel('Operator mode').check()
    await expect(page.getByRole('button', { name: 'Experiment' })).toHaveCount(0)

    await page.goto('/#/experiment')
    await expect(page).toHaveURL(/#\/operator$/)
  })

  test('launches a casedata demo session from the configuration drawer', async ({ page }) => {
    const sessionSummaries: SessionSummary[] = []
    let startDemoPayload: Record<string, unknown> | null = null

    await mockShellApi(page, sessionSummaries)

    await page.route(/\/sessions\/casedata\/catalog(?:\?.*)?$/, async route => {
      await route.fulfill({
        json: {
          root: 'data/casedata',
          cases: [
            {
              case_dir: 'Site_b - MACHINE_B1 - CASE_B1',
              label: 'Site_b fallback machine',
              default_operation_id: 'OF09001',
              default_valid_operation_id: 'OF09001',
              operations: [
                {
                  operation_id: 'OF09001',
                  tool_id: 'T11',
                  tool_label: 'Tool 11',
                  n_channels: 6,
                  harmonic_ready: true,
                  missing_fields: [],
                },
              ],
            },
            {
              case_dir: 'SITE_C - MACHINE_C1 - CASE_C1',
              label: 'SITE_C demo machine',
              default_operation_id: 'OF00001',
              default_valid_operation_id: 'OF00001',
              operations: [
                {
                  operation_id: 'OF00001',
                  tool_id: 'T01',
                  tool_label: 'Tool 1',
                  n_channels: 6,
                  harmonic_ready: true,
                  missing_fields: [],
                },
                {
                  operation_id: 'OF00002',
                  tool_id: 'T02',
                  tool_label: 'Tool 2',
                  n_channels: 4,
                  harmonic_ready: false,
                  missing_fields: ['Vibration_Peak_1_X_Amplitude'],
                },
              ],
            },
          ],
        },
      })
    })

    await page.route(/\/sessions\/start-demo$/, async route => {
      startDemoPayload = JSON.parse(route.request().postData() || '{}')
      sessionSummaries.splice(0, sessionSummaries.length, buildSessionSummary())
      await route.fulfill({
        json: {
          session_id: 'demo-casedata-001',
          ws_url: '/streams/demo-casedata-001',
          mode: 'casedata',
          source: 'simulated_casedata',
          n_events: 0,
          status: 'loading',
        },
      })
    })

    await page.route(/\/sessions\/demo-casedata-001$/, async route => {
      await route.fulfill({
        json: {
          config: { speed: 1, samples_per_tick: 1 },
          metadata: {
            source: 'simulated_casedata',
            casedata: {
              case_dir: 'SITE_C - MACHINE_C1 - CASE_C1',
              operation_id: 'OF00001',
            },
          },
          running: false,
          paused: false,
          position: 15,
        },
      })
    })

    await page.route(/\/agent\/memory\/session\/demo-casedata-001(?:\?.*)?$/, async route => {
      await route.fulfill({ json: { memories: [], total_count: 0 } })
    })

    await page.route(/\/harmonic\/status(?:\?.*)?$/, async route => {
      await route.fulfill({
        json: {
          available: true,
          torch_installed: true,
          model_loaded: true,
          dataset_name: 'pair_lfl',
          scorer_kind: 'pair',
          n_harm_features: 0,
          n_params: 0,
          harmonic_mode: 'pre_extracted',
          cnn_window: 0,
          trained_at: null,
          training_metrics: {},
          model_save_path: 'data/models/harmonic_pair_lfl.pt',
          model_path_exists: true,
          checkpoint_statuses: {},
        },
      })
    })

    await page.goto('/#/detailed')
    await page.locator('summary', { hasText: 'Configuration' }).click()

    await expect(page.getByLabel('Demo source')).toHaveValue('simulated_casedata')
    await expect(page.getByLabel('Harmonic-ready tools only')).toBeChecked()
    await expect(page.getByLabel('Start at first cutting row')).toBeChecked()
    await expect(page.getByLabel('Casedata machine')).toHaveValue('SITE_C - MACHINE_C1 - CASE_C1')
    await expect(page.getByLabel('Casedata start operation')).toHaveValue('OF00001')

    const preflightPanel = page.locator('.panel').filter({ hasText: 'Demo preflight' }).last()
    await expect(preflightPanel).toContainText('Ready for the recommended operator demo.')
    await expect(preflightPanel).toContainText('Backend API')
    await expect(preflightPanel).toContainText('Casedata catalog')
    await expect(preflightPanel).toContainText('Harmonic pair_lfl')
    await expect(preflightPanel).toContainText('LLM explanations')
    await expect(preflightPanel).toContainText('SINDIT / GraphDB')

    await page.getByLabel('Casedata skip ahead').fill('15')
    await page.getByRole('button', { name: /start demo/i }).click()

    await expect.poll(() => startDemoPayload).not.toBeNull()
    expect(startDemoPayload).toMatchObject({
      mode: 'casedata',
      source: 'simulated_casedata',
      reset_priors: false,
      speed: 1,
      samples_per_tick: 1,
      valid_tools_only: true,
      start_at_first_cutting_row: true,
      case_dir: 'SITE_C - MACHINE_C1 - CASE_C1',
      operation_id: 'OF00001',
      start_position: 15,
    })

    await expect(page.getByLabel('Active session')).toHaveValue('demo-casedata-001')
  })
})