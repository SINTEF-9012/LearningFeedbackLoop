/**
 * Harmonics dashboard smoke test.
 *
 * Exercises the Experiment Dashboard Harmonics tab against mocked API
 * responses shaped like the real harmonic status and feedback retrain
 * routes. This covers the new checkpoint controls and feedback bucket
 * flow without requiring a live backend fixture server.
 */
import { test, expect } from '@playwright/test'

function harmonicStatusPayload(dataset: string) {
  const isPair = dataset === 'pair_raw' || dataset === 'pair_casedata' || dataset === 'pair_lfl'
  return {
    available: true,
    torch_installed: true,
    model_loaded: true,
    dataset_name: dataset,
    scorer_kind: isPair ? 'pair' : 'context',
    n_harm_features: isPair ? 16 : 8,
    n_params: isPair ? 5 : 3,
    harmonic_mode: 'pre_extracted',
    cnn_window: isPair ? 32 : 16,
    trained_at: '2026-05-31T09:00:00Z',
    training_metrics: {
      best_val_loss: isPair ? 0.5418 : 0.182,
      best_val_acc: isPair ? 0.8 : 0.91,
      random_seed: isPair ? 0 : 7,
    },
    model_save_path: `data/models/harmonic_${dataset}.pt`,
    model_path_exists: true,
    checkpoint_statuses: {
      [dataset]: {
        available: true,
        torch_installed: true,
        model_loaded: true,
        dataset_name: dataset,
        scorer_kind: isPair ? 'pair' : 'context',
        harmonic_mode: 'pre_extracted',
        cnn_window: isPair ? 32 : 16,
        trained_at: '2026-05-31T09:00:00Z',
        model_save_path: `data/models/harmonic_${dataset}.pt`,
        model_path_exists: true,
      },
    },
  }
}

function retrainStatusPayload(dataset: string, scorerKind: string) {
  return {
    total_feedback: 12,
    active_bucket: `${scorerKind}:${dataset}`,
    buckets: {
      [`${scorerKind}:${dataset}`]: {
        dataset_name: dataset,
        scorer_kind: scorerKind,
        total_feedback: 12,
        since_last_retrain: 12,
        retrain_threshold: 10,
        buffer_size: 12,
        confirmed_in_buffer: 7,
        dismissed_in_buffer: 5,
        should_retrain: true,
        retrain_count: 1,
        last_retrain: `Harmonic model retrained for ${scorerKind}:${dataset}`,
        model_save_path: `data/models/harmonic_${dataset}.pt`,
      },
    },
  }
}

test.describe('Harmonics dashboard', () => {
  test('renders feedback retrain controls and posts the selected pair bucket', async ({ page }) => {
    await page.route('http://localhost:8000/agent/memory/experiment/runs', async route => {
      await route.fulfill({ json: { runs: [] } })
    })

    await page.route('http://localhost:8000/harmonic/train/result', async route => {
      await route.fulfill({ json: { status: 'completed', runtime_checkpoint_activated: false } })
    })

    await page.route('http://localhost:8000/harmonic/status**', async route => {
      const url = new URL(route.request().url())
      const dataset = url.searchParams.get('dataset') || 'casedata'
      await route.fulfill({ json: harmonicStatusPayload(dataset) })
    })

    await page.route('http://localhost:8000/harmonic/retrain/status**', async route => {
      const url = new URL(route.request().url())
      const dataset = url.searchParams.get('dataset') || 'casedata'
      const scorerKind = url.searchParams.get('scorer_kind') || 'context'
      await route.fulfill({ json: retrainStatusPayload(dataset, scorerKind) })
    })

    await page.route('http://localhost:8000/harmonic/retrain', async route => {
      const payload = JSON.parse(route.request().postData() || '{}')
      expect(payload).toMatchObject({
        dataset: 'pair_lfl',
        scorer_kind: 'pair',
        random_seed: 17,
        checkpoint_suffix: 'seed17',
      })
      await route.fulfill({
        json: {
          success: true,
          message: 'Harmonic model retrained for pair:pair_lfl',
          bucket_key: 'pair:pair_lfl',
          dataset_name: 'pair_lfl',
          scorer_kind: 'pair',
          n_samples_used: 12,
          n_confirmed: 7,
          n_dismissed: 5,
          model_path: 'data/models/harmonic_pair_lfl.seed17.pt',
          best_val_loss: 0.5418,
          best_val_acc: 0.8,
          duration_s: 6.5,
          training_result: {
            success: true,
            model_path: 'data/models/harmonic_pair_lfl.seed17.pt',
          },
        },
      })
    })

    await page.route('http://localhost:8000/harmonic/dev/seed-feedback', async route => {
      const payload = JSON.parse(route.request().postData() || '{}')
      expect(payload).toMatchObject({
        dataset: 'pair_lfl',
        scorer_kind: 'pair',
        confirmed: 12,
        dismissed: 8,
        clear_existing: true,
      })
      await route.fulfill({
        json: {
          enabled: true,
          bucket_key: 'pair:pair_lfl',
          dataset_name: 'pair_lfl',
          scorer_kind: 'pair',
          added_confirmed: 12,
          added_dismissed: 8,
          cleared_existing: true,
          removed_buffer_size: 0,
          removed_total_feedback: 0,
          total_feedback: 20,
          buffer_size: 20,
          confirmed_in_buffer: 12,
          dismissed_in_buffer: 8,
          should_retrain: true,
        },
      })
    })

    await page.goto('/#/experiment')
    await expect(page.getByRole('heading', { name: /experiment dashboard/i })).toBeVisible()

    await page.locator('.expTabs').getByRole('button', { name: /harmonics/i }).click()
    await expect(page.getByText('Feedback Retrain', { exact: true })).toBeVisible()

    await page.getByLabel('Dataset').last().selectOption('pair_lfl')
    await expect(page.getByText('Bucket pair:pair_lfl')).toBeVisible()
    await expect(page.getByText(/feedback bucket meets the automatic retrain threshold/i)).toBeVisible()

    await page.getByRole('button', { name: /seed demo feedback bucket/i }).click()
    await expect(page.getByText('Seeded pair:pair_lfl with 12 confirmed / 8 dismissed samples.', { exact: true })).toBeVisible()

    await page.getByLabel('Random seed').fill('17')
    await page.getByLabel('Checkpoint suffix').fill('seed17')
    await page.getByRole('button', { name: /retrain from feedback/i }).click()

    await expect(page.getByText('Harmonic model retrained for pair:pair_lfl', { exact: true })).toBeVisible()
    await expect(page.getByText('Last Feedback Retrain', { exact: true })).toBeVisible()
    await expect(page.locator('pre').filter({ hasText: 'harmonic_pair_lfl.seed17.pt' })).toBeVisible()
  })
})