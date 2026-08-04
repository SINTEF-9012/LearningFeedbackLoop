/**
 * ScoreAnalysisTab — Tab 1: Score histograms for eval/test phases.
 */
import React from 'react'
import { ScoreHistogram } from '../charts'
import { HelpIcon } from '../Tooltip'
import type { ExperimentTabProps } from './types'

export function ScoreAnalysisTab({ evalPhase, testPhase }: ExperimentTabProps) {
  if (!evalPhase && !testPhase) return null

  return (
    <>
      {[evalPhase, testPhase].filter(Boolean).map(phase => (
        <div key={phase!.phase} className="card" style={{ padding: 16 }}>
          <div style={{ fontWeight: 700, marginBottom: 12 }}>
            {phase!.phase === 'eval' ? 'Evaluation' : 'Test'} Phase — {phase!.operation}
            <HelpIcon text="Score histogram showing the distribution of significance scores for positive (pre-stoppage) and negative (normal) samples. The vertical line marks the classification threshold. Good separation = low overlap between the two distributions." />
          </div>
          <ScoreHistogram
            positive={phase!.scores_positive}
            negative={phase!.scores_negative}
            threshold={phase!.threshold}
            title="Significance Score Distribution"
          />
          <div className="small" style={{ marginTop: 8, display: 'flex', gap: 16 }}>
            <span>Threshold: {phase!.threshold != null ? phase!.threshold.toFixed(3) : '–'}</span>
            <span>Adapted: {phase!.adapted_threshold != null ? phase!.adapted_threshold.toFixed(3) : '–'} <HelpIcon text="Threshold after adaptation by the feedback loop. Starts at the base threshold and is adjusted based on false-positive/false-negative feedback." position="top" /></span>
            <span>Flipped: {phase!.n_predictions_flipped ?? '–'} <HelpIcon text="Number of samples whose prediction changed due to feedback-driven threshold or prior adjustments." position="top" /></span>
            <span>Retrains: {phase!.n_model_retrains ?? '–'} <HelpIcon text="Number of times the anomaly model was retrained during this phase due to accumulated feedback." position="top" /></span>
          </div>
        </div>
      ))}
    </>
  )
}
