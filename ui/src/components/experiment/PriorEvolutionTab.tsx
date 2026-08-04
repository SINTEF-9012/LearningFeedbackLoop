/**
 * PriorEvolutionTab — Tab 2: Prior evolution SVG charts for eval/test phases.
 */
import React from 'react'
import { PriorEvolutionSVG } from '../charts'
import { HelpIcon } from '../Tooltip'
import type { ExperimentTabProps } from './types'

export function PriorEvolutionTab({ evalPhase, testPhase }: ExperimentTabProps) {
  if (!evalPhase && !testPhase) return null

  return (
    <>
      <div className="card" style={{ padding: '12px 16px 0' }}>
        <div className="small" style={{ color: 'var(--muted)' }}>Prior trajectories show how each pattern's Bayesian prior probability evolves as feedback is received. Rising priors indicate patterns being confirmed; falling priors indicate dismissals. <HelpIcon text="Each line is one pattern. The x-axis is sample index (time). The y-axis is the pattern's prior probability (0–1). Patterns that receive confirm feedback rise; dismiss feedback causes them to fall. Compare eval (with feedback) vs test (without) to see the feedback loop effect." /></div>
      </div>
      {[evalPhase, testPhase].filter(Boolean).map(phase => (
        <div key={phase!.phase} className="card" style={{ padding: 16 }}>
          <PriorEvolutionSVG
            evolution={phase!.prior_history || {}}
            title={`${phase!.phase === 'eval' ? 'Evaluation' : 'Test'}: ${phase!.operation} — Prior Trajectories`}
          />
        </div>
      ))}
    </>
  )
}
