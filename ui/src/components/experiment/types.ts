/**
 * Shared prop types for experiment tab components.
 */
import type { UseQueryResult } from '@tanstack/react-query'
import type {
  RunSummary,
  PhaseDetail,
  EvaluationDetail,
} from '../../state/experimentStore'

export interface ExperimentTabProps {
  effectiveRunId: string
  selectedRun?: RunSummary
  runs: RunSummary[]
  setSelectedRunId: (id: string) => void
  detail: EvaluationDetail | null
  evalPhase: PhaseDetail | null
  testPhase: PhaseDetail | null
  fullResultsQ: UseQueryResult<unknown>
  evaluateQ: UseQueryResult<EvaluationDetail>
}
