/**
 * HarmonicsPage — dedicated live harmonic monitoring view.
 */
import React from 'react'

import { ErrorBoundary } from '../components/ErrorBoundary'
import { HarmonicsLivePanel } from '../components/HarmonicsLivePanel'
import { useAppContext } from '../contexts/AppContext'

export default function HarmonicsPage() {
  const ctx = useAppContext()
  const sessionInfo = (ctx.sessionInfoQuery.data as Record<string, unknown> | null) ?? null

  return (
    <ErrorBoundary label="Harmonics View">
      <div className="panel">
        <HarmonicsLivePanel sessionInfo={sessionInfo} />
      </div>
    </ErrorBoundary>
  )
}