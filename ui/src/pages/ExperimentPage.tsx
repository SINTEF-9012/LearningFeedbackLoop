/**
 * ExperimentPage — route wrapper for the Experiment Dashboard.
 */
import React from 'react'
import { ExperimentDashboard } from '../components/ExperimentDashboard'
import { ErrorBoundary } from '../components/ErrorBoundary'

export default function ExperimentPage() {
  return (
    <ErrorBoundary label="Experiment Dashboard">
      <div className="panel" style={{ overflow: 'auto' }}>
        <ExperimentDashboard />
      </div>
    </ErrorBoundary>
  )
}
