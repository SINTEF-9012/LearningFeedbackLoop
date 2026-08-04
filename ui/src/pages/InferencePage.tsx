/**
 * InferencePage — route wrapper for the Inference view.
 */
import React from 'react'
import { InferencePanel } from '../components/InferencePanel'
import { FaultIndicatorPanel } from '../components/FaultIndicatorPanel'
import { LiveSignificanceChart } from '../components/LiveSignificanceChart'
import { LivePriorChart } from '../components/LivePriorChart'
import { ErrorBoundary } from '../components/ErrorBoundary'

export default function InferencePage() {
  return (
    <ErrorBoundary label="Inference View">
      <div className="panel">
        <FaultIndicatorPanel />
        <div className="hr" />
        <InferencePanel />
        <div className="hr" />
        <LiveSignificanceChart />
        <div className="hr" />
        <LivePriorChart />
      </div>
    </ErrorBoundary>
  )
}
