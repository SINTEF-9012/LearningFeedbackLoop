/**
 * SinditPage — route wrapper for the SINDIT Digital Twin view.
 */
import React from 'react'
import { SinditView } from '../components/SinditView'
import { ErrorBoundary } from '../components/ErrorBoundary'

export default function SinditPage() {
  return (
    <ErrorBoundary label="SINDIT Digital Twin">
      <div className="panel" style={{ overflow: 'auto' }}>
        <SinditView />
      </div>
    </ErrorBoundary>
  )
}
