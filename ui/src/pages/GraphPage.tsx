/**
 * GraphPage — route wrapper for the Knowledge Graph view.
 */
import React from 'react'
import { KnowledgeGraphView } from '../components/KnowledgeGraphView'
import { ErrorBoundary } from '../components/ErrorBoundary'

export default function GraphPage() {
  return (
    <ErrorBoundary label="Knowledge Graph">
      <div className="panel" style={{ overflow: 'auto' }}>
        <KnowledgeGraphView />
      </div>
    </ErrorBoundary>
  )
}
