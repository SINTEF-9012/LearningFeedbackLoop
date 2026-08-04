/**
 * CapturePanel — capture window + label/note + create-memory button.
 *
 * Extracted from the `captureSel &&` conditional block in App.tsx.
 */
import React, { useState } from 'react'
import { api, baseUrl } from '../api/http'
import { useStreamStore } from '../state/streamStore'
import type { UseQueryResult } from '@tanstack/react-query'
import type { ListMemoriesResponse, PriorsResponse } from '../contexts/AppContext'

interface Props {
  captureSel: { i0: number; i1: number }
  onClose: () => void
  memoriesQuery: UseQueryResult<ListMemoriesResponse>
  priorsQuery: UseQueryResult<PriorsResponse>
}

export function CapturePanel({ captureSel, onClose, memoriesQuery, priorsQuery }: Props) {
  const streamSessionId = useStreamStore((s) => s.sessionId)
  const getWindowSamples = useStreamStore((s) => s.getWindowSamples)

  const [note, setNote] = useState('')
  const [label, setLabel] = useState('')

  const doCapture = async () => {
    const { i0, i1 } = captureSel
    const win = getWindowSamples(i0, i1)
    const { channels, fs, samples } = win

    await api('/agent/memory/capture', 'POST', {
      session_id: streamSessionId,
      window: { i0: win.i0, i1: win.i1, fs },
      channels,
      samples,
      annotation_text: note,
      label: label || null,
      tags: ['operator_capture'],
      created_by: 'operator',
      compute_metrics: true,
      compute_patterns: true,
      include_feature_vector: true,
      metadata: {},
    })

    onClose()
    setNote('')
    setLabel('')
    await memoriesQuery.refetch()
    await priorsQuery.refetch()
  }

  return (
    <div className="panel" style={{ marginTop: 12 }}>
      <div style={{ fontWeight: 700 }}>Capture window — create memory</div>
      <div className="small">
        selection: i0={captureSel.i0} i1={captureSel.i1} (len {captureSel.i1 - captureSel.i0})
      </div>

      <div className="row" style={{ marginTop: 8 }}>
        <div>
          <div className="small">Label (optional)</div>
          <input value={label} onChange={(e) => setLabel(e.target.value)} placeholder="e.g., vibration_modulation_review" />
        </div>
        <div>
          <div className="small">Annotation</div>
          <input value={note} onChange={(e) => setNote(e.target.value)} placeholder="Operator note" />
        </div>
      </div>

      <div className="hrow" style={{ marginTop: 10 }}>
        <button className="primary" onClick={doCapture}>
          Create memory
        </button>
        <button
          onClick={() => {
            onClose()
            setNote('')
            setLabel('')
          }}
        >
          Cancel
        </button>
        <div className="small">POST {baseUrl()}/agent/memory/capture</div>
      </div>
    </div>
  )
}
