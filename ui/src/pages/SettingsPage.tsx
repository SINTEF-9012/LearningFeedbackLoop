import React from 'react'

import { ErrorBoundary } from '../components/ErrorBoundary'
import { useAppContext } from '../contexts/AppContext'

const visibleTabsInOperatorMode = ['Monitoring', 'Detailed', 'Inference', 'Settings']
const hiddenTabsInOperatorMode = ['Experiment', 'Dataset', 'Documents', 'Development', 'Knowledge Graph', 'Digital Twin', 'Learnings']

function infoCardStyle(): React.CSSProperties {
  return {
    border: '1px solid var(--border)',
    borderRadius: 10,
    padding: '12px 14px',
    background: 'linear-gradient(180deg, var(--panel) 0%, var(--panel2) 100%)',
  }
}

export default function SettingsPage() {
  const ctx = useAppContext()

  return (
    <ErrorBoundary label="Settings">
      <div className="panel">
        <div className="hrow" style={{ justifyContent: 'space-between', alignItems: 'flex-start', gap: 16, flexWrap: 'wrap' }}>
          <div>
            <h2 style={{ margin: 0 }}>Settings</h2>
            <div className="small" style={{ marginTop: 6, maxWidth: 760 }}>
              Operator mode updates the navigation live for demos. It hides advanced tabs but keeps the underlying routes and features available when you turn the mode back off.
            </div>
          </div>
          <div className="small" style={{ color: 'var(--muted)' }}>Saved in this browser.</div>
        </div>

        <div style={{ display: 'grid', gap: 14, marginTop: 18 }}>
          <label
            style={{
              ...infoCardStyle(),
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: 16,
              cursor: 'pointer',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'flex-start', gap: 12 }}>
              <input
                type="checkbox"
                checked={ctx.operatorMode}
                onChange={(event) => ctx.setOperatorMode(event.target.checked)}
                style={{ marginTop: 3 }}
              />
              <div>
                <div style={{ fontWeight: 700 }}>Operator mode</div>
                <div className="small" style={{ marginTop: 4 }}>
                  Keep the demo shell focused on the core operator surfaces and hide the advanced tabs until this is turned off.
                </div>
              </div>
            </div>
            <div className="small" style={{ color: ctx.operatorMode ? 'var(--ok)' : 'var(--muted)', whiteSpace: 'nowrap' }}>
              {ctx.operatorMode ? 'Enabled' : 'Disabled'}
            </div>
          </label>

          <div style={infoCardStyle()}>
            <div style={{ fontWeight: 700 }}>Shown while operator mode is enabled</div>
            <div className="small" style={{ marginTop: 6 }}>
              {visibleTabsInOperatorMode.join(' · ')}
            </div>
          </div>

          <div style={infoCardStyle()}>
            <div style={{ fontWeight: 700 }}>Hidden from the top navigation</div>
            <div className="small" style={{ marginTop: 6 }}>
              {hiddenTabsInOperatorMode.join(' · ')}
            </div>
          </div>
        </div>
      </div>
    </ErrorBoundary>
  )
}