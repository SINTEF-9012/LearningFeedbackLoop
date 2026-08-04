import React, { Component, type ErrorInfo, type ReactNode } from 'react'

interface Props {
  children: ReactNode
  /** Label shown in the fallback UI (e.g. "Operator View") */
  label?: string
}

interface State {
  hasError: boolean
  error: Error | null
}

/**
 * Generic React error boundary (P3 fix).
 *
 * Catches render-time exceptions in child components and shows a
 * recoverable fallback instead of crashing the entire app.
 */
export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // eslint-disable-next-line no-console
    console.error(`[ErrorBoundary${this.props.label ? `: ${this.props.label}` : ''}]`, error, info.componentStack)
  }

  handleRetry = () => {
    this.setState({ hasError: false, error: null })
  }

  render() {
    if (this.state.hasError) {
      return (
        <div
          className="panel"
          style={{
            margin: 16,
            padding: 24,
            borderColor: 'rgba(247, 118, 142, 0.5)',
            textAlign: 'center',
          }}
        >
          <div style={{ fontSize: 18, fontWeight: 700, marginBottom: 8 }}>
            ⚠️ {this.props.label ?? 'Component'} crashed
          </div>
          <div className="small" style={{ marginBottom: 12, color: 'var(--muted)' }}>
            {this.state.error?.message ?? 'An unexpected error occurred.'}
          </div>
          <button className="primary" onClick={this.handleRetry}>
            Retry
          </button>
        </div>
      )
    }

    return this.props.children
  }
}

export default ErrorBoundary
