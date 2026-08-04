/**
 * Design tokens — Agent K Phase 2 (2026-04-24).
 *
 * Centralised design values so future components can opt out of
 * inline styles. Existing components remain untouched; new work
 * should reach for these tokens instead of hardcoding values.
 */

export const colors = {
  bg: '#0f1115',
  surface: '#171a21',
  surfaceAlt: '#1f232d',
  border: '#2b3140',
  text: '#e5e7eb',
  textMuted: '#8b93a7',
  textDim: '#6b7385',
  accent: '#4c9aff',
  accentDim: '#2d5fa8',
  good: '#57c785',
  warn: '#f0b429',
  bad: '#e0645e',
  chart: ['#4c9aff', '#57c785', '#f0b429', '#b76ed8', '#e0645e', '#40c0c0'],
} as const

export const spacing = {
  xs: 4,
  sm: 8,
  md: 12,
  lg: 16,
  xl: 24,
  xxl: 32,
} as const

export const fontSize = {
  xs: 11,
  sm: 12,
  base: 13,
  md: 14,
  lg: 16,
  xl: 20,
  xxl: 24,
} as const

export const radii = {
  sm: 4,
  md: 6,
  lg: 10,
} as const

export const shadows = {
  panel: '0 1px 2px rgba(0, 0, 0, 0.25)',
  raised: '0 4px 12px rgba(0, 0, 0, 0.35)',
} as const

export const zIndex = {
  base: 0,
  overlay: 10,
  modal: 100,
  toast: 200,
} as const

export type ChartColor = (typeof colors.chart)[number]
