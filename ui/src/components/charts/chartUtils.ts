/**
 * Shared constants and utilities for experiment chart components.
 */

/** Standard color palette for chart lines/areas. */
export const PAL = ['#7aa2f7','#bb9af7','#9ece6a','#e0af68','#f7768e','#73daca','#ff9e64','#2ac3de']

/** Format a number as a percentage string (e.g. 0.123 → "12.3%"). */
export function pct(n: number | null | undefined): string {
  return n != null && Number.isFinite(n) ? `${(n * 100).toFixed(1)}%` : '\u2013'
}

/** Format a number to 3 decimal places. */
export function f3(n: number | null | undefined): string {
  return n != null && Number.isFinite(n) ? n.toFixed(3) : '\u2013'
}

/** Clamp a value between lo and hi. */
export function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v))
}

/** Coerce any value to a finite number (0 if not). */
export function num(v: any): number {
  const n = Number(v)
  return Number.isFinite(n) ? n : 0
}
