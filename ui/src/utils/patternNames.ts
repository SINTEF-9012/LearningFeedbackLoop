/**
 * Human-readable display names for internal pattern detector keys.
 *
 * Pattern keys from the backend follow the format PREFIX_DETAIL:VALUE, e.g.
 *   RATIO_Fx_Fy:>5    → "Force ratio Fx/Fy > 5"
 *   signature:modulated_tooth_passing_vibration → "Modulated tooth-passing vibration"
 *   SPECTRAL_PEAK_512Hz→ "Frequency spike at 512 Hz"
 *
 * This module provides:
 *  - humanPattern(key)              → one human-readable string
 *  - humanPatterns(keys)            → array of human-readable strings
 *  - patternCategory(keys)          → short category label (e.g. "Chatter")
 *  - patternOrigin(key)             → 'domain' | 'detected' | 'live'
 *  - patternDescription(key)        → detailed explanation string
 *  - groupPatternsByOrigin(keys)     → { domain: [...], detected: [...], live: [...] }
 *  - humanReason(reason)            → rewrite a scorer reason string
 */

// ── Prefix → display label mapping ──────────────────────────────────────────

const PREFIX_MAP: [RegExp, string][] = [
  // Modulation / imbalance family
  [/^CHATTER_DETECTED$/i, 'Modulated vibration observed'],
  [/^RATIO_Fx_Fy:>(\S+)$/i, 'Force ratio Fx/Fy exceeds $1'],
  [/^RATIO_Fx_Fy:(\S+)$/i, 'Force ratio Fx/Fy = $1'],

  // Spectral peaks
  [/^SPECTRAL_PEAK[_:](\d+\.?\d*)Hz$/i, 'Frequency spike at $1 Hz'],
  [/^SPECTRAL_PEAK[_:](.+)$/i, 'Frequency spike ($1)'],

  // Anomaly
  [/^ANOMALY:>(\S+)$/i, 'Anomaly score > $1'],
  [/^ANOMALY[_:](.+)$/i, 'Anomaly ($1)'],

  // Amplitude
  [/^amp:(\w+):loud$/i, 'High amplitude ($1)'],
  [/^amp:(\w+):normal$/i, 'Normal amplitude ($1)'],
  [/^amp:(\w+):quiet$/i, 'Low amplitude ($1)'],
  [/^amp:(\w+):(\w+)$/i, 'Amplitude $1 = $2'],

  // Frequency
  [/^freq:(\w+):high$/i, 'High frequency content ($1)'],
  [/^freq:(\w+):mid$/i, 'Mid-range frequency ($1)'],
  [/^freq:(\w+):low$/i, 'Low frequency content ($1)'],
  [/^freq:(\w+):(\w+)$/i, 'Frequency $1 = $2'],

  // External signals
  [/^EXTERNAL_(.+):(.+)$/i, 'External: $1 = $2'],

  // Signature / legacy fault aliases rendered as observations
  [/^signature:hf_burst_periodicity_loss$/i, 'High-frequency burst with periodicity loss'],
  [/^signature:modulated_tooth_passing_vibration$/i, 'Modulated tooth-passing vibration'],
  [/^signature:irregular_tooth_passing$/i, 'Irregular tooth-passing pattern'],
  [/^signature:spindle_shift_phase_change$/i, 'Spindle-order shift with phase change'],
  [/^fault:tool_breakage$/i, 'High-frequency burst with periodicity loss'],
  [/^fault:chatter$/i, 'Modulated tooth-passing vibration'],
  [/^fault:chip_adhesion$/i, 'Irregular tooth-passing pattern'],
  [/^fault:workpiece_slip$/i, 'Spindle-order shift with phase change'],
  [/^hypothesis:tool_breakage$/i, 'High-frequency burst with periodicity loss'],
  [/^hypothesis:chatter$/i, 'Modulated tooth-passing vibration'],
  [/^hypothesis:chip_adhesion$/i, 'Irregular tooth-passing pattern'],
  [/^hypothesis:workpiece_slip$/i, 'Spindle-order shift with phase change'],

  // Observable patterns (stoppage experiment)
  [/^SPINDLE_POWER_SURGE$/i, 'Spindle power surge'],
  [/^VIBRATION_REGIME_SHIFT$/i, 'Vibration regime shift'],
  [/^FEED_OVERRIDE_DROP$/i, 'Feed override drop'],
  [/^SENSOR_DECORRELATION$/i, 'Sensor decorrelation'],
  [/^SPINDLE_LOAD_RAMP$/i, 'Spindle load ramp'],
  [/^FEED_STALL$/i, 'Feed stall'],
  [/^CHATTER_ONSET$/i, 'Chatter onset'],
  [/^THERMAL_DRIFT$/i, 'Thermal drift'],
  [/^ANOMALY_HIGH_POWER$/i, 'Anomaly: high power'],
  [/^ANOMALY_HIGH_VIBRATION$/i, 'Anomaly: high vibration'],
  [/^ANOMALY_FEED_DEVIATION$/i, 'Anomaly: feed deviation'],

  // New domain patterns
  [/^POWER_ASYMMETRY$/i, 'Power asymmetry (X/Y axis)'],
  [/^ENERGY_ACCUMULATION$/i, 'Energy accumulation ramp'],

  // Time-series derived patterns
  [/^VARIANCE_EXPLOSION$/i, 'Variance explosion'],
  [/^TREND_REVERSAL$/i, 'Trend reversal'],
  [/^AUTOCORRELATION_BREAK$/i, 'Autocorrelation break'],

  // Discovered & suppressed patterns (from PatternDiscovery engine)
  [/^discovered:(.+)$/i, 'Discovered: $1'],
  [/^suppressed:(.+)$/i, 'Suppressed: $1'],

  // Spectral fault patterns
  [/^spectral:hf_burst$/i, 'High-frequency energy burst'],
  [/^spectral:modulated_vibration$/i, 'Modulated vibration'],
  [/^spectral:irregular_tooth_passing$/i, 'Irregular tooth-passing pattern'],
  [/^spectral:spindle_freq_shift$/i, 'Spindle frequency shift'],
  [/^spectral:tp_harmonic_(\d+)x$/i, 'Tooth-passing harmonic $1\u00d7'],

  // Temporal fault patterns
  [/^temporal:periodicity_loss$/i, 'Loss of periodicity'],
  [/^temporal:amplitude_modulation$/i, 'Amplitude modulation'],
  [/^temporal:severity_growth$/i, 'Severity growth'],
  [/^temporal:phase_shift$/i, 'Phase shift at spindle frequency'],
  [/^temporal:impulsive_burst$/i, 'Impulsive burst'],
]

// ── Category mapping (first match wins) ─────────────────────────────────────

const CATEGORY_MAP: [RegExp, string][] = [
  [/signature:modulated_tooth_passing_vibration|chatter|RATIO_Fx_Fy/i, 'Vibration Modulation'],
  [/anomaly/i, 'Anomaly'],
  [/spectral|freq:/i, 'Frequency'],
  [/amp:/i, 'Amplitude'],
  [/external/i, 'External'],
  [/signature:hf_burst_periodicity_loss|fault:tool_breakage|hypothesis:tool_breakage|hf_burst|periodicity_loss|impulsive_burst/i, 'Impulsive Regime Change'],
  [/SPINDLE_POWER_SURGE|VIBRATION_REGIME_SHIFT|FEED_OVERRIDE_DROP|SENSOR_DECORRELATION/i, 'Process Anomaly'],
  [/SPINDLE_LOAD_RAMP|FEED_STALL/i, 'Process Anomaly'],
  [/POWER_ASYMMETRY|ENERGY_ACCUMULATION/i, 'Process Anomaly'],
  [/VARIANCE_EXPLOSION|TREND_REVERSAL|AUTOCORRELATION_BREAK/i, 'Signal Instability'],
  [/^discovered:/i, 'Discovered'],
  [/^suppressed:/i, 'Suppressed'],
  [/signature:irregular_tooth_passing|fault:chip_adhesion|hypothesis:chip_adhesion|irregular_tooth/i, 'Tooth-Passing Irregularity'],
  [/signature:spindle_shift_phase_change|fault:workpiece_slip|hypothesis:workpiece_slip|spindle_freq_shift|phase_shift/i, 'Spindle-Order Shift'],
  [/breakage|wear/i, 'Tool condition (legacy)'],
]

// ── Reason rewrite (scorer emits things like "Pattern RATIO_Fx_Fy:>5 matches chatter rule") ─

const REASON_REWRITES: [RegExp, string][] = [
  [/Pattern\s+([\w:>.<]+)\s+matches chatter rule/i, 'Vibration-modulation pattern triggered'],
  [/Pattern\s+([\w:>.<]+)\s+matches/i, 'Pattern match: $1'],
  [/Force ratio\s+Fx\/Fy\s+exceeds\s+threshold/i, 'Force ratio exceeded threshold — cross-axis imbalance observed'],
  [/Classical model alert.*score\s*=?\s*([\d.]+)/i, 'External ML model alert (score $1)'],
  [/Anomaly deviation.*z\s*=?\s*([\d.]+)/i, 'Statistical anomaly (z = $1)'],
  [/Historical prior boost.*(\+[\d.]+)/i, 'Prior boost from operator feedback ($1)'],
  [/signature:hf_burst_periodicity_loss/i, 'High-frequency burst with periodicity loss pattern'],
  [/signature:modulated_tooth_passing_vibration/i, 'Modulated tooth-passing vibration pattern'],
  [/signature:irregular_tooth_passing/i, 'Irregular tooth-passing pattern'],
  [/signature:spindle_shift_phase_change/i, 'Spindle-order shift with phase change pattern'],
  [/fault:tool_breakage/i, 'High-frequency burst with periodicity loss pattern'],
  [/fault:chatter/i, 'Modulated tooth-passing vibration pattern'],
  [/fault:chip_adhesion/i, 'Irregular tooth-passing pattern'],
  [/fault:workpiece_slip/i, 'Spindle-order shift with phase change pattern'],
  [/hypothesis:tool_breakage/i, 'High-frequency burst with periodicity loss pattern'],
  [/hypothesis:chatter/i, 'Modulated tooth-passing vibration pattern'],
  [/hypothesis:chip_adhesion/i, 'Irregular tooth-passing pattern'],
  [/hypothesis:workpiece_slip/i, 'Spindle-order shift with phase change pattern'],
  [/FAULT_TOOL_BREAKAGE/i, 'Legacy tool-condition indicator'],
  [/FAULT_CHATTER/i, 'Legacy vibration-modulation indicator'],
  [/FAULT_CHIP_ADHESION/i, 'Legacy tooth-passing irregularity indicator'],
  [/FAULT_WORKPIECE_SLIP/i, 'Legacy spindle-shift indicator'],
]

// ── Public API ──────────────────────────────────────────────────────────────
// ---- Origin classification ------------------------------------------------
// Domain-knowledge patterns: defined from engineering understanding of CNC
// machining physics.  These exist before any data is collected.
const DOMAIN_PATTERNS = new Set([
  // Builtin (derived from expert domain knowledge of CNC sensor physics)
  'SPINDLE_POWER_SURGE',
  'VIBRATION_REGIME_SHIFT',
  'FEED_OVERRIDE_DROP',
  'SENSOR_DECORRELATION',
  // Domain-derived (from fault-type models in domain config)
  'SPINDLE_LOAD_RAMP',
  'FEED_STALL',
  // Domain-derived (from physical sensor relationships)
  'POWER_ASYMMETRY',
  'ENERGY_ACCUMULATION',
])

// Detected-during-operation patterns: anomaly detectors that fire based on
// statistical deviation from the normal baseline learned during training.
const DETECTED_PATTERNS = new Set([
  // Time-series derived (from statistical signal properties)
  'VARIANCE_EXPLOSION',
  'TREND_REVERSAL',
  'AUTOCORRELATION_BREAK',
])

// Detailed descriptions explaining what each pattern means for the operator
const PATTERN_DESCRIPTIONS: Record<string, string> = {
  // Domain-knowledge patterns
  SPINDLE_POWER_SURGE: 'Spindle or Y-axis power consumption jumped beyond the normal 95th percentile. May indicate sudden tool loading, workpiece hardness variation, or tool engagement change.',
  VIBRATION_REGIME_SHIFT: 'Vibration severity or chatter frequency shifted from the normal operating envelope. Could indicate tool wear progression, resonance onset, or workpiece clamping change.',
  FEED_OVERRIDE_DROP: 'Feed override dropped or entered an abnormally low band. Often occurs when the operator or machine controller reduces feed rate in response to perceived cutting problems.',
  SENSOR_DECORRELATION: 'The correlation between spindle power and vibration sensors decoupled from normal. When these signals stop tracking together, it can indicate a fundamental change in cutting dynamics.',
  SPINDLE_LOAD_RAMP: 'Spindle load is gradually increasing over consecutive windows. Typical of progressive tool wear or chip packing.',
  FEED_STALL: 'Feed rate dropped to near-zero while the spindle is still running. May indicate a control intervention or mechanical resistance.',
  CHATTER_ONSET: 'Legacy concept label for early chatter-like behavior. This is not a canonical pattern in the current detector set; prefer direct detector-backed patterns such as VIBRATION_REGIME_SHIFT or fault-specific chatter outputs.',
  THERMAL_DRIFT: 'Legacy concept label for slow thermal baseline shift. This is not a canonical pattern in the current detector set; when relevant, explain it through sensor context and model explanations rather than a first-class pattern prior.',
  // Legacy generic anomaly labels kept only for backward-compatible explanation
  ANOMALY_HIGH_POWER: 'Legacy generic anomaly label. In the current pipeline, unusually high power should be explained through unsupervised model outputs and canonical detector-backed patterns such as SPINDLE_POWER_SURGE.',
  ANOMALY_HIGH_VIBRATION: 'Legacy generic anomaly label. In the current pipeline, unusually high vibration should be explained through unsupervised model outputs and canonical detector-backed patterns such as VIBRATION_REGIME_SHIFT or spectral fault indicators.',
  ANOMALY_FEED_DEVIATION: 'Legacy generic anomaly label. In the current pipeline, unusual feed behavior should be explained through unsupervised model outputs and direct feed-related patterns rather than a standalone generic pattern.',
  // Domain-derived (new)
  POWER_ASYMMETRY: 'X vs Y axis power diverges — uneven cutting load that may indicate chatter risk or tool deflection.',
  ENERGY_ACCUMULATION: 'Total energy consumption is ramping faster than normal baseline — possible progressive tool wear.',
  // Time-series derived
  VARIANCE_EXPLOSION: 'Vibration or power standard deviation jumped >3× its own mean — sudden instability in a previously stable signal.',
  TREND_REVERSAL: 'Spindle power slope and delta-mean have opposite signs — a regime change is occurring.',
  AUTOCORRELATION_BREAK: 'Vibration IQR/range exceeds 0.7 — signal has become non-stationary, autocorrelation structure has broken down.',
}

export type PatternOrigin = 'domain' | 'detected' | 'live'

/**
 * Classify a pattern key by its origin.
 * - 'domain': defined from engineering domain knowledge (exists before any data)
 * - 'detected': discovered by statistical anomaly detectors during operation
 * - 'live': learned dynamically from operator feedback (discovered: prefix)
 */
export function patternOrigin(key: string): PatternOrigin {
  const k = (key || '').trim()
  if (DOMAIN_PATTERNS.has(k)) return 'domain'
  if (DETECTED_PATTERNS.has(k)) return 'detected'
  if (k.startsWith('discovered:') || k.startsWith('suppressed:')) return 'live'
  // Fallback: check if it looks like a fault-type or spectral/temporal pattern
  if (/^fault:|^hypothesis:|^signature:|^spectral:|^temporal:/.test(k)) return 'domain'
  return 'detected'
}

/**
 * Return a detailed human-readable explanation of what the pattern means.
 * Falls back to the short humanPattern() name if no description exists.
 */
export function patternDescription(key: string): string {
  return PATTERN_DESCRIPTIONS[key] || humanPattern(key)
}

/**
 * Group an array of pattern keys by origin.
 */
export function groupPatternsByOrigin(keys: string[]): { domain: string[]; detected: string[]; live: string[] } {
  const result = { domain: [] as string[], detected: [] as string[], live: [] as string[] }
  for (const k of keys || []) {
    result[patternOrigin(k)].push(k)
  }
  return result
}
export function humanPattern(key: string): string {
  const k = (key || '').trim()
  if (!k) return ''
  for (const [re, repl] of PREFIX_MAP) {
    if (re.test(k)) return k.replace(re, repl)
  }
  // Fallback: replace underscores with spaces, title-case first word
  const fallback = k.replace(/_/g, ' ').replace(/:>/g, ' > ').replace(/:/g, ': ')
  return fallback.charAt(0).toUpperCase() + fallback.slice(1)
}

export function humanPatterns(keys: string[]): string[] {
  return (keys || []).map(humanPattern).filter(Boolean)
}

export function patternCategory(keys: string[]): string {
  const arr = Array.isArray(keys) ? keys : []
  for (const k of arr) {
    for (const [re, label] of CATEGORY_MAP) {
      if (re.test(k)) return label
    }
  }
  return ''
}

export function humanReason(reason: string): string {
  const r = (reason || '').trim()
  if (!r) return ''
  const significantPattern = r.match(/^Significant pattern:\s*(.+)$/i)
  if (significantPattern) {
    return `Pattern triggered: ${humanPattern(significantPattern[1].trim())}`
  }
  const criticalPatternType = r.match(/^Critical pattern type:\s*(.+)$/i)
  if (criticalPatternType) {
    const kind = criticalPatternType[1].trim().replace(/_/g, ' ')
    return `Critical pattern family: ${kind.charAt(0).toUpperCase()}${kind.slice(1)}`
  }
  const historicalPrior = r.match(/^Historical significance:\s*(.+?)\s*\(prior=([\d.]+)\)$/i)
  if (historicalPrior) {
    return `Historical prior: ${humanPattern(historicalPrior[1].trim())} (${historicalPrior[2]})`
  }
  for (const [re, repl] of REASON_REWRITES) {
    if (re.test(r)) return r.replace(re, repl)
  }
  // Fallback: pass through but strip internal key patterns
  return r
    .replace(/\b[a-zA-Z_]+(?::[a-zA-Z0-9_.><-]+)+\b/g, (m) => humanPattern(m))
    .replace(/_/g, ' ')
}
