export const DISPLAY_SPINDLE_COMMAND_MAX_RPM = 100_000
export const DEFAULT_VISIBLE_PLOT_CHANNEL_LIMIT = 6

type PlotChannelRole =
  | 'spindle_speed_actual'
  | 'feed_rate_actual'
  | 'spindle_power'
  | 'active_power'
  | 'vibration'
  | 'feed_override'
  | 'spindle_override'
  | 'axis_power_y'
  | 'axis_power_x'
  | 'axis_power_z'
  | 'power_factor'
  | 'spindle_speed'
  | 'feed_rate'
  | 'power_other'
  | 'other'

const PLOT_CHANNEL_ROLE_PRIORITY: PlotChannelRole[] = [
  'spindle_speed_actual',
  'feed_rate_actual',
  'spindle_power',
  'active_power',
  'vibration',
  'feed_override',
  'spindle_override',
  'axis_power_y',
  'axis_power_x',
  'axis_power_z',
  'power_factor',
  'spindle_speed',
  'feed_rate',
  'power_other',
  'other',
]

function normalizePlotChannel(channel: string): string {
  return channel.trim().toLowerCase().replace(/[^a-z0-9]+/g, '_')
}

function inferPlotChannelRole(channel: string): PlotChannelRole {
  const normalized = normalizePlotChannel(channel)

  if (
    normalized.includes('spindle_speed_actual')
    || normalized.includes('spindle_speed_actrev')
    || (normalized.includes('spindle') && normalized.includes('speed') && (normalized.includes('actual') || normalized.includes('actrev')))
  ) {
    return 'spindle_speed_actual'
  }

  if (
    normalized.includes('power_spindle')
    || normalized.includes('spindle_power')
    || normalized.includes('spindle_load')
    || normalized.includes('load_spindle')
    || normalized.includes('spindle_current')
  ) {
    return 'spindle_power'
  }

  if (
    normalized.includes('axis_feedrate_actual')
    || normalized.includes('feed_rate_actual')
    || normalized.includes('feedrate_actual')
    || normalized === 'axis_feedrate'
    || normalized === 'feed_rate'
  ) {
    return 'feed_rate_actual'
  }

  if (normalized.includes('power_active') || normalized.includes('active_power') || normalized.includes('main_power') || normalized.includes('total_power')) {
    return 'active_power'
  }

  if (normalized.includes('override') && normalized.includes('feed')) {
    return 'feed_override'
  }

  if (normalized.includes('override') && normalized.includes('spindle')) {
    return 'spindle_override'
  }

  if (normalized.includes('vibration') || normalized.includes('chatter') || normalized.includes('severity') || normalized.includes('accel')) {
    return 'vibration'
  }

  if (normalized.includes('axis_power_y') || normalized === 'power_y') {
    return 'axis_power_y'
  }

  if (normalized.includes('axis_power_x') || normalized === 'power_x') {
    return 'axis_power_x'
  }

  if (normalized.includes('axis_power_z') || normalized === 'power_z') {
    return 'axis_power_z'
  }

  if (normalized.includes('power_factor')) {
    return 'power_factor'
  }

  if (normalized.includes('spindle') && normalized.includes('speed')) {
    return 'spindle_speed'
  }

  if (normalized.includes('feedrate') || normalized.includes('feed_rate')) {
    return 'feed_rate'
  }

  if (normalized.includes('power')) {
    return 'power_other'
  }

  return 'other'
}

export function sortPlotChannelsByImportance(channels: string[]): string[] {
  const roleRank = new Map(PLOT_CHANNEL_ROLE_PRIORITY.map((role, index) => [role, index]))

  return channels
    .map((channel, index) => ({
      channel,
      index,
      normalized: normalizePlotChannel(channel),
      rank: roleRank.get(inferPlotChannelRole(channel)) ?? PLOT_CHANNEL_ROLE_PRIORITY.length,
    }))
    .sort((left, right) => (
      left.rank - right.rank
      || left.normalized.localeCompare(right.normalized)
      || left.index - right.index
    ))
    .map((entry) => entry.channel)
}

export function isAutoHiddenSpindleCommandChannel(channel: string): boolean {
  const normalized = channel.toLowerCase()
  return normalized.includes('spindle') && (
    normalized.includes('commanded')
    || normalized.includes('programed')
    || normalized.includes('programmed')
  )
}

export function hasImplausiblyHighSpindleCommand(values: number[]): boolean {
  return values.some((value) => Number.isFinite(value) && Math.abs(value) > DISPLAY_SPINDLE_COMMAND_MAX_RPM)
}

export function getDefaultHiddenPlotChannels(yByChannel: Record<string, number[]>): string[] {
  return Object.entries(yByChannel)
    .filter(([channel, values]) => isAutoHiddenSpindleCommandChannel(channel) && hasImplausiblyHighSpindleCommand(values))
    .map(([channel]) => channel)
}

export function getDefaultVisiblePlotChannels(yByChannel: Record<string, number[]>): string[] {
  const allChannels = Object.keys(yByChannel)
  const hidden = new Set(getDefaultHiddenPlotChannels(yByChannel))
  const visible = sortPlotChannelsByImportance(allChannels.filter((channel) => !hidden.has(channel)))
  if (visible.length === 0) return sortPlotChannelsByImportance(allChannels)
  if (visible.length <= DEFAULT_VISIBLE_PLOT_CHANNEL_LIMIT) return visible
  return visible.slice(0, DEFAULT_VISIBLE_PLOT_CHANNEL_LIMIT)
}

export function resolveVisiblePlotChannels(
  yByChannel: Record<string, number[]>,
  selectedChannels: string[],
): string[] {
  const allChannels = Object.keys(yByChannel)
  const hasExplicitNone = selectedChannels.includes('__none__')
  if (hasExplicitNone) return []

  const explicit = selectedChannels.filter((channel) => channel !== '__none__' && allChannels.includes(channel))
  if (explicit.length > 0) return explicit

  return getDefaultVisiblePlotChannels(yByChannel)
}
