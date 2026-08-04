/**
 * Config — fetch /config/urls (Agent K Phase 1 backend) once at boot.
 *
 * Kept separate from `api/http.ts` so existing users of `baseUrl()`
 * remain untouched. Consumers that care about the SINDIT / GraphDB
 * / Neo4j URLs (e.g. the `LearningsPage`, future graph components)
 * call `getRuntimeUrls()` which caches the first successful fetch.
 */

import { api } from './http'

export interface RuntimeUrls {
  sindit?: string
  graphdb?: string
  influxdb?: string
  neo4j?: string
  upstream_knowledge?: string
  ui?: string
}

let _cache: RuntimeUrls | null = null
let _inflight: Promise<RuntimeUrls> | null = null

export async function getRuntimeUrls(force = false): Promise<RuntimeUrls> {
  if (!force && _cache) return _cache
  if (_inflight) return _inflight
  _inflight = (async () => {
    try {
      const urls = await api<RuntimeUrls>('/config/urls')
      _cache = urls || {}
    } catch {
      _cache = {}
    } finally {
      _inflight = null
    }
    return _cache!
  })()
  return _inflight
}

/** For tests. Never called by the app itself. */
export function _resetRuntimeUrlsCache() {
  _cache = null
  _inflight = null
}
