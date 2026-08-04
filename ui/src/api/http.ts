export type HttpMethod = 'GET' | 'POST' | 'PATCH' | 'DELETE'

const API_BASE_URL_KEY = 'apiBaseUrl'
const LEGACY_DEFAULT_API_BASE_URL = 'http://localhost:8000'

function inferredBaseUrl(): string {
  if (typeof window === 'undefined' || !window.location) {
    return LEGACY_DEFAULT_API_BASE_URL
  }

  const protocol = window.location.protocol === 'https:' ? 'https:' : 'http:'
  const hostname = window.location.hostname || 'localhost'
  return `${protocol}//${hostname}:8000`
}

export function baseUrl(): string {
  if (typeof localStorage === 'undefined') {
    return inferredBaseUrl()
  }

  const stored = localStorage.getItem(API_BASE_URL_KEY)?.trim()
  const fallback = inferredBaseUrl()
  const resolved = stored && stored.length > 0
    ? (stored === LEGACY_DEFAULT_API_BASE_URL ? fallback : stored)
    : fallback

  return resolved.replace(/\/+$/, '')
}

export function setBaseUrl(url: string) {
  localStorage.setItem(API_BASE_URL_KEY, url.replace(/\/+$/, ''))
}

export async function api<T>(path: string, method: HttpMethod = 'GET', body?: unknown): Promise<T> {
  const url = `${baseUrl()}${path}`
  const resp = await fetch(url, {
    method,
    headers: {
      'Content-Type': 'application/json',
    },
    body: body === undefined ? undefined : JSON.stringify(body),
  })

  if (!resp.ok) {
    const text = await resp.text().catch(() => '')
    throw new Error(`${method} ${path} failed: ${resp.status} ${text}`)
  }

  return (await resp.json()) as T
}

export function wsUrl(path: string): string {
  const b = baseUrl()
  const u = new URL(b)
  const proto = u.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${proto}//${u.host}${path}`
}
