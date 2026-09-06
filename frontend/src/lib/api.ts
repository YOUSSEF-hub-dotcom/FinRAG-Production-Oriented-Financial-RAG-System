import { getApiBaseUrl } from "@/lib/settings"

export interface ApiError {
  detail: string
  error_code?: string
}

export interface TokenResponse {
  access_token: string
  refresh_token: string
  token_type: string
  expires_in: number
}

export interface UserProfile {
  user_id: string
  email: string
  full_name: string
  role: "user" | "admin"
  is_active: boolean
  created_at?: string
}

export interface SignupPayload {
  email: string
  password: string
  full_name: string
}

export interface LoginPayload {
  email: string
  password: string
}

export interface PerformanceMetrics {
  total_queries: number
  avg_latency_ms: number
  avg_ttft_ms: number
  cache_hit_rate: number
  guardrail_pass_rate: number
  p50_latency_ms: number
  p95_latency_ms: number
  p99_latency_ms: number
}

export interface EvalScores {
  faithfulness: number
  answer_relevance: number
  context_precision: number
  context_recall: number
  sample_count: number
  judge_model: string
}

export interface ContextQuality {
  high: number
  mid: number
  low: number
  total: number
}

export interface VolumeStats {
  total_chunks: number
  by_ticker: Record<string, number>
}

export interface AnalyticsSummary {
  performance: PerformanceMetrics
  eval_scores: EvalScores
  context_quality: ContextQuality
  volume: VolumeStats
}

let accessToken: string | null = null
let refreshPromise: Promise<string | null> | null = null

const TOKEN_KEY = "finsight_access_token"

export function getAccessToken(): string | null {
  if (accessToken) return accessToken
  if (typeof window !== "undefined") {
    accessToken = localStorage.getItem(TOKEN_KEY)
  }
  return accessToken
}

export function setAccessToken(token: string | null): void {
  accessToken = token
  if (typeof window !== "undefined") {
    if (token) {
      localStorage.setItem(TOKEN_KEY, token)
    } else {
      localStorage.removeItem(TOKEN_KEY)
    }
  }
}

async function tryRefreshToken(): Promise<string | null> {
  try {
    const res = await fetch(`${getApiBaseUrl()}/api/v1/auth/refresh`, {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
    })
    if (!res.ok) return null
    const data: TokenResponse = await res.json()
    accessToken = data.access_token
    return data.access_token
  } catch {
    return null
  }
}

async function fetchWithAuth(
  url: string,
  options: RequestInit = {}
): Promise<Response> {
  const headers = new Headers(options.headers)
  if (accessToken) {
    headers.set("Authorization", `Bearer ${accessToken}`)
  }
  let res = await fetch(url, { ...options, headers, credentials: "include" })
  if (res.status === 401 && accessToken) {
    if (!refreshPromise) {
      refreshPromise = tryRefreshToken()
    }
    const newToken = await refreshPromise
    refreshPromise = null
    if (newToken) {
      headers.set("Authorization", `Bearer ${newToken}`)
      res = await fetch(url, { ...options, headers, credentials: "include" })
    } else {
      accessToken = null
      if (typeof window !== "undefined") {
        window.location.href = "/login"
      }
    }
  }
  return res
}

export async function apiGet<T>(path: string): Promise<T> {
  const res = await fetchWithAuth(`${getApiBaseUrl()}${path}`)
  if (!res.ok) {
    const err: ApiError = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || `Request failed: ${res.status}`)
  }
  return res.json()
}

export async function apiPost<T>(
  path: string,
  body?: unknown,
  options?: { noAuth?: boolean }
): Promise<T> {
  const headers: Record<string, string> = {}
  if (body) headers["Content-Type"] = "application/json"
  const fetchFn = options?.noAuth ? fetch : fetchWithAuth
  const url = `${getApiBaseUrl()}${path}`
  const res = await fetchFn(url, {
    method: "POST",
    headers,
    body: body ? JSON.stringify(body) : undefined,
    credentials: "include",
  })
  if (!res.ok) {
    const err: ApiError = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || `Request failed: ${res.status}`)
  }
  return res.json()
}

export interface DocumentUploadResponse {
  filename: string
  ticker: string
  fiscal_year: string
  chunks_created: number
  mongo_count: number
  qdrant_count: number
  elapsed_seconds: number
  task_id?: string
}

export async function apiUploadFile(
  path: string,
  file: File,
  params?: Record<string, string>
): Promise<DocumentUploadResponse> {
  const url = new URL(`${getApiBaseUrl()}${path}`)
  if (params) {
    for (const [key, value] of Object.entries(params)) {
      if (value) url.searchParams.set(key, value)
    }
  }
  const formData = new FormData()
  formData.append("file", file)
  const res = await fetchWithAuth(url.toString(), { method: "POST", body: formData })
  if (!res.ok) {
    const err: ApiError = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || `Upload failed: ${res.status}`)
  }
  return res.json()
}

export async function apiLogout(): Promise<void> {
  try {
    await fetchWithAuth(`${getApiBaseUrl()}/api/v1/auth/logout`, { method: "POST" })
  } finally {
    setAccessToken(null)
  }
}

export async function apiGetAnalytics(days: number = 30): Promise<AnalyticsSummary> {
  return apiGet<AnalyticsSummary>(`/api/v1/analytics/summary?days=${days}`)
}
