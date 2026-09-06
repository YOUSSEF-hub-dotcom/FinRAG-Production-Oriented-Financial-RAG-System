export interface SourceChunk {
  chunk_id: string
  text: string
  score: number
  ticker?: string
  fiscal_year?: string
  section?: string
  source_file?: string
  page_number?: number
  // Fallback keys for backward compatibility
  page_content?: string
  chunk_text?: string
  content?: string
  snippet?: string
  ticker_tag?: string
  year?: string
  doc_type?: string
  filename?: string
  file?: string
  metadata?: { text?: string; [key: string]: unknown }
}

/**
 * Sentinel values that must be treated as "missing" when rendering
 * metadata headers, so titles never show fragments like "- N/A".
 */
const NULLISH_META_VALUES = new Set([
  "",
  "n/a",
  "na",
  "none",
  "null",
  "undefined",
  "unknown",
])

/**
 * Return the first non-nullish metadata value from `values`.
 * Null/undefined/""/"N/A"/"UNKNOWN" etc. are skipped entirely so callers
 * can render clean headers instead of appending empty fragments.
 */
export function cleanMetaValue(
  ...values: Array<string | number | null | undefined>
): string | undefined {
  for (const value of values) {
    if (value === null || value === undefined) continue
    const normalized = String(value).trim()
    if (!normalized || NULLISH_META_VALUES.has(normalized.toLowerCase())) {
      continue
    }
    return normalized
  }
  return undefined
}

/**
 * Extract context text across every backend payload naming variant:
 * page_content / text / chunk_text / content / snippet.
 */
export function resolveChunkText(src: Partial<SourceChunk> | null | undefined): string {
  if (!src) return ""
  const content =
    src.page_content ||
    src.text ||
    src.chunk_text ||
    src.content ||
    src.snippet ||
    (typeof src.metadata?.text === "string" ? src.metadata.text : "") ||
    ""
  return typeof content === "string" ? content : String(content)
}

export interface StreamCallbacks {
  onToken: (text: string) => void
  onAnswer: (text: string) => void
  onSources: (sources: SourceChunk[]) => void
  onDone: () => void
  onError: (error: Error) => void
}

import { getApiBaseUrl } from "@/lib/settings"

export async function streamRagQuery(
  query: string,
  ticker: string,
  callbacks: StreamCallbacks,
  signal?: AbortSignal,
  sessionId?: string | null
): Promise<void> {
  try {
    const payload: Record<string, unknown> = { user_query: query, ticker }
    if (sessionId) payload.session_id = sessionId
    const res = await fetch(`${getApiBaseUrl()}/api/v1/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal,
    })

    if (!res.ok) {
      throw new Error(`HTTP ${res.status}: ${res.statusText}`)
    }

    const reader = res.body?.getReader()
    if (!reader) throw new Error("No response body")

    const decoder = new TextDecoder()
    let buffer = ""
    let eventType = "message"

    const handleData = (payload: string) => {
      if (payload === "[DONE]") {
        callbacks.onDone()
        return
      }
      if (eventType === "token") {
        callbacks.onToken(payload)
        return
      }
      if (eventType === "answer") {
        callbacks.onAnswer(payload)
        return
      }
      if (eventType === "sources") {
        try {
          const sources = JSON.parse(payload)
          if (Array.isArray(sources)) {
            callbacks.onSources(sources)
          }
        } catch {
          // ignore malformed sources
        }
        return
      }
      if (eventType === "done") {
        callbacks.onDone()
        return
      }
    }

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split("\n")
      buffer = lines.pop() || ""

      for (const line of lines) {
        const trimmed = line.trim()
        if (!trimmed) continue
        if (trimmed.startsWith("event:")) {
          eventType = trimmed.slice(6).trim()
        } else if (trimmed.startsWith("data:")) {
          handleData(trimmed.slice(5).trim())
        }
      }
    }

    callbacks.onDone()
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      callbacks.onDone()
      return
    }
    callbacks.onError(err instanceof Error ? err : new Error(String(err)))
  }
}
