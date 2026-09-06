"use client"

import { useState, useRef, useCallback } from "react"
import { useApp } from "@/context/AppContext"
import { streamRagQuery, type SourceChunk } from "@/lib/ragStream"
import { ChatInput } from "./ChatInput"
import { ChatMessageList, type ChatMessage } from "./ChatMessageList"
import { CitationInspectorDrawer } from "./CitationInspectorDrawer"
import { SuggestedPrompts } from "./SuggestedPrompts"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Trash2, Wifi, Loader2 } from "lucide-react"

type StreamStatus = "idle" | "thinking" | "streaming"

/**
 * One conversation = one stable session id. The backend keys per-session
 * conversation memory by this id, so it must be reused across turns and only
 * regenerated when the user starts a new chat (clears the conversation).
 * Mirrors the reference session pattern in app/ui/streamlit_app.py.
 */
function createSessionId(): string {
  const id =
    typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
      ? crypto.randomUUID()
      : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`
  return `sess_${id}`
}

export function RagChatTab() {
  const { activeTicker } = useApp()
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [sessionId, setSessionId] = useState<string>(() => createSessionId())
  const [status, setStatus] = useState<StreamStatus>("idle")
  const [inspectorSource, setInspectorSource] = useState<SourceChunk | null>(null)
  const [inspectorOpen, setInspectorOpen] = useState(false)
  const sourcesRef = useRef<SourceChunk[]>([])
  const abortRef = useRef<AbortController | null>(null)
  const streamBufferRef = useRef("")

  const handleSend = useCallback(
    (query: string) => {
      const userMsg: ChatMessage = {
        id: `user-${Date.now()}`,
        role: "user",
        content: query,
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, userMsg])
      setStatus("thinking")
      sourcesRef.current = []
      streamBufferRef.current = ""

      const controller = new AbortController()
      abortRef.current = controller

      streamRagQuery(query, activeTicker.symbol, {
        onToken: (text) => {
          setStatus("streaming")
          // Word-safe join: SSE token events are assembled from model output
          // chunked into small word groups. The frontend parser trims each
          // frame (dropping any leading space the backend attached to mark a
          // chunk boundary), so concatenating raw frames can fuse the last
          // word of one frame with the first of the next ("year2025"). Insert
          // a space at the boundary whenever both sides are non-whitespace.
          const buf = streamBufferRef.current
          const needSpace =
            buf.length > 0 &&
            text.length > 0 &&
            !/\s$/.test(buf) &&
            !/^\s/.test(text)
          streamBufferRef.current = needSpace ? buf + " " + text : buf + text
          setMessages((prev) => {
            const updated = [...prev]
            const last = updated[updated.length - 1]
            if (last?.role === "assistant" && last.id.startsWith("stream-")) {
              updated[updated.length - 1] = {
                ...last,
                content: streamBufferRef.current,
              }
            } else {
              updated.push({
                id: `stream-${Date.now()}`,
                role: "assistant",
                content: streamBufferRef.current,
                timestamp: new Date(),
              })
            }
            return updated
          })
        },
        onAnswer: (text) => {
          streamBufferRef.current = text
          setMessages((prev) => {
            const updated = [...prev]
            const last = updated[updated.length - 1]
            if (last?.role === "assistant" && last.id.startsWith("stream-")) {
              updated[updated.length - 1] = {
                ...last,
                content: text,
              }
            } else {
              updated.push({
                id: `stream-${Date.now()}`,
                role: "assistant",
                content: text,
                timestamp: new Date(),
              })
            }
            return updated
          })
        },
        onSources: (srcs) => {
          sourcesRef.current = srcs
        },
        onDone: () => {
          const currentSources = sourcesRef.current
          setMessages((prev) => {
            const updated = [...prev]
            const last = updated[updated.length - 1]
            if (last?.role === "assistant" && last.id.startsWith("stream-")) {
              updated[updated.length - 1] = {
                ...last,
                id: `assistant-${Date.now()}`,
                sources: currentSources,
              }
            }
            return updated
          })
          setStatus("idle")
        },
        onError: (err) => {
          const errMsg: ChatMessage = {
            id: `error-${Date.now()}`,
            role: "assistant",
            content: `**Error:** ${err.message}. Please check that the FastAPI backend is running.`,
            timestamp: new Date(),
          }
          setMessages((prev) => [...prev, errMsg])
          setStatus("idle")
        },
      }, controller.signal, sessionId)
    },
    [activeTicker.symbol, sessionId]
  )

  const handleStop = () => {
    abortRef.current?.abort()
    setStatus("idle")
  }

  const handleClear = () => {
    abortRef.current?.abort()
    setMessages([])
    setSessionId(createSessionId())
    sourcesRef.current = []
    setStatus("idle")
  }

  const handleCitationClick = (source: SourceChunk) => {
    setInspectorSource(source)
    setInspectorOpen(true)
  }

  const handleSuggestedSelect = (prompt: string) => {
    handleSend(prompt)
  }

  const statusLabel =
    status === "idle"
      ? null
      : status === "thinking"
        ? "Thinking..."
        : "Streaming..."

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b px-4 py-3">
        <div className="flex items-center gap-3">
          <h2 className="text-sm font-semibold">AI Financial Copilot</h2>
          <Badge variant="outline" className="font-mono text-[10px]">
            {activeTicker.symbol}
          </Badge>
          {statusLabel && (
            <Badge variant="secondary" className="gap-1 text-[10px]">
              {status === "thinking" && (
                <Loader2 className="h-2.5 w-2.5 animate-spin" />
              )}
              {statusLabel}
            </Badge>
          )}
        </div>
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
            {status === "idle" ? (
              <Wifi className="h-3 w-3 text-success" />
            ) : (
              <Loader2 className="h-3 w-3 animate-spin text-primary" />
            )}
            {status === "idle" ? "Ready" : "Connected"}
          </div>
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            onClick={handleClear}
            disabled={messages.length === 0}
          >
            <Trash2 className="h-3.5 w-3.5" />
          </Button>
        </div>
      </div>

      <div className="flex-1 overflow-hidden">
        {messages.length === 0 ? (
          <div className="flex h-full flex-col items-center justify-center px-4">
            <div className="mb-6 max-w-lg text-center">
              <h3 className="text-lg font-semibold">RAG Financial Analyst</h3>
              <p className="mt-1 text-sm text-muted-foreground">
                Ask questions about SEC filings. Answers are grounded in real 10-K documents
                with source citations.
              </p>
            </div>
            <SuggestedPrompts
              tickerSymbol={activeTicker.symbol}
              onSelect={handleSuggestedSelect}
            />
          </div>
        ) : (
          <ChatMessageList
            messages={messages}
            onCitationClick={handleCitationClick}
          />
        )}
      </div>

      <div className="border-t p-4">
        <ChatInput
          onSend={handleSend}
          onStop={handleStop}
          isLoading={status !== "idle"}
          tickerSymbol={activeTicker.symbol}
        />
      </div>

      <CitationInspectorDrawer
        source={inspectorSource}
        open={inspectorOpen}
        onOpenChange={setInspectorOpen}
      />
    </div>
  )
}
