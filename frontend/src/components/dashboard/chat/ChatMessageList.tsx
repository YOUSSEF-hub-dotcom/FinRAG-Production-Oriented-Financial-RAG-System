"use client"

import { useEffect, useRef } from "react"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import { User, Bot, ExternalLink } from "lucide-react"
import { cn } from "@/lib/utils"
import type { SourceChunk } from "@/lib/ragStream"
import { cleanMetaValue } from "@/lib/ragStream"

export interface ChatMessage {
  id: string
  role: "user" | "assistant"
  content: string
  sources?: SourceChunk[]
  timestamp: Date
}

interface ChatMessageListProps {
  messages: ChatMessage[]
  onCitationClick: (source: SourceChunk) => void
}

/**
 * Defensive helper: if the message content is a raw JSON object leaking
 * from the LLM (e.g. ``{"internal_thought":...,"answer":"..."}``), extract
 * only the ``answer`` field.  Returns the original string otherwise.
 */
function stripJsonPayload(raw: string): string {
  const trimmed = raw.trim()
  if (!trimmed.startsWith("{")) return raw
  try {
    const parsed = JSON.parse(trimmed)
    if (parsed && typeof parsed === "object" && typeof parsed.answer === "string") {
      return parsed.answer
    }
  } catch {
    // Not valid JSON — return as-is
  }
  return raw
}

export function ChatMessageList({ messages, onCitationClick }: ChatMessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  if (messages.length === 0) {
    return null
  }

  return (
    <div className="flex-1 overflow-y-auto px-1 py-4">
      <div className="space-y-6">
        {messages.map((msg) => {
          const displayContent = msg.role === "assistant" ? stripJsonPayload(msg.content) : msg.content
          return (
          <div
            key={msg.id}
            className={cn(
              "flex gap-3",
              msg.role === "user" ? "justify-end" : "justify-start"
            )}
          >
            {msg.role === "assistant" && (
              <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground">
                <Bot className="h-3.5 w-3.5" />
              </div>
            )}
            <div
              className={cn(
                "max-w-[80%] rounded-xl px-4 py-3 text-sm",
                msg.role === "user"
                  ? "bg-primary text-primary-foreground"
                  : "bg-muted"
              )}
            >
              {msg.role === "assistant" ? (
                <div className="prose prose-sm dark:prose-invert max-w-none">
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={{
                      table: ({ children }) => (
                        <div className="my-2 overflow-x-auto">
                          <table className="min-w-full border-collapse text-xs">
                            {children}
                          </table>
                        </div>
                      ),
                      th: ({ children }) => (
                        <th className="border-b border-border px-2 py-1 text-left font-medium">
                          {children}
                        </th>
                      ),
                      td: ({ children }) => (
                        <td className="border-b border-border px-2 py-1">
                          {children}
                        </td>
                      ),
                      code: ({ className, children, ...props }) => {
                        const isInline = !className
                        if (isInline) {
                          return (
                            <code
                              className="rounded bg-muted-foreground/20 px-1 py-0.5 text-xs font-mono"
                              {...props}
                            >
                              {children}
                            </code>
                          )
                        }
                        return (
                          <code className={className} {...props}>
                            {children}
                          </code>
                        )
                      },
                    }}
                  >
                    {displayContent}
                  </ReactMarkdown>
                  {msg.sources && msg.sources.length > 0 && (
                    <div className="mt-3 flex flex-wrap gap-1.5 border-t border-border pt-2">
                      {msg.sources.map((src, i) => {
                        // Clean metadata: null/undefined/"N/A" values are
                        // dropped so chips never render "- N/A" fragments.
                        const sectionLabel = cleanMetaValue(src.section)
                        const yearLabel = cleanMetaValue(src.fiscal_year, src.year)
                        return (
                        <button
                          key={`${src.chunk_id || 'src'}-${i}`}
                          onClick={() => onCitationClick(src)}
                          className="inline-flex items-center gap-1 rounded-md bg-primary/10 px-2 py-0.5 text-[10px] font-medium text-primary hover:bg-primary/20 transition-colors"
                        >
                          <ExternalLink className="h-2.5 w-2.5" />
                          {sectionLabel || cleanMetaValue(src.source_file) || `Source ${i + 1}`}
                          {yearLabel && (
                            <span className="text-muted-foreground">
                              ({yearLabel})
                            </span>
                          )}
                        </button>
                        )
                      })}
                    </div>
                  )}
                </div>
              ) : (
                <p>{msg.content}</p>
              )}
              <div
                className={cn(
                  "mt-1 text-[10px]",
                  msg.role === "user"
                    ? "text-primary-foreground/60"
                    : "text-muted-foreground"
                )}
              >
                {msg.timestamp.toLocaleTimeString([], {
                  hour: "2-digit",
                  minute: "2-digit",
                })}
              </div>
            </div>
            {msg.role === "user" && (
              <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-muted text-muted-foreground">
                <User className="h-3.5 w-3.5" />
              </div>
            )}
          </div>
          )
        })}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}
