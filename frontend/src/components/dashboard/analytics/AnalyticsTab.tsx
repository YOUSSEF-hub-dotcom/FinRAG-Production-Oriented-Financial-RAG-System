"use client"

import { useEffect, useState } from "react"
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  LabelList,
} from "recharts"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { getApiBaseUrl } from "@/lib/settings"
import { apiGetAnalytics, type AnalyticsSummary } from "@/lib/api"
import { cn } from "@/lib/utils"
import {
  Activity,
  Timer,
  Zap,
  Cpu,
  Server,
  Database,
  Layers,
  Bot,
  CheckCircle2,
  XCircle,
  Loader2,
  Gauge,
  HardDrive,
} from "lucide-react"

interface HealthServices {
  mongodb: string
  qdrant: string
  redis: string
}

interface ServiceStatus {
  name: string
  detail: string
  icon: typeof Server
  status: "ok" | "degraded" | "down" | "unknown"
  live?: boolean
}

export function AnalyticsTab() {
  const [health, setHealth] = useState<{ status: string; services?: HealthServices } | null>(null)
  const [checking, setChecking] = useState(true)
  const [live, setLive] = useState(false)
  const [analytics, setAnalytics] = useState<AnalyticsSummary | null>(null)
  const [analyticsLoading, setAnalyticsLoading] = useState(true)
  const [analyticsError, setAnalyticsError] = useState<string | null>(null)

  // Health check
  useEffect(() => {
    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), 4000)
    fetch(`${getApiBaseUrl()}/health`, { signal: controller.signal })
      .then(async (res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const payload = (await res.json()) as { status: string; services?: HealthServices }
        setHealth(payload)
        setLive(true)
      })
      .catch(() => {
        setHealth(null)
        setLive(false)
      })
      .finally(() => {
        setChecking(false)
        clearTimeout(timeout)
      })
    return () => {
      clearTimeout(timeout)
      controller.abort()
    }
  }, [])

  // Analytics data
  useEffect(() => {
    let cancelled = false
    setAnalyticsLoading(true)
    apiGetAnalytics(30)
      .then((data) => {
        if (!cancelled) {
          setAnalytics(data)
          setAnalyticsError(null)
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setAnalyticsError(err instanceof Error ? err.message : "Failed to load analytics")
        }
      })
      .finally(() => {
        if (!cancelled) setAnalyticsLoading(false)
      })
    return () => { cancelled = true }
  }, [])

  const perf = analytics?.performance
  const evalScores = analytics?.eval_scores
  const cq = analytics?.context_quality
  const volume = analytics?.volume

  const serviceStatus = (key: "qdrant" | "redis", fallback: boolean): ServiceStatus["status"] => {
    if (checking) return "unknown"
    if (!live || !health?.services) return fallback ? "ok" : "down"
    const value = health.services[key]
    if (value === "ok") return "ok"
    return value?.startsWith("unavailable") ? "down" : "degraded"
  }

  const services: ServiceStatus[] = [
    {
      name: "FastAPI Backend",
      detail: "RAG query · ingestion · auth",
      icon: Server,
      status: checking ? "unknown" : live ? "ok" : "down",
      live,
    },
    {
      name: "Qdrant Vector DB",
      detail: `financial_vectors · ${volume?.total_chunks ?? "..."} chunks`,
      icon: Database,
      status: serviceStatus("qdrant", true),
    },
    {
      name: "Redis Cache",
      detail: `Cache hit rate: ${perf?.cache_hit_rate != null ? `${perf.cache_hit_rate}%` : "..."}`,
      icon: Layers,
      status: serviceStatus("redis", true),
    },
    {
      name: "LLM Service (Groq)",
      detail: evalScores?.judge_model ? `Judge: ${evalScores.judge_model}` : "Groq models",
      icon: Bot,
      status: checking ? "unknown" : live ? "ok" : "down",
    },
  ]

  // Performance KPI cards — built from real data
  const perfCards = [
    {
      label: "Avg RAG Latency",
      value: perf?.avg_latency_ms != null ? String(Math.round(perf.avg_latency_ms)) : "—",
      unit: "ms",
      trend: perf?.p50_latency_ms != null ? `P50=${Math.round(perf.p50_latency_ms)}ms` : "",
      trendDirection: "flat" as const,
      icon: Timer,
      hint: `P95=${perf?.p95_latency_ms != null ? Math.round(perf.p95_latency_ms) : "—"}ms · P99=${perf?.p99_latency_ms != null ? Math.round(perf.p99_latency_ms) : "—"}ms`,
    },
    {
      label: "LLM First-Token Latency",
      value: perf?.avg_ttft_ms != null ? String(Math.round(perf.avg_ttft_ms)) : "—",
      unit: "ms",
      trend: `${perf?.total_queries ?? 0} queries (30d)`,
      trendDirection: "flat" as const,
      icon: Zap,
      hint: "Time to first token — Groq streaming",
    },
    {
      label: "Cache Hit Rate",
      value: perf?.cache_hit_rate != null ? String(perf.cache_hit_rate) : "—",
      unit: "%",
      trend: perf?.guardrail_pass_rate != null ? `Guardrail: ${perf.guardrail_pass_rate}%` : "",
      trendDirection: perf?.cache_hit_rate != null && perf.cache_hit_rate > 50 ? "up" : "flat",
      icon: Activity,
      hint: "Semantic similarity cache — two-tier matching",
    },
    {
      label: "Total Chunks Indexed",
      value: volume?.total_chunks != null ? String(volume.total_chunks) : "—",
      unit: "docs",
      trend: volume?.by_ticker
        ? Object.entries(volume.by_ticker).map(([t, c]) => `${t}:${c}`).join(" · ")
        : "",
      trendDirection: "flat" as const,
      icon: HardDrive,
      hint: "AAPL + MSFT + NVDA SEC 10-K filings",
    },
  ]

  // Evaluation scores bar chart data
  const retrievalMetrics = evalScores
    ? [
        { metric: "Faithfulness", score: evalScores.faithfulness, color: "var(--chart-3)" },
        { metric: "Answer Relevance", score: evalScores.answer_relevance, color: "var(--chart-1)" },
        { metric: "Context Precision", score: evalScores.context_precision, color: "var(--chart-2)" },
        { metric: "Context Recall", score: evalScores.context_recall, color: "var(--chart-4)" },
      ]
    : []

  // Context quality distribution bar chart data — explicit visible colors
  // (green/amber/blue) so bars render regardless of theme CSS variables.
  const contextQualityData = cq
    ? [
        { bucket: "High (>0.8)", share: cq.high, color: "#22c55e" },
        { bucket: "Mid (0.5-0.8)", share: cq.mid, color: "#f59e0b" },
        { bucket: "Low (<0.5)", share: cq.low, color: "#3b82f6" },
      ]
    : []
  const hasContextQuality = contextQualityData.some((entry) => (entry.share ?? 0) > 0)

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Gauge className="h-5 w-5 text-primary" />
              RAG System Analytics &amp; Diagnostics
            </CardTitle>
            <CardDescription>Live performance telemetry and system health</CardDescription>
          </div>
          <Badge variant={live ? "success" : "warning"} className="gap-1.5">
            {checking ? (
              <Loader2 className="h-3 w-3 animate-spin" />
            ) : live ? (
              <CheckCircle2 className="h-3 w-3" />
            ) : (
              <XCircle className="h-3 w-3" />
            )}
            {checking ? "Probing /health..." : live ? "Live telemetry" : "Backend offline"}
          </Badge>
        </CardHeader>
      </Card>

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {perfCards.map((card) => {
          const Icon = card.icon
          return (
            <Card key={card.label} className="relative overflow-hidden">
              <div
                className={cn(
                  "absolute inset-x-0 top-0 h-0.5",
                  card.trendDirection === "down"
                    ? "bg-success"
                    : card.trendDirection === "up"
                      ? "bg-primary"
                      : "bg-muted-foreground/40"
                )}
              />
              <CardHeader className="space-y-0 pb-2">
                <CardTitle className="text-xs font-medium text-muted-foreground">
                  {card.label}
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-baseline gap-1.5">
                  <span className="text-2xl font-bold tracking-tight">{card.value}</span>
                  <span className="text-xs text-muted-foreground">{card.unit}</span>
                </div>
                <div className="mt-1.5 flex items-center gap-1.5 text-xs">
                  <Icon className="h-3.5 w-3.5 text-muted-foreground" />
                  <span className="font-medium text-muted-foreground">
                    {card.trend}
                  </span>
                </div>
                <p className="mt-2 border-t pt-2 text-[11px] leading-snug text-muted-foreground">
                  {card.hint}
                </p>
              </CardContent>
            </Card>
          )
        })}
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Retrieval Quality (LLM-as-a-Judge)</CardTitle>
            <CardDescription>
              {evalScores
                ? `${evalScores.sample_count}-sample evaluation · judge: ${evalScores.judge_model}`
                : analyticsLoading
                  ? "Loading evaluation scores..."
                  : "No evaluation data available"}
            </CardDescription>
          </CardHeader>
          <CardContent>
            {analyticsLoading ? (
              <div className="flex h-[260px] items-center justify-center">
                <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
              </div>
            ) : retrievalMetrics.length > 0 ? (
              <div className="h-[260px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={retrievalMetrics} layout="vertical" margin={{ left: 8 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" horizontal={false} />
                    <XAxis
                      type="number"
                      domain={[0, 1]}
                      tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 11 }}
                      axisLine={{ stroke: "hsl(var(--border))" }}
                    />
                    <YAxis
                      type="category"
                      dataKey="metric"
                      width={130}
                      tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 11 }}
                      axisLine={{ stroke: "hsl(var(--border))" }}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "hsl(var(--card))",
                        border: "1px solid hsl(var(--border))",
                        borderRadius: "8px",
                        fontSize: "12px",
                      }}
                      formatter={(value) => [Number(value).toFixed(3), "Score"]}
                    />
                    <Bar dataKey="score" radius={[0, 4, 4, 0]} barSize={22}>
                      {retrievalMetrics.map((entry) => (
                        <Cell key={entry.metric} fill={entry.color} />
                      ))}
                      <LabelList
                        dataKey="score"
                        position="right"
                        formatter={(value) => Number(value).toFixed(3)}
                        style={{ fill: "hsl(var(--foreground))", fontSize: 11 }}
                      />
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <div className="flex h-[260px] items-center justify-center text-sm text-muted-foreground">
                {analyticsError ? `Error: ${analyticsError}` : "No data yet — run evaluation first"}
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Context Quality Distribution</CardTitle>
            <CardDescription>
              {cq && cq.total > 0
                ? `${cq.total} chunks scored across audit logs`
                : "Share of retrieved contexts by relevance bucket"}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {analyticsLoading ? (
              <div className="flex h-[160px] items-center justify-center">
                <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
              </div>
            ) : contextQualityData.length > 0 && hasContextQuality ? (
              <>
                <div className="h-[160px] w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={contextQualityData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
                      <XAxis
                        dataKey="bucket"
                        tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 11 }}
                        axisLine={{ stroke: "hsl(var(--border))" }}
                      />
                      <YAxis
                        tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 11 }}
                        axisLine={{ stroke: "hsl(var(--border))" }}
                        tickFormatter={(v) => `${v}%`}
                      />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "hsl(var(--card))",
                          border: "1px solid hsl(var(--border))",
                          borderRadius: "8px",
                          fontSize: "12px",
                        }}
                        formatter={(value) => [`${value}%`, "Share"]}
                      />
                      <Bar dataKey="share" radius={[4, 4, 0, 0]} barSize={48}>
                        {contextQualityData.map((entry) => (
                          <Cell key={entry.bucket} fill={entry.color} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <div className="space-y-2">
                  {contextQualityData.map((entry) => (
                    <div key={entry.bucket} className="flex items-center gap-3 text-xs">
                      <span
                        className="h-2.5 w-2.5 shrink-0 rounded-sm"
                        style={{ backgroundColor: entry.color }}
                      />
                      <span className="text-muted-foreground">{entry.bucket}</span>
                      <span className="ml-auto font-semibold">{entry.share}%</span>
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <div className="flex h-[160px] items-center justify-center text-sm text-muted-foreground">
                No context quality data available yet
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>System Health</CardTitle>
          <CardDescription>
            Service availability probe —{" "}
            {live
              ? `backend reports "${health?.status ?? "ok"}"`
              : "using cached state (backend unreachable)"}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {services.map((service) => {
              const Icon = service.icon
              const isOk = service.status === "ok"
              const isDown = service.status === "down"
              const isChecking = service.status === "unknown"
              return (
                <div
                  key={service.name}
                  className={cn(
                    "flex items-center gap-3 rounded-xl border p-4",
                    isOk && "border-success/40 bg-success/5",
                    isDown && "border-destructive/50 bg-destructive/5",
                    isChecking && "border-border"
                  )}
                >
                  <div
                    className={cn(
                      "flex h-10 w-10 shrink-0 items-center justify-center rounded-lg",
                      isOk && "bg-success/15 text-success",
                      isDown && "bg-destructive/15 text-destructive",
                      isChecking && "bg-muted text-muted-foreground"
                    )}
                  >
                    {isChecking ? (
                      <Loader2 className="h-5 w-5 animate-spin" />
                    ) : isOk ? (
                      <CheckCircle2 className="h-5 w-5" />
                    ) : (
                      <XCircle className="h-5 w-5" />
                    )}
                  </div>
                  <div className="min-w-0">
                    <div className="flex items-center gap-2">
                      <Icon className="h-3.5 w-3.5 text-muted-foreground" />
                      <span className="truncate text-sm font-semibold">{service.name}</span>
                    </div>
                    <p className="mt-0.5 truncate text-[11px] text-muted-foreground">
                      {service.detail}
                    </p>
                    <Badge
                      variant={
                        isOk ? "success" : isDown ? "destructive" : "outline"
                      }
                      className="mt-1.5 gap-1 px-1.5 py-0 text-[10px]"
                    >
                      <span
                        className={cn(
                          "h-1.5 w-1.5 rounded-full",
                          isOk && "bg-success animate-pulse",
                          isDown && "bg-destructive",
                          isChecking && "bg-muted-foreground"
                        )}
                      />
                      {isChecking ? "Checking..." : isOk ? "Healthy" : isDown ? "Unavailable" : "Degraded"}
                    </Badge>
                  </div>
                </div>
              )
            })}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
