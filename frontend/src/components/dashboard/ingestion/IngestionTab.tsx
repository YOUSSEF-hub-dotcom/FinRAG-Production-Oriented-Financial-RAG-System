"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import { apiGet, apiUploadFile } from "@/lib/api"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Loader2, UploadCloud, FileUp, CheckCircle2, XCircle, Database, ScanText, Layers, Sparkles, ListChecks } from "lucide-react"
import { cn } from "@/lib/utils"

type StageId = "extraction" | "chunking" | "embedding" | "upsert"
type StageState = "pending" | "active" | "done" | "error"

interface PipelineStage {
  id: StageId
  label: string
  description: string
  icon: typeof ScanText
  state: StageState
  detail?: string
}

const INITIAL_STAGES: PipelineStage[] = [
  { id: "extraction", label: "Extraction", description: "Parse 10-K HTML/PDF → clean text", icon: ScanText, state: "pending" },
  { id: "chunking", label: "Chunking", description: "3-tier token-bounded hybrid split", icon: Layers, state: "pending" },
  { id: "embedding", label: "Embedding", description: "nomic-embed-text-v1.5 (768-dim)", icon: Sparkles, state: "pending" },
  { id: "upsert", label: "Upsert", description: "Write vectors to Qdrant collection", icon: Database, state: "pending" },
]

interface IndexedDocument {
  id: string
  ticker: string
  filingDate: string
  fiscalYear: string
  chunks: number
  collectionStatus: "indexed" | "pending" | "failed"
  size: string
}

const MOCK_DOCUMENTS: IndexedDocument[] = [
  { id: "doc-1", ticker: "AAPL", filingDate: "2025-01-24", fiscalYear: "FY2024", chunks: 218, collectionStatus: "indexed", size: "3.1 MB" },
  { id: "doc-2", ticker: "MSFT", filingDate: "2025-01-29", fiscalYear: "FY2024", chunks: 241, collectionStatus: "indexed", size: "2.8 MB" },
  { id: "doc-3", ticker: "NVDA", filingDate: "2025-02-26", fiscalYear: "FY2025", chunks: 194, collectionStatus: "indexed", size: "2.4 MB" },
  { id: "doc-4", ticker: "AAPL", filingDate: "2024-01-25", fiscalYear: "FY2023", chunks: 205, collectionStatus: "indexed", size: "3.0 MB" },
  { id: "doc-5", ticker: "MSFT", filingDate: "2024-01-30", fiscalYear: "FY2023", chunks: 228, collectionStatus: "indexed", size: "2.7 MB" },
]

interface IngestionTaskStatus {
  task_id: string
  status: "queued" | "processing" | "completed" | "failed"
  chunks_created?: number
  error?: string
}

const POLL_INTERVAL_MS = 1500
const MAX_POLL_FAILURES = 20

export function IngestionTab() {
  const [stages, setStages] = useState<PipelineStage[]>(INITIAL_STAGES)
  const [queued, setQueued] = useState(false)
  const [lastRun, setLastRun] = useState<{ status: "idle" | "success" | "error"; message: string; taskId?: string }>({
    status: "idle",
    message: "No ingestion job has been run yet.",
  })
  const [documents] = useState<IndexedDocument[]>(MOCK_DOCUMENTS)
  const [dragging, setDragging] = useState(false)
  const [files, setFiles] = useState<File[]>([])
  const [uploading, setUploading] = useState(false)
  const [ticker, setTicker] = useState("AAPL")
  const [fiscalYear, setFiscalYear] = useState("FY2025")
  const fileInputRef = useRef<HTMLInputElement>(null)

  const resetStages = () =>
    setStages((prev) =>
      prev.map((s) => ({ ...s, state: "pending" as StageState, detail: undefined }))
    )

  const handleFiles = useCallback(
    (incoming: File[]) => {
      setFiles((prev) => [...prev, ...incoming].slice(0, 3))
    },
    []
  )

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragging(false)
    handleFiles(Array.from(e.dataTransfer.files))
  }

  const handleUpload = async () => {
    if (files.length === 0 || uploading) return
    setUploading(true)
    setQueued(false)
    setLastRun({ status: "idle", message: `Uploading ${files[0].name} to the ingestion pipeline...` })
    resetStages()
    try {
      const payload = await apiUploadFile(
        "/api/v1/documents/upload",
        files[0],
        { ticker, fiscal_year: fiscalYear }
      )
      setQueued(true)
      setLastRun({
        status: "success",
        message: `Queued ingestion of ${payload.filename} (task ${payload.task_id ?? "?"}) — ticker=${payload.ticker}, fiscal_year=${payload.fiscal_year}. Processing in background: extraction → chunking → embedding → Qdrant upsert.`,
        taskId: payload.task_id,
      })
      setFiles([])
    } catch (err) {
      setLastRun({
        status: "error",
        message: `Upload failed: ${err instanceof Error ? err.message : String(err)}`,
      })
    } finally {
      setUploading(false)
    }
  }

  const pollFailuresRef = useRef(0)

  // Poll the backend for ingestion task status every 1.5s while a job is
  // queued; stops as soon as a terminal state (completed/failed) is reached.
  useEffect(() => {
    const taskId = lastRun.taskId
    if (!taskId || !queued) return
    pollFailuresRef.current = 0
    const interval = setInterval(async () => {
      try {
        const task = await apiGet<IngestionTaskStatus>(`/api/v1/documents/tasks/${taskId}`)
        if (task.status === "completed") {
          setQueued(false)
          setStages((prev) => prev.map((s) => ({ ...s, state: "done" as StageState })))
          setLastRun((prev) => ({
            ...prev,
            message: `Ingestion completed (task ${taskId})${typeof task.chunks_created === "number" ? ` — ${task.chunks_created} chunks indexed into Mongo + Qdrant.` : "."}`,
          }))
        } else if (task.status === "failed") {
          setQueued(false)
          setStages((prev) => prev.map((s) => ({ ...s, state: "error" as StageState })))
          setLastRun({
            status: "error",
            message: `Ingestion failed (task ${taskId}): ${task.error ?? "unknown error"}.`,
          })
        }
      } catch {
        pollFailuresRef.current += 1
        if (pollFailuresRef.current >= MAX_POLL_FAILURES) {
          setQueued(false)
          setLastRun({
            status: "error",
            message: `Lost track of ingestion task ${taskId} — status endpoint unreachable.`,
          })
        }
      }
    }, POLL_INTERVAL_MS)
    return () => clearInterval(interval)
  }, [lastRun.taskId, queued])

  const stageStats: Record<StageId, { count: string; unit: string }> = {
    extraction: { count: "653", unit: "chunks" },
    chunking: { count: "768-1024", unit: "tok" },
    embedding: { count: "768", unit: "dims" },
    upsert: { count: "714", unit: "points" },
  }

  const completedStages = stages.filter((s) => s.state === "done").length

  return (
    <div className="space-y-6">
      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FileUp className="h-5 w-5 text-primary" />
              Upload New 10-K Filing
            </CardTitle>
            <CardDescription>
              Drag &amp; drop a SEC 10-K PDF/HTML filing or browse to upload
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div
              onDragOver={(e) => {
                e.preventDefault()
                setDragging(true)
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className={cn(
                "flex cursor-pointer flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed p-8 text-center transition-colors",
                dragging
                  ? "border-primary bg-primary/5"
                  : "border-border hover:border-primary/50 hover:bg-muted/40"
              )}
            >
              <div className="flex h-12 w-12 items-center justify-center rounded-full bg-primary/10">
                <UploadCloud className="h-6 w-6 text-primary" />
              </div>
              <div>
                <p className="text-sm font-medium">
                  {dragging ? "Drop the filing here" : "Click to browse or drag & drop"}
                </p>
                <p className="mt-1 text-xs text-muted-foreground">
                  Supported: .html, .pdf, .txt · max 50 MB
                </p>
              </div>
              <input
                ref={fileInputRef}
                type="file"
                accept=".html,.pdf,.txt"
                className="hidden"
                onChange={(e) => {
                  if (e.target.files) handleFiles(Array.from(e.target.files))
                  e.target.value = ""
                }}
              />
            </div>

            {files.length > 0 && (
              <div className="space-y-2">
                {files.map((file) => (
                  <div
                    key={`${file.name}-${file.size}`}
                    className="flex items-center justify-between rounded-lg border px-3 py-2 text-xs"
                  >
                    <div className="flex min-w-0 items-center gap-2">
                      <FileUp className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
                      <span className="truncate font-medium">{file.name}</span>
                    </div>
                    <span className="ml-2 shrink-0 text-muted-foreground">
                      {(file.size / 1024 / 1024).toFixed(2)} MB
                    </span>
                  </div>
                ))}
              </div>
            )}

            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1.5">
                <Label className="text-xs">Ticker</Label>
                <Input value={ticker} onChange={(e) => setTicker(e.target.value.toUpperCase())} className="font-mono text-xs" />
              </div>
              <div className="space-y-1.5">
                <Label className="text-xs">Fiscal Year</Label>
                <Input value={fiscalYear} onChange={(e) => setFiscalYear(e.target.value)} className="font-mono text-xs" />
              </div>
            </div>

            <Button
              onClick={handleUpload}
              disabled={files.length === 0 || uploading}
              className="w-full"
            >
              {uploading ? <Loader2 className="h-4 w-4 animate-spin" /> : <UploadCloud className="h-4 w-4" />}
              {uploading ? "Uploading & Queueing..." : "Upload to Ingestion Pipeline"}
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <ListChecks className="h-5 w-5 text-primary" />
              Ingestion Job Status
            </CardTitle>
            <CardDescription>
              Uploaded filings are queued for background processing via FastAPI
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="rounded-lg border bg-muted/30 p-3 text-xs text-muted-foreground">
              <p className="font-medium text-foreground">Pipeline</p>
              <div className="mt-2 grid grid-cols-2 gap-2">
                <div>
                  Upload: <code className="font-mono text-primary">POST /api/v1/documents/upload</code>
                </div>
                <div>
                  Queue: <code className="font-mono text-primary">BackgroundTasks · Arq</code>
                </div>
                <div>
                  Embeddings: <code className="font-mono text-primary">nomic-v1.5</code>
                </div>
                <div>
                  Store: <code className="font-mono text-primary">Mongo + Qdrant</code>
                </div>
              </div>
            </div>

            <Alert
              variant={
                lastRun.status === "error"
                  ? "destructive"
                  : lastRun.status === "success"
                    ? "default"
                    : "default"
              }
            >
              <CheckCircle2
                className={cn(
                  "h-4 w-4",
                  lastRun.status === "success" && "text-success",
                  lastRun.status === "error" && "text-destructive"
                )}
              />
              <AlertTitle className="text-xs">
                {lastRun.status === "success" ? "Job Queued" : lastRun.status === "error" ? "Job Failed" : "Ready"}
              </AlertTitle>
              <AlertDescription className="text-xs">{lastRun.message}</AlertDescription>
            </Alert>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0">
          <div>
            <CardTitle>Ingestion Pipeline Status</CardTitle>
            <CardDescription>
              Extraction → Chunking → Embedding → Qdrant Upsert
            </CardDescription>
          </div>
          <div className="flex items-center gap-2">
            {queued ? (
              <Badge variant="secondary" className="gap-1.5">
                <Loader2 className="h-3 w-3 animate-spin" />
                Queued — processing in background
              </Badge>
            ) : (
              <Badge variant={completedStages === 4 ? "success" : "outline"} className="gap-1.5">
                <CheckCircle2 className="h-3 w-3" />
                {completedStages === 4 ? "Pipeline complete" : "Idle"}
              </Badge>
            )}
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {stages.map((stage) => {
              const Icon = stage.icon
              const isDone = stage.state === "done"
              const stats = stageStats[stage.id]
              return (
                <div
                  key={stage.id}
                  className={cn(
                    "relative overflow-hidden rounded-xl border p-4 transition-colors",
                    isDone && "border-success/40 bg-success/5",
                    queued && "border-primary/40 bg-primary/5",
                    stage.state === "error" && "border-destructive/50 bg-destructive/5"
                  )}
                >
                  {queued && (
                    <div className="absolute inset-x-0 top-0 h-0.5 animate-pulse bg-primary" />
                  )}
                  <div className="flex items-center gap-3">
                    <div
                      className={cn(
                        "flex h-9 w-9 shrink-0 items-center justify-center rounded-lg",
                        isDone && "bg-success/15 text-success",
                        queued && "bg-primary/15 text-primary",
                        stage.state === "error" && "bg-destructive/15 text-destructive",
                        stage.state === "pending" && "bg-muted text-muted-foreground"
                      )}
                    >
                      {isDone ? <CheckCircle2 className="h-4.5 w-4.5" /> : <Icon className="h-4.5 w-4.5" />}
                    </div>
                    <div className="min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-semibold">{stage.label}</span>
                        {queued && <Loader2 className="h-3 w-3 animate-spin text-primary" />}
                      </div>
                      <p className="truncate text-[11px] text-muted-foreground">{stage.description}</p>
                    </div>
                  </div>
                  <div className="mt-3 flex items-center justify-between border-t pt-2 text-[11px] text-muted-foreground">
                    <span>{stats.count} {stats.unit}</span>
                    <span
                      className={cn(
                        "font-medium",
                        isDone && "text-success",
                        queued && "text-primary"
                      )}
                    >
                      {isDone ? "Complete" : queued ? "Queued" : stage.state === "error" ? "Failed" : "Pending"}
                    </span>
                  </div>
                </div>
              )
            })}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Indexed Documents</CardTitle>
          <CardDescription>SEC 10-K filings indexed into the vector collection</CardDescription>
        </CardHeader>
        <CardContent className="p-0">
          <div className="overflow-x-auto">
            <table className="w-full min-w-[640px] border-collapse text-sm">
              <thead>
                <tr className="border-b bg-muted/30 text-left text-xs uppercase tracking-wider text-muted-foreground">
                  <th className="px-4 py-3 font-medium">Ticker</th>
                  <th className="px-4 py-3 font-medium">SEC Filing Date</th>
                  <th className="px-4 py-3 font-medium">Fiscal Year</th>
                  <th className="px-4 py-3 font-medium">Chunk Count</th>
                  <th className="px-4 py-3 font-medium">Size</th>
                  <th className="px-4 py-3 text-right font-medium">Vector Collection</th>
                </tr>
              </thead>
              <tbody>
                {documents.map((doc) => (
                  <tr key={doc.id} className="border-b transition-colors hover:bg-muted/40">
                    <td className="px-4 py-3 font-semibold">{doc.ticker}</td>
                    <td className="px-4 py-3 font-mono text-xs">{doc.filingDate}</td>
                    <td className="px-4 py-3 text-muted-foreground">{doc.fiscalYear}</td>
                    <td className="px-4 py-3 font-mono">{doc.chunks}</td>
                    <td className="px-4 py-3 text-muted-foreground">{doc.size}</td>
                    <td className="px-4 py-3 text-right">
                      {doc.collectionStatus === "indexed" ? (
                        <Badge variant="success" className="gap-1">
                          <CheckCircle2 className="h-3 w-3" />
                          Indexed
                        </Badge>
                      ) : doc.collectionStatus === "failed" ? (
                        <Badge variant="destructive" className="gap-1">
                          <XCircle className="h-3 w-3" />
                          Failed
                        </Badge>
                      ) : (
                        <Badge variant="warning" className="gap-1">
                          <Loader2 className="h-3 w-3 animate-spin" />
                          Pending
                        </Badge>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
