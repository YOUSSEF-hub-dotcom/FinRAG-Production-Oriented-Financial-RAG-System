"use client"

import { useState } from "react"
import { useApp } from "@/context/AppContext"
import { useAuth } from "@/context/AuthContext"
import type { Theme } from "@/lib/settings"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"
import { Sun, Moon, Monitor, Database, Cpu, Globe, RotateCcw } from "lucide-react"
import { cn } from "@/lib/utils"

const THEME_OPTIONS: { value: Theme; label: string; icon: typeof Sun }[] = [
  { value: "dark", label: "Dark", icon: Moon },
  { value: "light", label: "Light", icon: Sun },
  { value: "system", label: "System", icon: Monitor },
]

const COLLECTION_OPTIONS = ["financial_vectors", "financial_vectors_test", "finsight_v2"]

export function SettingsModal() {
  const { settings, updateSettings, resetSettings, settingsOpen, setSettingsOpen } = useApp()
  const { user } = useAuth()
  const isAdmin = user?.role === "admin"
  const [draft, setDraft] = useState({
    apiBaseUrl: settings.apiBaseUrl,
    vectorCollection: settings.vectorCollection,
    chunkSize: String(settings.chunkSize),
    chunkOverlap: String(settings.chunkOverlap),
  })

  const handleOpenChange = (open: boolean) => {
    if (!isAdmin) return
    setSettingsOpen(open)
    if (open) {
      setDraft({
        apiBaseUrl: settings.apiBaseUrl,
        vectorCollection: settings.vectorCollection,
        chunkSize: String(settings.chunkSize),
        chunkOverlap: String(settings.chunkOverlap),
      })
    }
  }

  const handleSave = () => {
    updateSettings({
      apiBaseUrl: draft.apiBaseUrl.trim().replace(/\/+$/, "") || "http://localhost:8000",
      vectorCollection: draft.vectorCollection.trim() || "financial_vectors",
      chunkSize: Math.max(256, Number(draft.chunkSize) || 1024),
      chunkOverlap: Math.min(80, Math.max(0, Number(draft.chunkOverlap) || 0)),
    })
    setSettingsOpen(false)
  }

  return (
    <Dialog open={settingsOpen} onOpenChange={handleOpenChange}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Globe className="h-5 w-5 text-primary" />
            System &amp; RAG Configuration
          </DialogTitle>
          <DialogDescription>
            Configure API connectivity, vector storage, chunking preferences and UI theme. Changes
            are persisted locally in your browser.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-5 py-2">
          <div className="space-y-2">
            <Label htmlFor="api-base-url" className="flex items-center gap-1.5">
              <Globe className="h-3.5 w-3.5 text-muted-foreground" />
              API Base URL
            </Label>
            <Input
              id="api-base-url"
              value={draft.apiBaseUrl}
              onChange={(e) => setDraft((prev) => ({ ...prev, apiBaseUrl: e.target.value }))}
              placeholder="http://localhost:8000"
              className="font-mono text-xs"
            />
            <p className="text-xs text-muted-foreground">
              Base URL of the FastAPI backend (e.g. <code>http://localhost:8000</code>).
            </p>
          </div>

          <div className="space-y-2">
            <Label className="flex items-center gap-1.5">
              <Database className="h-3.5 w-3.5 text-muted-foreground" />
              Vector DB Collection
            </Label>
            <div className="flex flex-wrap gap-2">
              {COLLECTION_OPTIONS.map((col) => (
                <button
                  key={col}
                  type="button"
                  onClick={() => setDraft((prev) => ({ ...prev, vectorCollection: col }))}
                  className={cn(
                    "rounded-md border px-3 py-1.5 font-mono text-xs transition-colors",
                    draft.vectorCollection === col
                      ? "border-primary bg-primary/10 text-primary"
                      : "text-muted-foreground hover:bg-accent"
                  )}
                >
                  {col}
                </button>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="chunk-size" className="flex items-center gap-1.5">
                <Cpu className="h-3.5 w-3.5 text-muted-foreground" />
                Chunk Size (tokens)
              </Label>
              <Input
                id="chunk-size"
                type="number"
                min={256}
                step={128}
                value={draft.chunkSize}
                onChange={(e) => setDraft((prev) => ({ ...prev, chunkSize: e.target.value }))}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="chunk-overlap" className="flex items-center gap-1.5">
                <Cpu className="h-3.5 w-3.5 text-muted-foreground" />
                Overlap (%)
              </Label>
              <Input
                id="chunk-overlap"
                type="number"
                min={0}
                max={80}
                value={draft.chunkOverlap}
                onChange={(e) => setDraft((prev) => ({ ...prev, chunkOverlap: e.target.value }))}
              />
            </div>
          </div>

          <div className="space-y-2">
            <Label className="flex items-center gap-1.5">
              <Monitor className="h-3.5 w-3.5 text-muted-foreground" />
              UI Theme
            </Label>
            <div className="grid grid-cols-3 gap-2">
              {THEME_OPTIONS.map((option) => {
                const Icon = option.icon
                const isActive = settings.theme === option.value
                return (
                  <button
                    key={option.value}
                    type="button"
                    onClick={() => updateSettings({ theme: option.value })}
                    className={cn(
                      "flex flex-col items-center gap-1.5 rounded-md border px-3 py-2.5 text-xs font-medium transition-colors",
                      isActive
                        ? "border-primary bg-primary/10 text-primary"
                        : "text-muted-foreground hover:bg-accent"
                    )}
                  >
                    <Icon className="h-4 w-4" />
                    {option.label}
                  </button>
                )
              })}
            </div>
            <p className="text-xs text-muted-foreground">
              Current effective theme:{" "}
              <Badge variant="outline" className="px-1.5 py-0 text-[10px]">
                {settings.theme}
              </Badge>
            </p>
          </div>
        </div>

        <DialogFooter className="flex items-center justify-between gap-2 sm:justify-between">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => {
              resetSettings()
              setDraft({
                apiBaseUrl: "http://localhost:8000",
                vectorCollection: "financial_vectors",
                chunkSize: "1024",
                chunkOverlap: "20",
              })
            }}
          >
            <RotateCcw className="mr-1.5 h-3.5 w-3.5" />
            Reset Defaults
          </Button>
          <div className="flex gap-2">
            <Button variant="outline" size="sm" onClick={() => setSettingsOpen(false)}>
              Cancel
            </Button>
            <Button size="sm" onClick={handleSave}>
              Save Changes
            </Button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
