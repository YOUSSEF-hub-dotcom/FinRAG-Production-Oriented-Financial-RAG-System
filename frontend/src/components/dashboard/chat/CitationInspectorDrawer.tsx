"use client"

import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent } from "@/components/ui/card"
import { FileText, Hash, BarChart3, Calendar, MapPin } from "lucide-react"
import type { SourceChunk } from "@/lib/ragStream"
import { cleanMetaValue, resolveChunkText } from "@/lib/ragStream"

interface CitationInspectorDrawerProps {
  source: SourceChunk | null
  open: boolean
  onOpenChange: (open: boolean) => void
}

function resolveTicker(src: SourceChunk): string | undefined {
  return cleanMetaValue(src.ticker, src.ticker_tag)
}

function resolveYear(src: SourceChunk): string | undefined {
  return cleanMetaValue(src.fiscal_year, src.year)
}

function resolveSection(src: SourceChunk): string | undefined {
  return cleanMetaValue(src.section, src.doc_type)
}

function resolveSourceFile(src: SourceChunk): string | undefined {
  return cleanMetaValue(src.source_file, src.filename, src.file)
}

export function CitationInspectorDrawer({
  source,
  open,
  onOpenChange,
}: CitationInspectorDrawerProps) {
  if (!source) return null

  const chunkText = resolveChunkText(source)
  const ticker = resolveTicker(source)
  const fiscalYear = resolveYear(source)
  const section = resolveSection(source)
  const sourceFile = resolveSourceFile(source)

  const metaItems = [
    { icon: FileText, label: "Section", value: section },
    { icon: Calendar, label: "Fiscal Year", value: fiscalYear },
    { icon: MapPin, label: "Ticker", value: ticker },
    { icon: Hash, label: "Chunk ID", value: source.chunk_id?.slice(0, 12) },
    { icon: BarChart3, label: "Relevance", value: source.score ? `${(source.score * 100).toFixed(1)}%` : undefined },
  ].filter((item) => item.value)

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent className="w-full sm:max-w-lg overflow-y-auto">
        <SheetHeader>
          <SheetTitle className="flex items-center gap-2">
            <FileText className="h-4 w-4" />
            Source Inspector
          </SheetTitle>
          <SheetDescription>
            {sourceFile || "SEC Filing Source"}
          </SheetDescription>
        </SheetHeader>
        <div className="mt-6 space-y-4">
          <div className="flex flex-wrap gap-2">
            {fiscalYear && (
              <Badge variant="outline">{fiscalYear}</Badge>
            )}
            {ticker && (
              <Badge variant="outline">{ticker}</Badge>
            )}
            {section && (
              <Badge variant="secondary">{section}</Badge>
            )}
          </div>
          {metaItems.length > 0 && (
            <Card>
              <CardContent className="grid grid-cols-2 gap-3 p-4">
                {metaItems.map((item) => (
                  <div key={item.label} className="space-y-1">
                    <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                      <item.icon className="h-3 w-3" />
                      {item.label}
                    </div>
                    <p className="text-sm font-medium truncate">{item.value}</p>
                  </div>
                ))}
              </CardContent>
            </Card>
          )}
          <div className="space-y-2">
            <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
              Extracted Context
            </h4>
            <div className="rounded-lg bg-muted p-4 text-sm leading-relaxed whitespace-pre-wrap">
              {chunkText || <span className="italic text-muted-foreground">No context text available</span>}
            </div>
          </div>
        </div>
      </SheetContent>
    </Sheet>
  )
}
