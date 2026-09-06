"use client"

import { useState, useEffect, useCallback } from "react"
import { useApp, SUPPORTED_TICKERS } from "@/context/AppContext"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Search, Building2 } from "lucide-react"

interface SearchModalProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

export function SearchModal({ open, onOpenChange }: SearchModalProps) {
  const { setActiveTicker, setActiveTab } = useApp()
  const [query, setQuery] = useState("")

  const filtered = SUPPORTED_TICKERS.filter(
    (t) =>
      t.symbol.toLowerCase().includes(query.toLowerCase()) ||
      t.name.toLowerCase().includes(query.toLowerCase())
  )

  const handleSelect = useCallback(
    (symbol: string) => {
      const ticker = SUPPORTED_TICKERS.find((t) => t.symbol === symbol)
      if (ticker) {
        setActiveTicker(ticker)
        setActiveTab("overview")
      }
      onOpenChange(false)
      setQuery("")
    },
    [setActiveTicker, setActiveTab, onOpenChange]
  )

  useEffect(() => {
    if (!open) return
    const handler = (e: KeyboardEvent) => {
      if (e.key === "k" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault()
        onOpenChange(!open)
      }
    }
    window.addEventListener("keydown", handler)
    return () => window.removeEventListener("keydown", handler)
  }, [open, onOpenChange])

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="gap-0 p-0 sm:max-w-md">
        <DialogHeader className="border-b px-4 py-3">
          <DialogTitle className="text-sm font-medium">Search Tickers</DialogTitle>
        </DialogHeader>
        <div className="flex items-center border-b px-3">
          <Search className="h-4 w-4 shrink-0 text-muted-foreground" />
          <Input
            placeholder="Search by symbol or company name..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="border-0 bg-transparent focus-visible:ring-0 focus-visible:ring-offset-0"
            autoFocus
          />
        </div>
        <div className="max-h-64 overflow-y-auto p-2">
          {filtered.length === 0 ? (
            <p className="py-6 text-center text-sm text-muted-foreground">
              No tickers found.
            </p>
          ) : (
            filtered.map((ticker) => (
              <button
                key={ticker.symbol}
                onClick={() => handleSelect(ticker.symbol)}
                className="flex w-full items-center gap-3 rounded-md px-3 py-2 text-left text-sm transition-colors hover:bg-accent hover:text-accent-foreground"
              >
                <Building2 className="h-4 w-4 shrink-0 text-muted-foreground" />
                <div className="flex flex-col">
                  <span className="font-medium">{ticker.symbol}</span>
                  <span className="text-xs text-muted-foreground">{ticker.name}</span>
                </div>
              </button>
            ))
          )}
        </div>
      </DialogContent>
    </Dialog>
  )
}
