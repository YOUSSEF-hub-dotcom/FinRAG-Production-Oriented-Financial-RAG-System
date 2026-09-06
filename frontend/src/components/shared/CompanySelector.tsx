"use client"

import { useApp, SUPPORTED_TICKERS } from "@/context/AppContext"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Building2 } from "lucide-react"

interface CompanySelectorProps {
  collapsed?: boolean
}

export function CompanySelector({ collapsed }: CompanySelectorProps) {
  const { activeTicker, setActiveTicker } = useApp()

  if (collapsed) {
    return (
      <div className="flex justify-center">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-sidebar-accent text-sidebar-accent-foreground">
          <Building2 className="h-4 w-4" />
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-1.5">
      <label className="text-xs font-medium text-muted-foreground">Active Ticker</label>
      <Select
        value={activeTicker.symbol}
        onValueChange={(symbol) => {
          const ticker = SUPPORTED_TICKERS.find((t) => t.symbol === symbol)
          if (ticker) setActiveTicker(ticker)
        }}
      >
        <SelectTrigger className="h-9 border-sidebar-border bg-sidebar-accent text-sidebar-accent-foreground">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {SUPPORTED_TICKERS.map((ticker) => (
            <SelectItem key={ticker.symbol} value={ticker.symbol}>
              <div className="flex items-center gap-2">
                <span className="font-semibold">{ticker.symbol}</span>
                <span className="text-muted-foreground">
                  {ticker.symbol === "ALL"
                    ? "- Cross-Entity"
                    : `- ${ticker.name}`}
                </span>
              </div>
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  )
}
