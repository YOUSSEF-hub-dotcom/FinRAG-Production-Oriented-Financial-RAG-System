"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { TrendingUp, TrendingDown } from "lucide-react"
import { cn } from "@/lib/utils"
import type { KeyRatio } from "@/lib/mockFinancialData"

interface FinancialMetricCardsProps {
  ratios: KeyRatio[]
}

export function FinancialMetricCards({ ratios }: FinancialMetricCardsProps) {
  return (
    <div className="grid gap-4 grid-cols-2 lg:grid-cols-4">
      {ratios.map((ratio) => {
        const isPositive = ratio.change >= 0
        return (
          <Card key={ratio.label} className="group relative overflow-hidden transition-shadow hover:shadow-md">
            <div
              className={cn(
                "absolute inset-x-0 top-0 h-0.5 transition-colors",
                isPositive ? "bg-success" : "bg-destructive"
              )}
            />
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-xs font-medium text-muted-foreground">
                {ratio.label}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold tracking-tight">{ratio.value}</div>
              <div className="mt-1 flex items-center gap-1 text-xs">
                {isPositive ? (
                  <TrendingUp className="h-3 w-3 text-success" />
                ) : (
                  <TrendingDown className="h-3 w-3 text-destructive" />
                )}
                <span
                  className={cn(
                    "font-medium",
                    isPositive ? "text-success" : "text-destructive"
                  )}
                >
                  {isPositive ? "+" : ""}
                  {ratio.change}%
                </span>
                <span className="text-muted-foreground">vs prev year</span>
              </div>
            </CardContent>
          </Card>
        )
      })}
    </div>
  )
}
