"use client"

import { useApp } from "@/context/AppContext"
import { MOCK_COMPANIES } from "@/lib/mockFinancialData"
import { CompanyHeader } from "./CompanyHeader"
import { FinancialMetricCards } from "./FinancialMetricCards"
import { FinancialPerformanceChart } from "./FinancialPerformanceChart"
import { FilingsOverviewCard } from "./FilingsOverviewCard"
import { Card, CardContent } from "@/components/ui/card"
import { AlertCircle } from "lucide-react"

export function OverviewTab() {
  const { activeTicker, setActiveTab } = useApp()
  const company = MOCK_COMPANIES[activeTicker.symbol]

  if (!company) {
    return (
      <Card>
        <CardContent className="flex flex-col items-center justify-center py-12 text-center">
          <AlertCircle className="mb-4 h-8 w-8 text-muted-foreground" />
          <h3 className="text-lg font-semibold">No Data Available</h3>
          <p className="text-sm text-muted-foreground">
            Financial data for {activeTicker.symbol} is not yet available.
          </p>
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="space-y-6">
      <CompanyHeader company={company} />
      <FinancialMetricCards ratios={company.keyRatios} />
      <FinancialPerformanceChart performance={company.performance} ticker={company.ticker} />
      <FilingsOverviewCard
        company={company}
        onAnalyze={() => setActiveTab("chat")}
      />
    </div>
  )
}
