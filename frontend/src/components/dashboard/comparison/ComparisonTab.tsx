"use client"

import { useMemo, useState } from "react"
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  Radar,
} from "recharts"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { MOCK_COMPANIES } from "@/lib/mockFinancialData"
import { ArrowRightLeft } from "lucide-react"

type ChartMode = "bar" | "radar"

const TICKERS = ["AAPL", "MSFT", "NVDA"] as const

const DEBT_LEVELS: Record<string, { label: string; value: number }> = {
  AAPL: { label: "Low (net cash)", value: 20 },
  MSFT: { label: "Low (net cash)", value: 18 },
  NVDA: { label: "Minimal", value: 8 },
}

const CHART_COLORS: Record<string, string> = {
  AAPL: "var(--chart-1)",
  MSFT: "var(--chart-2)",
  NVDA: "var(--chart-3)",
}

interface MetricRow {
  label: string
  description: string
  values: Record<string, string>
  normalized: Record<string, number>
}

function parseNumber(raw: string): number | null {
  const cleaned = raw.replace(/[$,%]/g, "")
  const num = parseFloat(cleaned)
  return Number.isFinite(num) ? num : null
}

export function ComparisonTab() {
  const [selected, setSelected] = useState<string[]>(["AAPL", "MSFT", "NVDA"])
  const [chartMode, setChartMode] = useState<ChartMode>("bar")

  const toggleTicker = (ticker: string) => {
    setSelected((prev) => {
      if (prev.includes(ticker)) {
        if (prev.length <= 2) return prev
        return prev.filter((t) => t !== ticker)
      }
      if (prev.length >= 3) return prev
      return [...prev, ticker]
    })
  }

  const rows = useMemo<MetricRow[]>(() => {
    const get = (ticker: string) => MOCK_COMPANIES[ticker]
    const pct = (value: number | null) =>
      value === null ? "—" : `${value.toFixed(1)}%`

    const revenueGrowth: Record<string, number> = {}
    const eps: Record<string, string> = {}
    const margin: Record<string, number> = {}
    const rdRatio: Record<string, number> = {}
    const debt: Record<string, number> = {}

    for (const ticker of selected) {
      const company = get(ticker)
      const latest = company.performance[company.performance.length - 1]
      const prior = company.performance[company.performance.length - 2]
      revenueGrowth[ticker] =
        prior && prior.revenue > 0
          ? ((latest.revenue - prior.revenue) / prior.revenue) * 100
          : 0
      const epsRatio = company.keyRatios.find((r) => r.label.includes("EPS"))
      eps[ticker] = epsRatio?.value ?? "—"
      const marginRatio = company.keyRatios.find((r) => r.label.includes("Net Profit Margin"))
      margin[ticker] = parseNumber(marginRatio?.value ?? "") ?? 0
      rdRatio[ticker] = latest.revenue > 0 ? (latest.rdSpend / latest.revenue) * 100 : 0
      debt[ticker] = DEBT_LEVELS[ticker]?.value ?? 0
    }

    const maxRevenueGrowth = Math.max(...selected.map((t) => revenueGrowth[t] ?? 0), 1)
    const maxMargin = Math.max(...selected.map((t) => margin[t] ?? 0), 1)
    const maxRd = Math.max(...selected.map((t) => rdRatio[t] ?? 0), 1)
    const maxDebt = Math.max(...selected.map((t) => debt[t] ?? 0), 1)

    return [
      {
        label: "Revenue Growth",
        description: "Latest fiscal year YoY growth",
        values: Object.fromEntries(selected.map((t) => [t, pct(revenueGrowth[t])])),
        normalized: Object.fromEntries(
          selected.map((t) => [t, ((revenueGrowth[t] ?? 0) / maxRevenueGrowth) * 100])
        ),
      },
      {
        label: "EPS (Diluted)",
        description: "Latest 10-K diluted earnings per share",
        values: Object.fromEntries(selected.map((t) => [t, eps[t] ?? "—"])),
        normalized: Object.fromEntries(
          selected.map((t) => [t, parseNumber(eps[t] ?? "") ?? 0])
        ),
      },
      {
        label: "Net Profit Margin",
        description: "Latest fiscal year net margin",
        values: Object.fromEntries(selected.map((t) => [t, `${margin[t].toFixed(1)}%`])),
        normalized: Object.fromEntries(selected.map((t) => [t, ((margin[t] ?? 0) / maxMargin) * 100])),
      },
      {
        label: "R&D / Revenue",
        description: "R&D spend as share of revenue",
        values: Object.fromEntries(selected.map((t) => [t, `${rdRatio[t].toFixed(1)}%`])),
        normalized: Object.fromEntries(selected.map((t) => [t, ((rdRatio[t] ?? 0) / maxRd) * 100])),
      },
      {
        label: "Debt Levels",
        description: "Relative leverage from latest filings",
        values: Object.fromEntries(
          selected.map((t) => [t, DEBT_LEVELS[t]?.label ?? "—"])
        ),
        normalized: Object.fromEntries(
          selected.map((t) => [t, ((debt[t] ?? 0) / maxDebt) * 100])
        ),
      },
    ]
  }, [selected])

  const barData = useMemo(
    () =>
      rows.map((row) => ({
        metric: row.label,
        ...row.normalized,
      })),
    [rows]
  )

  const radarData = useMemo(
    () =>
      rows.map((row) => ({
        metric: row.label,
        ...Object.fromEntries(
          selected.map((ticker) => [ticker, Math.round(row.normalized[ticker] ?? 0)])
        ),
      })),
    [rows, selected]
  )

  const companies = selected.map((t) => MOCK_COMPANIES[t])

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader className="flex flex-row flex-wrap items-center justify-between gap-3 space-y-0">
          <div>
            <CardTitle className="flex items-center gap-2">
              <ArrowRightLeft className="h-5 w-5 text-primary" />
              Cross-Ticker Comparison Matrix
            </CardTitle>
            <CardDescription>
              Select 2-3 tickers to compare key financial metrics side by side
            </CardDescription>
          </div>
          <div className="flex flex-wrap gap-2">
            {TICKERS.map((ticker) => {
              const isSelected = selected.includes(ticker)
              const disabled = selected.length >= 3 && !isSelected
              return (
                <button
                  key={ticker}
                  type="button"
                  disabled={disabled}
                  onClick={() => toggleTicker(ticker)}
                  className={cn(
                    "rounded-md border px-3 py-1.5 text-xs font-semibold transition-colors",
                    isSelected
                      ? "border-primary bg-primary text-primary-foreground"
                      : "text-muted-foreground hover:bg-accent",
                    disabled && "cursor-not-allowed opacity-40"
                  )}
                >
                  {ticker}
                </button>
              )
            })}
          </div>
        </CardHeader>
      </Card>

      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0">
          <div>
            <CardTitle>Key Financial Metrics</CardTitle>
            <CardDescription>Latest fiscal year comparisons across selected tickers</CardDescription>
          </div>
          <div className="flex gap-1">
            <Button
              variant={chartMode === "bar" ? "default" : "ghost"}
              size="sm"
              onClick={() => setChartMode("bar")}
              className="h-8 text-xs"
            >
              Grouped Bar
            </Button>
            <Button
              variant={chartMode === "radar" ? "default" : "ghost"}
              size="sm"
              onClick={() => setChartMode("radar")}
              className="h-8 text-xs"
            >
              Radar
            </Button>
          </div>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="overflow-x-auto">
            <table className="w-full min-w-[640px] border-collapse text-sm">
              <thead>
                <tr className="border-b text-left">
                  <th className="px-3 py-2.5 text-xs font-medium uppercase tracking-wider text-muted-foreground">
                    Metric
                  </th>
                  {companies.map((company) => (
                    <th key={company.ticker} className="px-3 py-2.5 text-right">
                      <div className="flex flex-col items-end">
                        <span className="font-semibold">{company.ticker}</span>
                        <span className="text-[10px] font-normal text-muted-foreground">
                          {company.name}
                        </span>
                      </div>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr key={row.label} className="border-b transition-colors hover:bg-muted/50">
                    <td className="px-3 py-2.5">
                      <div className="font-medium">{row.label}</div>
                      <div className="text-[11px] text-muted-foreground">{row.description}</div>
                    </td>
                    {selected.map((ticker) => (
                      <td key={ticker} className="px-3 py-2.5 text-right font-mono font-semibold">
                        {row.values[ticker]}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="h-[340px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              {chartMode === "bar" ? (
                <BarChart data={barData} barGap={6}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                  <XAxis
                    dataKey="metric"
                    tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 11 }}
                    axisLine={{ stroke: "hsl(var(--border))" }}
                    interval={0}
                  />
                  <YAxis
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
                    formatter={(value) => [`${Number(value).toFixed(0)}`, ""]}
                  />
                  <Legend />
                  {selected.map((ticker) => (
                    <Bar
                      key={ticker}
                      dataKey={ticker}
                      fill={CHART_COLORS[ticker]}
                      radius={[4, 4, 0, 0]}
                    />
                  ))}
                </BarChart>
              ) : (
                <RadarChart data={radarData}>
                  <PolarGrid stroke="hsl(var(--border))" />
                  <PolarAngleAxis
                    dataKey="metric"
                    tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 11 }}
                  />
                  {selected.map((ticker) => (
                    <Radar
                      key={ticker}
                      name={ticker}
                      dataKey={ticker}
                      stroke={CHART_COLORS[ticker]}
                      fill={CHART_COLORS[ticker]}
                      fillOpacity={0.25}
                    />
                  ))}
                  <Legend />
                </RadarChart>
              )}
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      <div className="grid gap-4 sm:grid-cols-3">
        {companies.map((company) => {
          const latest = company.performance[company.performance.length - 1]
          return (
            <Card key={company.ticker}>
              <CardHeader className="space-y-0 pb-2">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-sm">{company.ticker}</CardTitle>
                  <Badge
                    variant="outline"
                    className="font-mono text-[10px] text-muted-foreground"
                  >
                    FY {latest.year}
                  </Badge>
                </div>
                <CardDescription className="text-xs">{company.name}</CardDescription>
              </CardHeader>
              <CardContent className="space-y-1.5 text-xs">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Market Cap</span>
                  <span className="font-semibold">{company.marketCap}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Revenue</span>
                  <span className="font-semibold">${latest.revenue.toFixed(1)}B</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Net Income</span>
                  <span className="font-semibold">${latest.netIncome.toFixed(1)}B</span>
                </div>
              </CardContent>
            </Card>
          )
        })}
      </div>
    </div>
  )
}
