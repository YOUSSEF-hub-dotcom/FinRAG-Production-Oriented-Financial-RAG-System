"use client"

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Area,
  AreaChart,
  Line,
} from "recharts"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { useState } from "react"
import { Button } from "@/components/ui/button"
import type { YearlyPerformance } from "@/lib/mockFinancialData"

interface FinancialPerformanceChartProps {
  performance: YearlyPerformance[]
  ticker: string
}

export function FinancialPerformanceChart({ performance, ticker }: FinancialPerformanceChartProps) {
  const [chartType, setChartType] = useState<"bar" | "area">("bar")

  const data = performance.map((p) => ({
    year: p.year.toString(),
    Revenue: Number(p.revenue.toFixed(1)),
    "Net Income": Number(p.netIncome.toFixed(1)),
    "R&D Spend": Number(p.rdSpend.toFixed(1)),
  }))

  return (
    <Card className="col-span-full">
      <CardHeader className="flex flex-row items-center justify-between space-y-0">
        <div>
          <CardTitle>Financial Performance</CardTitle>
          <CardDescription>Revenue, Net Income & R&amp;D Spend (2023-2025) for {ticker}</CardDescription>
        </div>
        <div className="flex gap-1">
          <Button
            variant={chartType === "bar" ? "default" : "ghost"}
            size="sm"
            onClick={() => setChartType("bar")}
            className="h-8 text-xs"
          >
            Bar
          </Button>
          <Button
            variant={chartType === "area" ? "default" : "ghost"}
            size="sm"
            onClick={() => setChartType("area")}
            className="h-8 text-xs"
          >
            Area
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <div className="h-[350px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            {chartType === "bar" ? (
              <BarChart data={data} barGap={4}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis
                  dataKey="year"
                  tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }}
                  axisLine={{ stroke: "hsl(var(--border))" }}
                />
                <YAxis
                  tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }}
                  axisLine={{ stroke: "hsl(var(--border))" }}
                  tickFormatter={(v) => `$${v}B`}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "hsl(var(--card))",
                    border: "1px solid hsl(var(--border))",
                    borderRadius: "8px",
                    fontSize: "12px",
                  }}
                  formatter={(value) => [`$${typeof value === "number" ? value.toFixed(1) : value}B`, undefined]}
                />
                <Legend />
                <Bar dataKey="Revenue" fill="var(--chart-1)" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Net Income" fill="var(--chart-2)" radius={[4, 4, 0, 0]} />
                <Bar dataKey="R&D Spend" fill="var(--chart-3)" radius={[4, 4, 0, 0]} />
              </BarChart>
            ) : (
              <AreaChart data={data}>
                <defs>
                  <linearGradient id="colorRevenue" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="var(--chart-1)" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="var(--chart-1)" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="colorNetIncome" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="var(--chart-2)" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="var(--chart-2)" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis
                  dataKey="year"
                  tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }}
                  axisLine={{ stroke: "hsl(var(--border))" }}
                />
                <YAxis
                  tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }}
                  axisLine={{ stroke: "hsl(var(--border))" }}
                  tickFormatter={(v) => `$${v}B`}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "hsl(var(--card))",
                    border: "1px solid hsl(var(--border))",
                    borderRadius: "8px",
                    fontSize: "12px",
                  }}
                  formatter={(value) => [`$${typeof value === "number" ? value.toFixed(1) : value}B`, undefined]}
                />
                <Legend />
                <Area
                  type="monotone"
                  dataKey="Revenue"
                  stroke="var(--chart-1)"
                  fillOpacity={1}
                  fill="url(#colorRevenue)"
                />
                <Area
                  type="monotone"
                  dataKey="Net Income"
                  stroke="var(--chart-2)"
                  fillOpacity={1}
                  fill="url(#colorNetIncome)"
                />
                <Line
                  type="monotone"
                  dataKey="R&D Spend"
                  stroke="var(--chart-3)"
                  strokeWidth={2}
                  dot={{ r: 4 }}
                />
              </AreaChart>
            )}
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  )
}
