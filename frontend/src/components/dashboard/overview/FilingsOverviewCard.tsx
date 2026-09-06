"use client"

import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { AlertTriangle, Lightbulb, MessageSquare, FileText, CheckCircle2 } from "lucide-react"
import type { CompanyData } from "@/lib/mockFinancialData"

interface FilingsOverviewCardProps {
  company: CompanyData
  onAnalyze?: () => void
}

export function FilingsOverviewCard({ company, onAnalyze }: FilingsOverviewCardProps) {
  return (
    <Card className="col-span-full">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <FileText className="h-4 w-4" />
              Filing Highlights & Analysis
            </CardTitle>
            <CardDescription>
              Key insights from {company.ticker} 10-K filing ({company.lastFilingDate})
            </CardDescription>
          </div>
          <Badge variant="success" className="gap-1">
            <CheckCircle2 className="h-3 w-3" />
            {company.filingStatus}
          </Badge>
        </div>
      </CardHeader>
      <CardContent>
        <div className="grid gap-6 md:grid-cols-2">
          <div className="space-y-3">
            <div className="flex items-center gap-2 text-sm font-medium text-destructive">
              <AlertTriangle className="h-4 w-4" />
              Key Risks
            </div>
            <ul className="space-y-2">
              {company.highlights.risks.map((risk, i) => (
                <li key={i} className="flex items-start gap-2 text-sm text-muted-foreground">
                  <span className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full bg-destructive" />
                  {risk}
                </li>
              ))}
            </ul>
          </div>
          <div className="space-y-3">
            <div className="flex items-center gap-2 text-sm font-medium text-success">
              <Lightbulb className="h-4 w-4" />
              Opportunities
            </div>
            <ul className="space-y-2">
              {company.highlights.opportunities.map((opp, i) => (
                <li key={i} className="flex items-start gap-2 text-sm text-muted-foreground">
                  <span className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full bg-success" />
                  {opp}
                </li>
              ))}
            </ul>
          </div>
        </div>
      </CardContent>
      <CardFooter>
        <Button onClick={onAnalyze} className="gap-2">
          <MessageSquare className="h-4 w-4" />
          Analyze with AI Chat
        </Button>
      </CardFooter>
    </Card>
  )
}
