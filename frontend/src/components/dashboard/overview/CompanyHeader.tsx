"use client"

import { Badge } from "@/components/ui/badge"
import { Building2, Calendar, FileText } from "lucide-react"
import type { CompanyData } from "@/lib/mockFinancialData"

interface CompanyHeaderProps {
  company: CompanyData
}

export function CompanyHeader({ company }: CompanyHeaderProps) {
  return (
    <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
      <div className="flex items-center gap-4">
        <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-primary/10 text-primary">
          <Building2 className="h-6 w-6" />
        </div>
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-2xl font-bold tracking-tight">{company.name}</h2>
            <Badge variant="outline" className="font-mono text-xs">
              {company.ticker}
            </Badge>
          </div>
          <p className="text-sm text-muted-foreground">
            {company.sector} &middot; {company.industry}
          </p>
        </div>
      </div>
      <div className="flex items-center gap-4 text-sm text-muted-foreground">
        <div className="flex items-center gap-1.5">
          <span className="font-medium text-foreground">{company.marketCap}</span>
          <span>Market Cap</span>
        </div>
        <div className="hidden h-4 w-px bg-border sm:block" />
        <div className="flex items-center gap-1.5">
          <FileText className="h-3.5 w-3.5" />
          <Badge variant={company.filingStatus === "Indexed" ? "success" : "secondary"} className="text-[10px]">
            10-K {company.filingStatus}
          </Badge>
        </div>
        <div className="hidden h-4 w-px bg-border sm:block" />
        <div className="flex items-center gap-1.5">
          <Calendar className="h-3.5 w-3.5" />
          <span>{company.lastFilingDate}</span>
        </div>
      </div>
    </div>
  )
}
