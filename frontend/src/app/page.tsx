"use client"

import { useEffect } from "react"
import { useRouter } from "next/navigation"
import { AppProvider, useApp } from "@/context/AppContext"
import { useAuth } from "@/context/AuthContext"
import { DashboardShell } from "@/components/layout/DashboardShell"
import { OverviewTab } from "@/components/dashboard/overview/OverviewTab"
import { RagChatTab } from "@/components/dashboard/chat/RagChatTab"
import { ComparisonTab } from "@/components/dashboard/comparison/ComparisonTab"
import { IngestionTab } from "@/components/dashboard/ingestion/IngestionTab"
import { AnalyticsTab } from "@/components/dashboard/analytics/AnalyticsTab"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"

function TabContent() {
  const { activeTab } = useApp()

  switch (activeTab) {
    case "overview":
      return <OverviewTab />
    case "chat":
      return <RagChatTab />
    case "comparison":
      return <ComparisonTab />
    case "ingestion":
      return <IngestionTab />
    case "analytics":
      return <AnalyticsTab />
    case "documents":
      return (
        <Card>
          <CardHeader>
            <CardTitle>10-K Filings & Documents</CardTitle>
            <CardDescription>Manage and browse ingested SEC filings</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-muted-foreground">Document manager coming in a later prompt...</p>
          </CardContent>
        </Card>
      )
    case "admin_logs":
      return (
        <Card>
          <CardHeader>
            <CardTitle>Guardrails & Audit Logs</CardTitle>
            <CardDescription>System audit trail and guardrail status</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-muted-foreground">Audit logs coming in a later prompt...</p>
          </CardContent>
        </Card>
      )
    default:
      return null
  }
}

function DashboardPage() {
  const { isAuthenticated, isLoading } = useAuth()
  const router = useRouter()

  useEffect(() => {
    if (!isLoading && !isAuthenticated) {
      router.push("/login")
    }
  }, [isLoading, isAuthenticated, router])

  if (isLoading) {
    return (
      <div className="flex h-screen items-center justify-center">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
      </div>
    )
  }

  if (!isAuthenticated) return null

  return (
    <AppProvider>
      <DashboardShell>
        <TabContent />
      </DashboardShell>
    </AppProvider>
  )
}

export default DashboardPage
