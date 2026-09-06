"use client"

import { useApp } from "@/context/AppContext"
import { useAuth } from "@/context/AuthContext"
import { CompanySelector } from "@/components/shared/CompanySelector"
import { UserProfileBadge } from "@/components/shared/UserProfileBadge"
import {
  LayoutDashboard,
  MessageSquare,
  FileText,
  Shield,
  PanelLeftClose,
  PanelLeft,
  TrendingUp,
  GitCompareArrows,
  DatabaseZap,
  BarChart3,
} from "lucide-react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import type { TabId } from "@/context/AppContext"

const NAV_ITEMS: { id: TabId; label: string; icon: React.ElementType; adminOnly?: boolean }[] = [
  { id: "overview", label: "Overview & Market Hub", icon: LayoutDashboard },
  { id: "chat", label: "AI Financial Analyst", icon: MessageSquare },
  { id: "comparison", label: "Financial Comparison", icon: GitCompareArrows },
  { id: "ingestion", label: "Ingestion & Indexing", icon: DatabaseZap, adminOnly: true },
  { id: "analytics", label: "System Analytics", icon: BarChart3, adminOnly: true },
  { id: "documents", label: "10-K Filings & Docs", icon: FileText },
  { id: "admin_logs", label: "Guardrails & Logs", icon: Shield, adminOnly: true },
]

export function Sidebar() {
  const { activeTab, setActiveTab, sidebarOpen, toggleSidebar } = useApp()
  const { user } = useAuth()
  const role = user?.role ?? "user"

  const visibleItems = NAV_ITEMS.filter((item) => !item.adminOnly || role === "admin")

  return (
    <TooltipProvider delayDuration={0}>
      <aside
        className={cn(
          "relative flex h-screen flex-col border-r border-sidebar-border bg-sidebar-background text-sidebar-foreground transition-all duration-300",
          sidebarOpen ? "w-64" : "w-16"
        )}
      >
        <div className="flex h-14 items-center gap-2 border-b border-sidebar-border px-4">
          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-primary">
            <TrendingUp className="h-4 w-4 text-primary-foreground" />
          </div>
          {sidebarOpen && (
            <div className="flex flex-col">
              <span className="text-sm font-bold tracking-tight text-sidebar-foreground">
                FinSight AI
              </span>
              <span className="text-[10px] text-muted-foreground">Financial RAG Suite</span>
            </div>
          )}
          <Button
            variant="ghost"
            size="icon"
            className={cn(
              "ml-auto h-7 w-7 text-sidebar-foreground hover:bg-sidebar-accent hover:text-sidebar-accent-foreground",
              !sidebarOpen && "ml-0"
            )}
            onClick={toggleSidebar}
          >
            {sidebarOpen ? <PanelLeftClose className="h-4 w-4" /> : <PanelLeft className="h-4 w-4" />}
          </Button>
        </div>

        <nav className="flex-1 space-y-1 p-2">
          {visibleItems.map((item) => {
            const isActive = activeTab === item.id
            const Icon = item.icon
            const btn = (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={cn(
                  "flex w-full items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
                  isActive
                    ? "bg-sidebar-primary text-sidebar-primary-foreground shadow-sm"
                    : "text-sidebar-foreground hover:bg-sidebar-accent hover:text-sidebar-accent-foreground"
                )}
              >
                <Icon className="h-4 w-4 shrink-0" />
                {sidebarOpen && <span className="truncate">{item.label}</span>}
              </button>
            )
            if (!sidebarOpen) {
              return (
                <Tooltip key={item.id}>
                  <TooltipTrigger asChild>{btn}</TooltipTrigger>
                  <TooltipContent side="right">{item.label}</TooltipContent>
                </Tooltip>
              )
            }
            return btn
          })}
        </nav>

        <div className="border-t border-sidebar-border p-3">
          <CompanySelector collapsed={!sidebarOpen} />
        </div>

        <div className="border-t border-sidebar-border p-3">
          <UserProfileBadge collapsed={!sidebarOpen} />
        </div>
      </aside>
    </TooltipProvider>
  )
}
