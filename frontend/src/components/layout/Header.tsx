"use client"

import { useState } from "react"
import { useApp } from "@/context/AppContext"
import { useAuth } from "@/context/AuthContext"
import { useRouter } from "next/navigation"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Sun,
  Moon,
  Activity,
  Bell,
  Search,
  LogIn,
  LogOut,
  Settings,
} from "lucide-react"
import { SettingsModal } from "@/components/dashboard/settings/SettingsModal"
import { SearchModal } from "@/components/shared/SearchModal"

const TAB_TITLES: Record<string, string> = {
  overview: "Overview & Market Hub",
  chat: "AI Financial Analyst",
  comparison: "Financial Comparison Matrix",
  ingestion: "Ingestion & Indexing Hub",
  analytics: "System Analytics & Diagnostics",
  documents: "10-K Filings & Docs",
  admin_logs: "Guardrails & Logs",
}

export function Header() {
  const { activeTab, activeTicker, theme, toggleTheme, setSettingsOpen } = useApp()
  const { isAuthenticated, user, logout } = useAuth()
  const router = useRouter()
  const [searchOpen, setSearchOpen] = useState(false)
  const title = TAB_TITLES[activeTab] ?? "Dashboard"
  const isAdmin = user?.role === "admin"

  const handleSignOut = async () => {
    await logout()
    router.push("/login")
  }

  return (
    <>
      <header className="sticky top-0 z-30 flex h-14 items-center gap-4 border-b bg-background/95 px-6 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="flex items-center gap-3">
          <h1 className="text-base font-semibold text-foreground">{title}</h1>
          <Badge variant="outline" className="gap-1 text-xs">
            <span className="h-1.5 w-1.5 rounded-full bg-success animate-pulse" />
            {activeTicker.symbol}
          </Badge>
        </div>
        <div className="flex-1" />
        <div className="flex items-center gap-2">
          <div className="hidden items-center gap-2 rounded-lg border px-3 py-1.5 text-xs sm:flex">
            <Activity className="h-3 w-3 text-success" />
            <span className="text-muted-foreground">API</span>
            <Badge variant="success" className="px-1.5 py-0 text-[10px]">
              Healthy
            </Badge>
          </div>
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8"
            onClick={() => setSearchOpen(true)}
            aria-label="Search tickers"
          >
            <Search className="h-4 w-4" />
          </Button>
          <Button variant="ghost" size="icon" className="relative h-8 w-8">
            <Bell className="h-4 w-4" />
            <span className="absolute right-1.5 top-1.5 h-2 w-2 rounded-full bg-primary" />
          </Button>
          <Button variant="ghost" size="icon" className="h-8 w-8" onClick={toggleTheme}>
            {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
          </Button>
          {isAdmin && (
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8"
              onClick={() => setSettingsOpen(true)}
              aria-label="Open settings"
            >
              <Settings className="h-4 w-4" />
            </Button>
          )}
          {isAuthenticated && user ? (
            <div className="flex items-center gap-2">
              <span className="hidden text-sm text-muted-foreground md:inline">
                {user.full_name}
              </span>
              <Button variant="ghost" size="sm" onClick={handleSignOut} className="gap-1.5">
                <LogOut className="h-4 w-4" />
                <span className="hidden md:inline">Sign Out</span>
              </Button>
            </div>
          ) : (
            <Button variant="default" size="sm" onClick={() => router.push("/login")}>
              <LogIn className="mr-2 h-4 w-4" />
              Sign In
            </Button>
          )}
        </div>
        <SettingsModal />
      </header>
      <SearchModal open={searchOpen} onOpenChange={setSearchOpen} />
    </>
  )
}
