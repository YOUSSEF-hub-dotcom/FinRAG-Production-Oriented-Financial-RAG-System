"use client"

import React, {
  createContext,
  useContext,
  useState,
  useCallback,
  useEffect,
  type ReactNode,
} from "react"
import {
  loadSettings,
  saveSettings,
  applyTheme,
  getSystemTheme,
  type RAGSettings,
  type Theme,
} from "@/lib/settings"

export type TabId =
  | "overview"
  | "chat"
  | "comparison"
  | "ingestion"
  | "analytics"
  | "documents"
  | "admin_logs"

export interface Ticker {
  symbol: string
  name: string
}

interface AppState {
  activeTab: TabId
  setActiveTab: (tab: TabId) => void
  activeTicker: Ticker
  setActiveTicker: (ticker: Ticker) => void
  theme: Theme
  setTheme: (theme: Theme) => void
  toggleTheme: () => void
  settings: RAGSettings
  updateSettings: (patch: Partial<RAGSettings>) => void
  resetSettings: () => void
  settingsOpen: boolean
  setSettingsOpen: (open: boolean) => void
  sidebarOpen: boolean
  setSidebarOpen: (open: boolean) => void
  toggleSidebar: () => void
}

const SUPPORTED_TICKERS: Ticker[] = [
  { symbol: "AAPL", name: "Apple Inc." },
  { symbol: "MSFT", name: "Microsoft Corporation" },
  { symbol: "NVDA", name: "NVIDIA Corporation" },
  { symbol: "ALL", name: "Cross-Entity" },
]

const AppContext = createContext<AppState | null>(null)

export function AppProvider({ children }: { children: ReactNode }) {
  const [activeTab, setActiveTab] = useState<TabId>("overview")
  const [activeTicker, setActiveTicker] = useState<Ticker>(SUPPORTED_TICKERS[0])
  const [settings, setSettings] = useState<RAGSettings>(() => loadSettings())
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [settingsOpen, setSettingsOpen] = useState(false)

  const { theme } = settings

  useEffect(() => {
    applyTheme(theme)
  }, [theme])

  useEffect(() => {
    if (theme !== "system") return
    const media = window.matchMedia?.("(prefers-color-scheme: dark)")
    if (!media) return
    const onChange = () => applyTheme("system")
    media.addEventListener?.("change", onChange)
    return () => media.removeEventListener?.("change", onChange)
  }, [theme])

  const setTheme = useCallback(
    (next: Theme) => {
      setSettings((prev) => {
        const updated = { ...prev, theme: next }
        saveSettings(updated)
        return updated
      })
    },
    []
  )

  const toggleTheme = useCallback(() => {
    setSettings((prev) => {
      const next: Theme = prev.theme === "dark" ? "light" : "dark"
      const updated = { ...prev, theme: next }
      saveSettings(updated)
      return updated
    })
  }, [])

  const updateSettings = useCallback((patch: Partial<RAGSettings>) => {
    setSettings((prev) => {
      const updated = { ...prev, ...patch }
      saveSettings(updated)
      return updated
    })
  }, [])

  const resetSettings = useCallback(() => {
    setSettings((prev) => {
      const updated = { ...loadSettings(), theme: prev.theme }
      saveSettings(updated)
      return updated
    })
  }, [])

  const toggleSidebar = useCallback(() => {
    setSidebarOpen((prev) => !prev)
  }, [])

  return (
    <AppContext.Provider
      value={{
        activeTab,
        setActiveTab,
        activeTicker,
        setActiveTicker,
        theme,
        setTheme,
        toggleTheme,
        settings,
        updateSettings,
        resetSettings,
        settingsOpen,
        setSettingsOpen,
        sidebarOpen,
        setSidebarOpen,
        toggleSidebar,
      }}
    >
      {children}
    </AppContext.Provider>
  )
}

export function useApp() {
  const ctx = useContext(AppContext)
  if (!ctx) throw new Error("useApp must be used within AppProvider")
  return ctx
}

export { SUPPORTED_TICKERS, getSystemTheme }
