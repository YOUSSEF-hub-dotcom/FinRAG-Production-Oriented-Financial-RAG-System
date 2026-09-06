export type Theme = "light" | "dark" | "system"

export interface RAGSettings {
  apiBaseUrl: string
  vectorCollection: string
  chunkSize: number
  chunkOverlap: number
  theme: Theme
}

export const DEFAULT_SETTINGS: RAGSettings = {
  apiBaseUrl: process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000",
  vectorCollection: "financial_vectors",
  chunkSize: 1024,
  chunkOverlap: 20,
  theme: "system",
}

const STORAGE_KEY = "finsight-settings-v1"

function isTheme(value: unknown): value is Theme {
  return value === "light" || value === "dark" || value === "system"
}

export function loadSettings(): RAGSettings {
  if (typeof window === "undefined") return { ...DEFAULT_SETTINGS }
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)
    if (!raw) return { ...DEFAULT_SETTINGS }
    const parsed = JSON.parse(raw) as Partial<RAGSettings>
    return {
      apiBaseUrl:
        typeof parsed.apiBaseUrl === "string" && parsed.apiBaseUrl
          ? parsed.apiBaseUrl
          : DEFAULT_SETTINGS.apiBaseUrl,
      vectorCollection:
        typeof parsed.vectorCollection === "string" && parsed.vectorCollection
          ? parsed.vectorCollection
          : DEFAULT_SETTINGS.vectorCollection,
      chunkSize:
        typeof parsed.chunkSize === "number" && parsed.chunkSize > 0
          ? parsed.chunkSize
          : DEFAULT_SETTINGS.chunkSize,
      chunkOverlap:
        typeof parsed.chunkOverlap === "number" && parsed.chunkOverlap >= 0
          ? parsed.chunkOverlap
          : DEFAULT_SETTINGS.chunkOverlap,
      theme: isTheme(parsed.theme) ? parsed.theme : DEFAULT_SETTINGS.theme,
    }
  } catch {
    return { ...DEFAULT_SETTINGS }
  }
}

export function saveSettings(settings: RAGSettings): void {
  if (typeof window === "undefined") return
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(settings))
  } catch {
    // Ignore persistence failures (private mode, storage full)
  }
}

export function getApiBaseUrl(): string {
  if (typeof window === "undefined") return DEFAULT_SETTINGS.apiBaseUrl
  return loadSettings().apiBaseUrl.replace(/\/+$/, "")
}

export function getSystemTheme(): "light" | "dark" {
  if (typeof window === "undefined") return "dark"
  return window.matchMedia?.("(prefers-color-scheme: dark)")?.matches
    ? "dark"
    : "light"
}

export function applyTheme(theme: Theme): void {
  if (typeof document === "undefined") return
  const root = document.documentElement
  const effective = theme === "system" ? getSystemTheme() : theme
  root.classList.toggle("dark", effective === "dark")
  root.style.colorScheme = effective
}
