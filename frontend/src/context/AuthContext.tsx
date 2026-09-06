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
  apiPost,
  apiGet,
  apiLogout,
  setAccessToken,
  getAccessToken,
  type UserProfile,
  type TokenResponse,
  type LoginPayload,
  type SignupPayload,
} from "@/lib/api"

interface AuthState {
  user: UserProfile | null
  isAuthenticated: boolean
  isLoading: boolean
  login: (credentials: LoginPayload) => Promise<void>
  signup: (payload: SignupPayload) => Promise<void>
  logout: () => Promise<void>
  fetchProfile: () => Promise<void>
}

const AuthContext = createContext<AuthState | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<UserProfile | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  const fetchProfile = useCallback(async () => {
    try {
      const profile = await apiGet<UserProfile>("/api/v1/auth/me")
      setUser(profile)
      document.cookie = `x-role=${profile.role};path=/;SameSite=Lax`
    } catch {
      setUser(null)
      setAccessToken(null)
      document.cookie = "x-role=;path=/;max-age=0"
    }
  }, [])

  useEffect(() => {
    const token = getAccessToken()
    if (token) {
      fetchProfile().finally(() => setIsLoading(false))
    } else {
      setIsLoading(false)
    }
  }, [fetchProfile])

  const login = useCallback(
    async (credentials: LoginPayload) => {
      const data = await apiPost<TokenResponse>(
        "/api/v1/auth/login",
        credentials,
        { noAuth: true }
      )
      setAccessToken(data.access_token)
      await fetchProfile()
    },
    [fetchProfile]
  )

  const signup = useCallback(
    async (payload: SignupPayload) => {
      const data = await apiPost<TokenResponse>(
        "/api/v1/auth/signup",
        payload,
        { noAuth: true }
      )
      setAccessToken(data.access_token)
      await fetchProfile()
    },
    [fetchProfile]
  )

  const logout = useCallback(async () => {
    await apiLogout()
    setUser(null)
    document.cookie = "x-role=;path=/;max-age=0"
  }, [])

  return (
    <AuthContext.Provider
      value={{
        user,
        isAuthenticated: !!user,
        isLoading,
        login,
        signup,
        logout,
        fetchProfile,
      }}
    >
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error("useAuth must be used within AuthProvider")
  return ctx
}
