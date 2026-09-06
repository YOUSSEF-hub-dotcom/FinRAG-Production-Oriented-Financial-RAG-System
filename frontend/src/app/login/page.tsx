"use client"

import { useRouter } from "next/navigation"
import { useAuth } from "@/context/AuthContext"
import { AuthModal } from "@/components/auth/AuthModal"
import { Loader2 } from "lucide-react"
import { useEffect } from "react"

export default function LoginPage() {
  const { isAuthenticated, isLoading } = useAuth()
  const router = useRouter()

  useEffect(() => {
    if (isAuthenticated) {
      router.push("/")
    }
  }, [isAuthenticated, router])

  if (isLoading) {
    return (
      <div className="flex h-screen items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    )
  }

  if (isAuthenticated) return null

  return (
    <div className="flex min-h-screen items-center justify-center bg-background p-4">
      <AuthModal initialView="login" onSuccess={() => router.push("/")} />
    </div>
  )
}
