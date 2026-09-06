"use client"

import { useState } from "react"
import { LoginForm } from "./LoginForm"
import { SignupForm } from "./SignupForm"

type AuthView = "login" | "signup"

interface AuthModalProps {
  initialView?: AuthView
  onSuccess?: () => void
}

export function AuthModal({ initialView = "login", onSuccess }: AuthModalProps) {
  const [view, setView] = useState<AuthView>(initialView)

  if (view === "signup") {
    return (
      <SignupForm
        onSwitchToLogin={() => setView("login")}
        onSuccess={onSuccess}
      />
    )
  }

  return (
    <LoginForm
      onSwitchToSignup={() => setView("signup")}
      onSuccess={onSuccess}
    />
  )
}
