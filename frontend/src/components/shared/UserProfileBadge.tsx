"use client"

import { useAuth } from "@/context/AuthContext"
import { useRouter } from "next/navigation"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { Badge } from "@/components/ui/badge"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { LogOut, User, LogIn } from "lucide-react"

interface UserProfileBadgeProps {
  collapsed?: boolean
}

export function UserProfileBadge({ collapsed }: UserProfileBadgeProps) {
  const { user, isAuthenticated, logout } = useAuth()
  const router = useRouter()

  if (!isAuthenticated || !user) {
    if (collapsed) {
      return (
        <div className="flex justify-center">
          <button
            onClick={() => router.push("/login")}
            className="flex h-8 w-8 items-center justify-center rounded-full bg-sidebar-accent text-sidebar-accent-foreground hover:bg-sidebar-accent/80"
          >
            <LogIn className="h-4 w-4" />
          </button>
        </div>
      )
    }
    return (
      <button
        onClick={() => router.push("/login")}
        className="flex w-full items-center gap-3 rounded-lg p-2 text-left text-sm text-muted-foreground transition-colors hover:bg-sidebar-accent hover:text-sidebar-accent-foreground"
      >
        <LogIn className="h-4 w-4" />
        <span>Sign In</span>
      </button>
    )
  }

  const initials = user.full_name
    .split(" ")
    .map((n) => n[0])
    .join("")
    .toUpperCase()

  if (collapsed) {
    return (
      <div className="flex justify-center">
        <Avatar className="h-8 w-8">
          <AvatarFallback className="bg-primary text-xs text-primary-foreground">
            {initials}
          </AvatarFallback>
        </Avatar>
      </div>
    )
  }

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button className="flex w-full items-center gap-3 rounded-lg p-2 text-left transition-colors hover:bg-sidebar-accent">
          <Avatar className="h-8 w-8">
            <AvatarFallback className="bg-primary text-xs text-primary-foreground">
              {initials}
            </AvatarFallback>
          </Avatar>
          <div className="flex flex-1 flex-col overflow-hidden">
            <span className="truncate text-sm font-medium text-sidebar-foreground">
              {user.full_name}
            </span>
            <Badge
              variant={user.role === "admin" ? "default" : "secondary"}
              className="mt-0.5 w-fit px-1.5 py-0 text-[10px]"
            >
              {user.role === "admin" ? "Admin" : "User"}
            </Badge>
          </div>
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-56">
        <DropdownMenuLabel>
          <div className="flex flex-col">
            <span>{user.full_name}</span>
            <span className="text-xs font-normal text-muted-foreground">{user.email}</span>
          </div>
        </DropdownMenuLabel>
        <DropdownMenuSeparator />
        <DropdownMenuItem>
          <User className="mr-2 h-4 w-4" />
          Profile
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuItem className="text-destructive" onClick={() => { logout(); router.push("/login") }}>
          <LogOut className="mr-2 h-4 w-4" />
          Logout
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
