import { NextResponse } from "next/server"
import type { NextRequest } from "next/server"

const ROLE_COOKIE = "x-role"
const PUBLIC_PATHS = new Set(["/login", "/signup"])

export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl
  const role = request.cookies.get(ROLE_COOKIE)?.value
  const isAuthenticated = !!role

  if (PUBLIC_PATHS.has(pathname)) {
    if (isAuthenticated) {
      return NextResponse.redirect(new URL("/", request.url))
    }
    return NextResponse.next()
  }

  if (pathname === "/") {
    if (!isAuthenticated) {
      return NextResponse.redirect(new URL("/login", request.url))
    }
    return NextResponse.next()
  }

  if (pathname.startsWith("/")) {
    const segment = pathname.split("/")[1]
    if (segment && !segment.startsWith("_")) {
      if (!isAuthenticated) {
        return NextResponse.redirect(new URL("/login", request.url))
      }
      if (segment === "ingestion" || segment === "analytics" || segment === "admin_logs") {
        if (role !== "admin") {
          return NextResponse.redirect(new URL("/", request.url))
        }
      }
    }
  }

  return NextResponse.next()
}

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico|api).*)"],
}
