# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Financial RAG — Frontend (Next.js 15.5.23)
#
# Multi-stage production build. Build context = ./frontend.
# Keeps next.config.ts untouched (no `output: standalone`): the runner stage
# ships the full dependency tree (see note in runner) and starts `next start`.
# ---------------------------------------------------------------------------

# ---- Base image shared by all stages (libc6-compat for native deps) ----
FROM node:20-alpine AS base
RUN apk add --no-cache libc6-compat
WORKDIR /app

# ---- Dependencies (full install; the builder stage needs devDeps) ----
FROM base AS deps
COPY package.json package-lock.json ./
RUN npm ci

# ---- Build ----
FROM base AS builder
COPY --from=deps /app/node_modules ./node_modules
COPY . .
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL \
    NEXT_TELEMETRY_DISABLED=1
RUN npm run build

# ---- Runner (production deps only, non-root) ----
FROM base AS runner
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME=0.0.0.0

COPY package.json package-lock.json ./
# Full install (incl. devDeps despite NODE_ENV=production above) so that
# `next start` can resolve next.config.ts and TypeScript at runtime without
# attempting a self-install as non-root.
RUN npm ci --include=dev && npm cache clean --force

COPY --from=builder /app/.next ./.next
COPY --from=builder /app/public ./public
COPY --from=builder /app/next.config.ts ./next.config.ts
COPY --from=builder /app/tsconfig.json ./tsconfig.json
COPY --from=builder /app/postcss.config.mjs ./postcss.config.mjs

USER node
EXPOSE 3000

CMD ["npm", "start"]