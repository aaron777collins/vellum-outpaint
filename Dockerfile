# Vellum — in-browser WebGPU outpainting studio
# Static Vite/React SPA, built and served behind the central Caddy proxy.
# No server runtime: stage 2 is nginx serving the prebuilt `dist/` output.

# ---- Stage 1: build ---------------------------------------------------------
FROM node:24-alpine AS build

WORKDIR /app

# Copy manifests first so dependency install is cached independently of
# source changes.
COPY package.json package-lock.json* ./

# `npm ci` requires a lockfile and gives reproducible installs; fall back to
# `npm install` if the lockfile is ever missing (e.g. a fresh clone before
# `npm install` has been run once).
RUN npm ci --no-audit --no-fund || npm install --no-audit --no-fund

COPY . .

RUN npm run build

# ---- Stage 2: runtime --------------------------------------------------------
FROM nginx:alpine AS runtime

# SPA fallback + COOP/COEP + asset caching + application/wasm MIME type.
COPY deploy/nginx.conf /etc/nginx/conf.d/default.conf

COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

# nginx:alpine's default CMD ("nginx -g daemon off;") is what we want.
