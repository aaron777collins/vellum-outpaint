# Deploying Vellum (outpaint.aaroncollins.info)

This is the copy-paste runbook for standing up the `outpaint` container and
wiring it into the central Caddy reverse proxy. It assumes you're on the same
host as the `caddy` container and the external `caddy` Docker network already
exists (it does — Caddy itself is attached to it).

## 1. Build and start the container

From the repo root (`/home/ubuntu/topics/stablediffusionoutpainter`):

```bash
docker compose up -d --build
```

This builds the two-stage image (Vite build → nginx runtime) and starts a
container named `outpaint`, attached to the external `caddy` network,
listening on port 80 *inside* the container only (no host port is published —
Caddy talks to it over the Docker network by container name).

Sanity-check it came up clean:

```bash
docker compose ps
docker logs outpaint --tail 50
```

## 2. Add the site to the central Caddyfile

The central Caddy config lives at `/home/ubuntu/webstack/caddy/Caddyfile` on
the host (bind-mounted into the `caddy` container at `/etc/caddy/Caddyfile`).

Append the contents of [`caddy-site.txt`](./caddy-site.txt) to that file:

```bash
cat /home/ubuntu/topics/stablediffusionoutpainter/deploy/caddy-site.txt \
  >> /home/ubuntu/webstack/caddy/Caddyfile
```

Or open `/home/ubuntu/webstack/caddy/Caddyfile` in an editor and paste in:

```caddyfile
# [NO SSO] Vellum — in-browser outpainting studio
outpaint.aaroncollins.info {
	reverse_proxy outpaint:80
}
```

## 3. Reload Caddy (no downtime for other sites)

Caddy runs as `caddy --config /etc/caddy/Caddyfile --adapter caddyfile`
(verified via `docker inspect caddy`), so a config-only reload is:

```bash
docker exec caddy caddy reload --config /etc/caddy/Caddyfile
```

This validates the new config and hot-swaps it; it will refuse to reload
(and leave the old config running) if there's a syntax error, so it's safe
to run.

If DNS for `outpaint.aaroncollins.info` isn't already pointed at this host,
set that up before reloading — otherwise Let's Encrypt's HTTP-01 challenge
(which Caddy runs automatically on first request to a new host) will fail
and Caddy will keep retrying in the background instead of serving HTTPS.

## 4. Verify

From the host (internal, via the Docker network — confirms the app container
itself is healthy):

```bash
docker exec caddy wget -qO- http://outpaint:80/ | head -20
```

From the public internet (confirms DNS + Caddy + TLS + the app, end to end):

```bash
curl -sI https://outpaint.aaroncollins.info/
```

Expect `HTTP/2 200`. Also confirm the cross-origin isolation headers that
onnxruntime-web's WebGPU/threaded-WASM backend needs are present:

```bash
curl -sI https://outpaint.aaroncollins.info/ | grep -i "cross-origin"
```

Expect to see both:

```
cross-origin-opener-policy: same-origin
cross-origin-embedder-policy: credentialless
```

## Redeploying after a code change

```bash
docker compose up -d --build
```

`docker compose` rebuilds the image (Vite re-bundles `src/` into a fresh
`dist/`) and recreates the container. No Caddy changes are needed for
ordinary code changes — only re-run steps 2–3 above if the domain, port, or
container name changes.

## Rolling back

```bash
git log --oneline -- .        # find the commit to roll back to
git checkout <commit> -- .
docker compose up -d --build
```

## Notes / assumptions

- Container name (`outpaint`) and network name (`caddy`, external) are fixed
  by `docker-compose.yml` — Caddy's `reverse_proxy outpaint:80` depends on
  both matching exactly.
- No host port is published; if you need to bypass Caddy for local
  debugging, temporarily add `ports: ["8080:80"]` under the `outpaint`
  service, or `docker exec outpaint wget -qO- http://localhost/`.
- The image bakes `dist/` in at build time — this is a pure static SPA with
  no server-side runtime, no environment variables, and no secrets, so there
  is nothing to inject at container-start time.
