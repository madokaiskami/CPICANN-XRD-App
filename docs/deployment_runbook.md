# Deployment Runbook

## Profiles

Web app only:

```bash
docker compose up -d --build
```

Web + API:

```bash
docker compose --profile api up -d --build
```

Complete local/intranet deployment, including Web, API and XDecomposer worker:

```bash
docker compose \
  -f compose.yaml \
  -f compose.xdecomposer.yaml \
  --profile api \
  --profile xdecomposer \
  up -d --build
```

Check container state and endpoints:

```bash
docker compose -f compose.yaml -f compose.xdecomposer.yaml --profile api --profile xdecomposer ps
curl -fsS http://127.0.0.1:8501/
curl -fsS http://127.0.0.1:8000/healthz
curl -fsS http://127.0.0.1:8000/capabilities
curl -fsS http://127.0.0.1:8100/readyz
```

Follow logs:

```bash
docker compose -f compose.yaml -f compose.xdecomposer.yaml --profile api --profile xdecomposer logs -f app api xdecomposer-worker
```

Stop the complete deployment:

```bash
docker compose -f compose.yaml -f compose.xdecomposer.yaml --profile api --profile xdecomposer down
```

GPU worker overlay:

```bash
docker compose \
  -f compose.yaml \
  -f compose.xdecomposer.yaml \
  -f compose.gpu.yaml \
  --profile xdecomposer-gpu \
  up -d --build
```

`compose.gpu.yaml` requests one NVIDIA GPU for the XDecomposer worker. Use it
only after the host driver and container runtime have been validated.

## Asset Mounts

Mount model assets read-only:

```text
./models:/app/models:ro
./models/xdecomposer:/app/models/xdecomposer:ro
```

Write run outputs to a persistent writable volume:

```text
./runs:/app/runs
```

Do not bake weights, `.env`, tokens or reference banks into images.

For complete XDecomposer deployment, the host must provide:

```text
models/xdecomposer/manifest.yaml
models/xdecomposer/checkpoints/xdecomposer/latest.pt
models/xdecomposer/checkpoints/pretrain/checkpoint_latest.pt
```

Use `reference_bank_required: false` for decomposition-only mode. Reference-bank
matching requires an additional reference bank path and license acceptance in
the manifest.

## Cloud Notes

- Put API and Web behind HTTPS reverse proxy such as Caddy or Nginx.
- Example reverse-proxy configs are provided in `deploy/Caddyfile` and
  `deploy/nginx.conf`; replace `cpicann.example.invalid` before use.
- Restrict access to trusted internal users unless authentication is added.
- Keep secrets in environment or platform secret stores.
- Confirm CUDA driver, container runtime and GPU visibility before enabling the
  XDecomposer GPU worker.
- Use immutable image tags or digests.
- Monitor `/healthz`, `/readyz`, `/capabilities` and `/metrics`.

## Backup and Restore

Back up:

- authorized `models/` manifests and asset hashes;
- persistent `runs/` volume when retention policy requires it;
- deployment compose files and environment variables;
- release notes with image digests.

Restore:

1. Stop app/API/worker containers.
2. Restore `models/` and `runs/` volumes.
3. Start the previous immutable image tag or digest.
4. Check `/healthz`, `/readyz` and a known single-sample smoke test.

## Troubleshooting

- `XDecomposer unavailable: xdecomposer_disabled`: enable only after assets and
  capability checks are ready.
- `XDecomposer unavailable: service_unavailable`: check that
  `xdecomposer-worker` is running and that the app/API containers can reach
  `http://xdecomposer-worker:8100/readyz`.
- `assets_invalid`: run `cpicann-xrd xdecomposer verify-assets --production`.
- `NO_CANDIDATES_AFTER_FILTER`: element constraints removed every catalog
  candidate.
- Docker permission denied: verify the user is in the `docker` group and the
  socket is owned by `root:docker` with mode `660`.
- Web cannot write output: make the host `runs/` directory writable by the
  container user.
