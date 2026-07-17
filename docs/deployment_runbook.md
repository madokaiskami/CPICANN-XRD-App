# Deployment Runbook

## Profiles

CPU-only app:

```bash
docker compose up -d --build
```

API profile:

```bash
docker compose --profile api up -d --build
```

XDecomposer worker profile:

```bash
docker compose -f compose.yaml -f compose.xdecomposer.yaml --profile xdecomposer up -d --build
```

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

## Cloud Notes

- Put API and Web behind HTTPS reverse proxy such as Caddy or Nginx.
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
- `assets_invalid`: run `cpicann-xrd xdecomposer verify-assets --production`.
- `NO_CANDIDATES_AFTER_FILTER`: element constraints removed every catalog
  candidate.
- Docker permission denied: verify the user is in the `docker` group and the
  socket is owned by `root:docker` with mode `660`.
- Web cannot write output: make the host `runs/` directory writable by the
  container user.
