# XDecomposer Isolated Runtime

日期：2026-07-17

## Scope

XD-2 adds an isolated runtime spike for XDecomposer. It does not install
XDecomposer dependencies into the CPICANN-XRD-App environment, does not load real
checkpoints, and does not change the default CPICANN Web/API/CLI behavior.

## Runtime Boundary

The independent service lives under `services/xdecomposer_service` and exposes:

- `GET /healthz`: process liveness.
- `GET /readyz`: Python, optional PyTorch, upstream import, manifest, and asset
  readiness.
- `GET /info`: runtime and configured asset metadata.
- `python -m xdecomposer_service.check_environment`: Docker smoke command.

The service reports `not_ready` when no manifest or assets are mounted. That is
the expected state for ordinary development and CI.

## Dependency Decision

`services/xdecomposer_service/upstream-requirements.reviewed.txt` records the
reviewed upstream dependency set from XDecomposer commit
`48c4efe3681256cd16d3b8d2f56d665cf58fd129`.

The XD-2 image deliberately installs only a minimal health-service stack:

- FastAPI
- Pydantic
- PyYAML
- Uvicorn

It does not install the upstream training stack, TensorFlow, torch-geometric,
CUDA 12.8 packages, or `skimage==0.0`. Real inference dependencies must be
revisited in XD-4 after authorized checkpoints and target GPU hosts are known.

## Docker and Compose

Build:

```bash
docker build -f docker/Dockerfile.xdecomposer -t xdecomposer-spike:test .
```

Smoke:

```bash
docker run --rm xdecomposer-spike:test python -m xdecomposer_service.check_environment
```

Opt-in Compose:

```bash
docker compose -f compose.yaml -f compose.xdecomposer.yaml config
docker compose -f compose.yaml -f compose.xdecomposer.yaml --profile xdecomposer up -d
curl --fail http://localhost:8100/healthz
curl http://localhost:8100/readyz
```

The compose service mounts `./models/xdecomposer` read-only and does not bake
weights into the image.

## Current Limitations

- The service only checks environment and assets.
- Upstream XDecomposer import is optional by default.
- No model is loaded and no decomposition endpoint exists yet.
- GPU/CUDA compatibility still requires target-host validation.
