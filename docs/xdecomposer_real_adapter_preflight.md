# XDecomposer Real Adapter Preflight

日期：2026-07-17

## Scope

XD-4 requires a hash-verified, human-confirmed XDecomposer manifest plus real
separator checkpoint and MAE checkpoint assets. Reference-bank assets are now
optional for decomposition-only smoke tests via `reference_bank_required:
false`; they remain required for reference matching. This preflight records the
current state and does not claim a completed real-model smoke test.

## Current State

Current branch:

```text
feature/xdecomposer-cpicann-pipeline
```

Tracked XDecomposer asset placeholders:

```text
models/xdecomposer/README.md
models/xdecomposer/manifest.example.yaml
```

Local ignored XDecomposer assets now present for decomposition-only smoke tests:

```text
models/xdecomposer/manifest.yaml
models/xdecomposer/checkpoints/xdecomposer/latest.pt
models/xdecomposer/checkpoints/pretrain/checkpoint_latest.pt
```

The local manifest uses `reference_bank_required: false`. No `reference_bank.pt`
is present.

## Fail-Closed Evidence

Production gate command:

```bash
.venv/bin/cpicann-xrd xdecomposer verify-assets \
  --manifest models/xdecomposer/manifest.yaml \
  --production \
  --json
```

Result:

```json
{
  "status": "failed",
  "error": {
    "code": "XDECOMPOSER_ASSET_LICENSE_UNCONFIRMED",
    "message": "XDecomposer manifest 仍包含未确认许可字段，不能用于生产模式",
    "details": {
      "manifest_path": "models/xdecomposer/manifest.yaml",
      "fields": ["checkpoint_license"]
    }
  }
}
```

This is expected: local decomposition-only assets can be hash-verified for
smoke tests, but production mode remains blocked until checkpoint licensing is
confirmed.

Non-production asset gate:

```bash
.venv/bin/cpicann-xrd xdecomposer verify-assets \
  --manifest models/xdecomposer/manifest.yaml \
  --json
```

Verified assets:

```text
separator_checkpoint:
  path: models/xdecomposer/checkpoints/xdecomposer/latest.pt
  sha256: f33186a06ddff78319e9153d18e1fc11de68c4a1eeaa84b137eefa78bce9555b
  size_bytes: 271119842
mae_checkpoint:
  path: models/xdecomposer/checkpoints/pretrain/checkpoint_latest.pt
  sha256: 1c66067583f652a95e10f92cdc85c81137c89ffe1e1a12fbd123f85b50fcfccc
  size_bytes: 296447531
```

Real adapter smoke with those assets now runs against the bundled XDecomposer
source under `services/xdecomposer_service/vendor/XDecomposer`:

```text
status: success
model_id: xdecomposer-local-v0.0.1-decomposition
components: 4
reconstruction_error: 0.026902971789240837
warnings: []
```

The bundled source is a minimal reviewed copy of the upstream MIT-licensed model
code needed by the isolated worker. The worker no longer depends on a separate
developer-machine XDecomposer checkout.

## XD-4 Remaining Gaps

The following XD-4 prerequisites remain unmet:

1. Checkpoint license is still unconfirmed.
2. Reference bank path and SHA-256 have not been supplied for reference
   matching mode.
3. Dataset/reference-bank license remains unconfirmed for reference matching
   mode.
4. No authorized read-only model mount has been validated for the isolated
   XDecomposer container.
5. Docker image build has not completed in this environment because dependency
   installation from PyPI failed with an SSL/network error during build.

## Commands To Run After Assets Are Supplied

Asset gate:

```bash
.venv/bin/cpicann-xrd xdecomposer verify-assets \
  --manifest models/xdecomposer/manifest.yaml
```

Container readiness:

```bash
docker compose -f compose.yaml -f compose.xdecomposer.yaml \
  --profile xdecomposer up -d xdecomposer-worker
curl http://127.0.0.1:8100/readyz
```

Future real-model test marker:

```bash
.venv/bin/pytest -q -m xdecomposer_model
```

## Safe Next Codex Work

Codex can continue only with work that keeps failures explicit:

- container smoke tests with mounted checkpoints after Docker dependency
  installation succeeds;
- report/artifact generation around verified adapter outputs;
- documentation and manual acceptance checklists.

Codex must not claim production XDecomposer availability until checkpoint and
reference-bank licenses are confirmed and container smoke tests pass with the
authorized read-only model mount.
