# XDecomposer Real Adapter Preflight

日期：2026-07-17

## Scope

XD-4 requires a hash-verified, human-confirmed XDecomposer manifest plus real
separator checkpoint, MAE checkpoint, and reference bank assets. This preflight
records the current state and deliberately does not implement or claim a real
single-sample decomposition adapter.

## Current State

Current branch:

```text
feature/xdecomposer-real-adapter
```

Current committed base:

```text
78440c4 feat: add XDecomposer contracts, preprocessing and stub service
```

Tracked XDecomposer asset placeholders:

```text
models/xdecomposer/README.md
models/xdecomposer/manifest.example.yaml
```

No real `models/xdecomposer/manifest.yaml`, separator checkpoint, MAE
checkpoint, or reference bank was found under the project workspace.

## Fail-Closed Evidence

Command:

```bash
.venv/bin/cpicann-xrd xdecomposer verify-assets \
  --manifest models/xdecomposer/manifest.example.yaml \
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
      "manifest_path": "models/xdecomposer/manifest.example.yaml",
      "fields": ["checkpoint_license", "dataset_license"]
    }
  }
}
```

This is the expected XD-1/XD-4 safety behavior: example assets are parseable but
cannot be used to start real production inference.

## XD-4 Entry Criteria Not Yet Met

The following XD-4 prerequisites remain unmet:

1. Real `manifest.yaml` has not been supplied.
2. Separator checkpoint path and SHA-256 have not been supplied.
3. MAE checkpoint path and SHA-256 have not been supplied.
4. Reference bank path and SHA-256 have not been supplied.
5. Checkpoint license is still unconfirmed.
6. Dataset/reference-bank license is still unconfirmed.
7. Real `num_sources` and checkpoint model configuration have not been
   independently verified.
8. No authorized read-only model mount is available for the isolated
   XDecomposer container.

## Commands To Run After Assets Are Supplied

Asset gate:

```bash
.venv/bin/cpicann-xrd xdecomposer verify-assets \
  --manifest models/xdecomposer/manifest.yaml \
  --production
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

Until the asset gate passes, Codex can continue only with work that does not
load real XDecomposer checkpoints:

- report/artifact generation around stub decomposition results;
- product entry points that remain disabled unless capabilities are ready;
- job orchestration skeletons with stub workers;
- documentation and manual acceptance checklists.

Codex must not claim XD-4 completion or real XDecomposer availability until the
real asset gate and real smoke test pass.
