# XDecomposer Asset Licenses

日期：2026-07-15

## Scope

XD-1 establishes a closed asset gate for XDecomposer. It does not download,
redistribute, or load real XDecomposer checkpoints. It only defines the manifest
fields that a future authorized deployment must provide.

## Source Code

| Item | Status |
| --- | --- |
| Upstream repository | `https://github.com/Licht0812/XDecomposer` |
| Reviewed commit | `48c4efe3681256cd16d3b8d2f56d665cf58fd129` |
| Source license | MIT |
| Required notice | Preserve upstream MIT license notice when source code is used or packaged. |

## Assets Requiring Human Approval

The following assets are not covered by this repository's license and must be
approved before real use:

| Asset | Example manifest field | Current status | Required human decision |
| --- | --- | --- | --- |
| Separator checkpoint | `separator_checkpoint` | UNKNOWN | May it be used, stored on the server, and redistributed? |
| MAE pretrain checkpoint | `mae_checkpoint` | UNKNOWN | May it be used, stored on the server, and redistributed? |
| Reference bank or database | `reference_bank` | UNKNOWN | May it be used and redistributed? Does it contain COD, MP20, RRUFF, or derived data? |
| MP20-derived data | `dataset_license` | UNKNOWN | Confirm usage boundary and attribution. |
| RRUFF-derived data | `dataset_license` | UNKNOWN | Confirm usage boundary and attribution. |

## Production Gate

Production startup must reject a manifest when any of these fields is still
`UNKNOWN`:

- `source_license`
- `checkpoint_license`
- `dataset_license`

The verifier also fails closed when any declared asset is missing, has an
invalid SHA-256 value, or does not match the declared SHA-256.

## Git and Image Boundary

Allowed in Git:

- `models/xdecomposer/README.md`
- `models/xdecomposer/manifest.example.yaml`
- Documentation and tests

Not allowed in Git or public images:

- `.pt`, `.pth`, `.ckpt`, `.safetensors`, `.onnx`
- real separator checkpoints
- real MAE checkpoints
- reference bank files
- MP20/RRUFF/COD-derived databases unless separately approved
- generated run directories
- `.env` files, tokens, or private URLs

## Manual Acceptance Checklist

Codex cannot complete these decisions:

1. Confirm checkpoint license and redistribution rights.
2. Confirm MAE pretrain checkpoint license and redistribution rights.
3. Confirm reference bank/database license and redistribution rights.
4. Confirm whether generated reports may include COD IDs, formulas, structures,
   or space-group metadata from the selected reference database.
5. Confirm exact SHA-256 values from trusted local files.
6. Decide whether third-party notices must be added to `NOTICE`.
