# Third-Party License Notes

This repository distributes CPICANN-XRD-App product code under the license in
`LICENSE`. It does not distribute real CPICANN or XDecomposer weight files.

## Runtime Packages

Runtime packages are declared in `pyproject.toml` and locked by `uv.lock`.
Release CI exports:

- `dependency-inventory.txt`
- `sbom.cdx.json`
- `dependency-audit.json`

The generated SBOM and audit output are release artifacts, not source files.

## CPICANN Assets

The CPICANN model architecture and pretrained weights are external research
artifacts. This repository contains configuration and verification code only.
Authorized users must place weights under `models/` locally and verify the
bundle hash before real-model use.

## XDecomposer Assets

The upstream XDecomposer source repository is MIT licensed, but checkpoint,
MAE-pretrain and reference-bank redistribution rights are not established by
this repository. The production asset gate rejects manifests with `UNKNOWN`
license fields.

Tracked XDecomposer files are limited to:

- `models/xdecomposer/README.md`
- `models/xdecomposer/manifest.example.yaml`

No real XDecomposer checkpoint, MAE checkpoint, dataset cache, reference bank,
token or private credential may be committed.

## Notices

Product releases must preserve:

- this repository `LICENSE`;
- `NOTICE`;
- upstream CPICANN notices required by the authorized weight provider;
- upstream XDecomposer MIT notice when source code is packaged or mirrored;
- dataset/reference-bank attribution required by institutional review.
