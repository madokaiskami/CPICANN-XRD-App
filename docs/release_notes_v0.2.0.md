# Release Notes: v0.2.0-rc1

Status: experimental release candidate.

## Highlights

- CPICANN single-phase ranking remains the default workflow.
- XDecomposer modes are explicit opt-in and disabled by default.
- Stub decomposition paths are available for UI/API/CLI contract testing.
- Real XDecomposer assets require manifest, hash and license verification.
- Asynchronous job API contract is available for cloud deployment planning.
- Scientific validation framework is available, but scientific validation is
  not complete.

## Images

Release workflows build:

- CPICANN app image from `docker/Dockerfile.cpu`;
- XDecomposer worker image from `docker/Dockerfile.xdecomposer`.

Immutable tags and digests must be copied from the CI release logs before
production deployment. Do not deploy an image identified only by `latest`.

## Required Manual Gates

- CPICANN and XDecomposer asset licenses confirmed.
- XDecomposer scientific validation reviewed by domain specialists.
- Cloud security review completed.
- HTTPS, authentication, firewall and backup policy configured.
- Empty-server deployment rehearsed.
- Rollback procedure tested.

## Rollback

Use the previous immutable image tag or digest and restore the previous mounted
`models/` and `runs/` volumes. See `docs/rollback.md`.
