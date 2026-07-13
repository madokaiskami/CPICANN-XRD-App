# Security Policy

## Supported Versions

`v0.1.0-rc1` is a release candidate for local validation. It is not intended as a
public multi-tenant service.

## Reporting

Report security issues privately to the repository owner. Do not attach model
weights, private spectra, tokens, or credentials to public issues.

## Current Boundary

- The API is synchronous and intended for trusted local or internal use.
- Upload limits are enforced for Web and API entry points, but deployments should
  still place the service behind trusted network controls.
- The Docker image does not include pretrained weights, `.env` files, run output,
  or checkpoint artifacts.
- Generated run bundles may contain user-supplied filenames and prediction
  outputs. Treat them as user data.
