# Changelog

## 0.2.0-rc1

Status: experimental release candidate.

Added:

- XDecomposer feasibility, asset manifest and license gate.
- Isolated XDecomposer worker skeleton and runtime readiness checks.
- Product schemas for decomposition requests, components and worker results.
- Deterministic `xdecomposer-v1` preprocessing and stub backend.
- Decomposition artifact bundle with CSV, JSON, Markdown, PNG and ZIP outputs.
- XDecomposer component to CPICANN ranking orchestration.
- Explicit opt-in CLI/API/Web XDecomposer entry points, disabled by default.
- Capability reporting for CPICANN-only, disabled, stub, invalid-assets and
  unavailable-service states.
- In-process asynchronous API job contract for queue, status, cancellation,
  timeout, isolated result download, audit log and Prometheus-style metrics.
- Reproducible scientific validation framework and draft validation protocol.
- Release, deployment, rollback and third-party license documentation.

Changed:

- Package version advanced to `0.2.0rc1`.
- Default CI continues to run without real CPICANN or XDecomposer weights.

Known limits:

- XDecomposer real-model inference remains experimental and requires authorized
  local assets plus human license review.
- The asynchronous job store is an in-process implementation for API contract
  validation. Production deployments should use an external durable queue.
- Scientific validation is not complete. Results must remain marked
  `experimental` until domain review is signed.

## 0.1.0-rc1

Initial CPICANN-XRD release candidate with reproducible CPICANN single-phase
ranking, CLI, Streamlit Web UI, FastAPI API, Docker deployment, catalog mapping,
report bundle generation and real-model smoke-test gates.
