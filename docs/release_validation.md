# Release Validation v0.1.0-rc1

## Fixed Assets

- Model: `cpicann-single-d1`
- Source revision: `3dbfaeab51d272e013d211c7f957760b46ab41cc`
- State-dict SHA-256:
  `0f3a452da5218df46eaa37f7d0dadb388e08cdeafa6e24e9d6e2e93512bb2e9a`
- Catalog SHA-256:
  `393fd648778c2f788efed0d29141051d4214f89152e46ef262e7213bc164e51f`
- Catalog records: `23073`
- Golden baseline: `tests/golden/real_model_samples.json`

## Golden Coverage

The real-model golden baseline covers `0-norm.txt`, `1-norm.txt`, and
`3-norm.txt` from `samples/CPICANN识别`.

For each sample it records:

- input and preprocessed tensor hashes;
- unfiltered Top-5 predictions;
- filtered Top-5 predictions for `include_must={Zr,O}` and
  `allowed_elements={Li,Zr,O}`;
- candidate counts before and after filtering;
- COD ID, formula, reduced formula, elements, space group, logits, probability,
  and filtered confidence;
- explicit absolute tolerances for floating-point values.

## Interface Consistency

`tests/integration/test_real_model.py` validates that direct service/Web helper,
CLI, and API paths produce the same filtered Top-5 signature for the golden
sample set.

## Local Validation

Validated on 2026-07-13 with locally installed authorized weights:

```bash
uv sync --frozen --all-extras
uv run ruff format --check .
uv run ruff check .
uv run mypy src scripts
uv run pytest -q -m "not model"
uv run pytest -q -m model
uv run python -m build
uv run cpicann-xrd doctor --backend cpicann
uv run cpicann-xrd batch \
  --backend cpicann \
  --input examples/spectra \
  --include-must Zr O \
  --allowed-elements Li Zr O \
  --top-k 5 \
  --output /tmp/cpicann-release-test

docker build -f docker/Dockerfile.cpu -t cpicann-xrd-app:rc1 .
docker run --rm \
  -v "$PWD/models:/app/models:ro" \
  -v "$PWD/examples:/app/examples:ro" \
  -v "<host-output-dir>:/app/runs" \
  cpicann-xrd-app:rc1 \
  cpicann-xrd batch \
  --backend cpicann \
  --input /app/examples/spectra \
  --include-must Zr O \
  --allowed-elements Li Zr O \
  --top-k 5 \
  --output /app/runs
```

The Docker output mount differs from the generic `/tmp/cpicann-container-test`
example because the runtime container uses a non-root user. The host output
directory must be writable by that container user.

## Redistribution Boundary

Release artifacts and Docker images must not contain:

- pretrained weights;
- checkpoint files;
- `.env` files;
- tokens or private URLs;
- generated run output.

Authorized users mount `./models` at runtime for real CPICANN inference.
