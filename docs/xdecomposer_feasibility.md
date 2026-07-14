# XDecomposer Integration Feasibility

Date: 2026-07-14

## Scope

This document is a feasibility precheck for adding XDecomposer as an optional
multiphase XRD decomposition backend. It does not change the existing CPICANN
inference, CLI, Web, API, or report paths.

No XDecomposer checkpoint, dataset, or model weight was downloaded or committed
during this review. The official repository was inspected read-only from a
shallow clone under `/tmp/xdecomposer-official` at commit
`48c4efe3681256cd16d3b8d2f56d665cf58fd129`.

Primary sources:

- Official repository: <https://github.com/Licht0812/XDecomposer>
- Paper page: <https://arxiv.org/abs/2605.05866>
- Upstream license file: <https://raw.githubusercontent.com/Licht0812/XDecomposer/main/LICENSE>

## Executive Feasibility Result

XDecomposer integration is technically feasible, but same-process integration is
not recommended for the first implementation. The official project is a research
training/evaluation codebase with a Python 3.10 environment, PyTorch 2.10.0,
CUDA 12.8 packages, TensorFlow, torch-geometric, and dataset/checkpoint path
assumptions. The current CPICANN-XRD-App runtime is Python `>=3.11,<3.14` and
uses the CPU PyTorch package source.

The first integration should therefore be an explicit opt-in, independently
containerized XDecomposer service or batch worker. The current CPICANN single
phase classifier should remain the default backend until real XDecomposer
checkpoints, datasets/reference banks, license terms, and smoke tests are
verified.

## XDecomposer Input/Output Contract

### Upstream Model Contract

The reviewed model class is `src/models/xdecomposer.py::XDecomposer`.

Inputs:

- tensor shape: `[batch, 1, xrd_length]`
- dtype: `torch.float32`
- default `xrd_length`: `3500`
- expected normalized intensity vector; upstream dataset utilities use max
  normalization for MP20/RRUFF evaluation paths
- common experimental range in the reviewed RRUFF path: `10.0` to `80.0`
  degrees 2-theta resampled to 3500 points

Outputs:

- `component_patterns`: tensor shape `[batch, num_sources, xrd_length]`
- `activity_logits`: tensor shape `[batch, num_sources]`
- activity probability is derived by `sigmoid(activity_logits)`
- component order is an unordered set prediction result; downstream evaluation
  aligns it with targets using permutation-invariant loss

The model separates a mixture into phase-resolved intensity curves. It does not,
by itself, return COD IDs, formulas, space groups, or structure records. The
official evaluation code adds identification by comparing separated patterns
against a reference bank and returning top-k matched reference IDs.

### Proposed Product Contract

The product-level contract should be separate from the existing CPICANN
single-phase classification contract because the task semantics differ.

Request fields:

- `sample_id`: stable caller-provided ID
- `source_filename`: optional original upload name
- `two_theta`: optional numeric axis; if present, adapter resamples to the
  configured XDecomposer axis
- `intensity`: raw or preprocessed intensity values
- `preprocessing`: declared or inferred preprocessing metadata
- `max_sources`: requested maximum number of separated components
- `activity_threshold`: threshold for marking a channel active
- `reference_top_k`: number of reference-bank matches per component
- `return_component_patterns`: whether to include arrays inline or artifact
  references only

Response fields:

- `backend`: `xdecomposer`
- `model_id`: checkpoint/config identity from the asset manifest
- `preprocessing_version`: adapter preprocessing contract version
- `components`: list of separated phase components
- `reconstruction_error`: optional scalar for sum-of-components vs input
- `warnings`: non-fatal validation or confidence warnings
- `artifacts`: paths or IDs for component curves, plots, and JSON output

Component fields:

- `component_index`: model output channel index
- `active_probability`: `sigmoid(activity_logits[index])`
- `is_active`: thresholded activity flag
- `estimated_weight`: optional intensity-derived estimate; must not be reported
  as quantitative phase fraction until validated
- `separated_pattern`: optional `[3500]` array or artifact reference
- `reference_matches`: ranked reference-bank matches

Reference match fields:

- `rank`
- `reference_id`
- optional `cod_id`, `formula`, `space_group` if a licensed mapping database is
  available
- `similarity`
- `source_database`

## Checkpoint, Configuration, and License Inventory

### Upstream Path Configuration

The official repository centralizes runtime paths in `configs/paths.sh`.
Relevant paths found during review:

| Kind | Upstream path variable | Default path |
| --- | --- | --- |
| MAE pretrain checkpoint | `PATH_CKPT_PRETRAIN` | `checkpoints/pretrain/checkpoint_latest.pt` |
| XDecomposer checkpoint | `PATH_CKPT_XDECOMPOSER` | `checkpoints/xdecomposer/latest.pt` |
| XDecomposer best checkpoint | `PATH_CKPT_XDECOMPOSER_BEST` | `checkpoints/xdecomposer/best.pt` |
| Single-phase MP20 data | `PATH_DATA_SINGLEPHASE` | `mp20-xrd_data/data` |
| MP20 crystal DB | `PATH_DATA_CRYSTAL_DB` | `data/UniqCryLabeled.db` |
| RRUFF DB | `PATH_DATA_RRUFF` | `data/UniqRruffCrystal.db` |
| Evaluation output | `PATH_OUTPUT_TEST` | `test_results` |

The GitHub repository advertises a `checkpoints` release, but this precheck did
not download it. Checkpoint filenames, sizes, hashes, and redistribution terms
are therefore unverified.

### Required Asset Manifest

Before any real integration, add a manifest outside committed weights, for
example `models/xdecomposer/manifest.yaml` in a mounted model directory:

```yaml
model_id: xdecomposer-mp20-v1
upstream_repo: https://github.com/Licht0812/XDecomposer
upstream_commit: 48c4efe3681256cd16d3b8d2f56d665cf58fd129
source_license: MIT
checkpoint_license: UNKNOWN
dataset_license: UNKNOWN
python: "3.10"
torch: "2.10.0"
cuda: "12.8"
xrd_length: 3500
mae_checkpoint:
  path: checkpoints/pretrain/checkpoint_latest.pt
  sha256: UNKNOWN
separator_checkpoint:
  path: checkpoints/xdecomposer/latest.pt
  sha256: UNKNOWN
reference_bank:
  path: data/reference_bank.pt
  sha256: UNKNOWN
```

### License Notes

The upstream source license is MIT. That covers source-code use subject to the
license notice. It does not automatically establish redistribution rights for
checkpoint files, MP20-derived data, RRUFF-derived data, or any reference-bank
database. Those assets must be separately reviewed before packaging, mirroring,
or mounting in shared infrastructure.

## Dependency Conflict Analysis

### Python

Current CPICANN-XRD-App:

- `requires-python = ">=3.11,<3.14"`
- Ruff and mypy configured for Python 3.11

XDecomposer official installation:

- conda environment named `xdecomposer`
- Python `3.10`

Conflict: same-process integration would need either a Python 3.10 downgrade for
CPICANN-XRD-App or unverified XDecomposer execution on Python 3.11+. Neither
should be assumed without tests.

### PyTorch and CUDA

Current CPICANN-XRD-App:

- `torch>=2.5,<3`
- `torch` resolves from the explicit `pytorch-cpu` index
- Docker/runtime path is designed around CPU inference and mounted model files

XDecomposer `requirements.txt`:

- `torch==2.10.0` appears twice
- `torchvision==0.25.0`
- `torchaudio==2.10.0`
- `triton==3.6.0`
- CUDA 12.8 NVIDIA wheels including cuBLAS, cuDNN, cuFFT, cuSOLVER, cuSPARSE,
  NCCL, and runtime packages

Conflict: installing XDecomposer dependencies into the CPICANN environment would
replace or constrain the CPU PyTorch setup and significantly enlarge the runtime.
GPU availability, driver compatibility, and CUDA 12.8 support must be validated
on the deployment hosts.

### Scientific Python Stack

Current CPICANN-XRD-App:

- `numpy>=1.26,<3`
- `pymatgen>=2024.10,<2027`
- SciPy is pulled transitively

XDecomposer pins:

- `numpy==2.0.1`
- `scipy==1.14.0`
- `pymatgen==2025.10.7`
- `ase==3.22.1`
- `pyxtal==0.6.7`
- `pyts==0.13.0`
- `skimage==0.0`

Conflict: the broad current constraints may be compatible in principle, but
XDecomposer exact pins should not be merged into the current lockfile without a
full resolution pass. The `skimage==0.0` package is a placeholder and should be
reviewed before use; the intended package is usually `scikit-image`.

### Extra ML Dependencies

XDecomposer additionally requires:

- `tensorflow==2.21.0`
- `torch-geometric==2.6.1`
- `swanlab==0.3.16`

Conflict: TensorFlow plus CUDA PyTorch in the same web/API process increases
binary dependency risk, memory footprint, and container image size. These
dependencies are a strong argument for isolation.

## Independent Container vs Same-Process Integration

| Option | Advantages | Risks |
| --- | --- | --- |
| Independent container/service | Isolates Python 3.10, CUDA 12.8, TensorFlow, PyG, and torch pins; keeps CPICANN default backend untouched; allows GPU scheduling and model mounts; limits checkpoint loading blast radius | More deployment moving parts; needs service health checks, request timeouts, artifact transfer, and reference-bank synchronization |
| Same-process Python integration | Lower local call overhead; one API process; simpler artifact sharing once stable | Python version conflict; PyTorch CPU/GPU conflict; CUDA libraries enter web/API image; upstream is not packaged as a stable library; unsafe checkpoint loading has wider impact; failures can crash CPICANN service |

Recommendation: implement XD-1 through XD-3 as an isolated container or worker
with a narrow JSON/file contract. Reconsider same-process integration only after
the adapter, assets, and real checkpoint validation are stable.

## Proposed Schemas, Interfaces, and Directory Structure

No code is added in this phase. Proposed future additions:

```text
configs/
  models/
    xdecomposer.example.yaml
docker/
  Dockerfile.xdecomposer
compose.xdecomposer.yaml
docs/
  xdecomposer_feasibility.md
models/
  xdecomposer/
    README.md
    manifest.example.yaml
src/cpicann_xrd/
  xdecomposer/
    __init__.py
    schemas.py
    preprocessing.py
    client.py
    contracts.py
tests/
  xdecomposer/
    test_schemas.py
    test_preprocessing_contract.py
    test_client_stub.py
```

Proposed interface boundary:

- `XDecomposerAssetManifest`: validates model/config/checkpoint/reference-bank
  identity without loading weights
- `XDecomposerRequest`: normalized product request for decomposition
- `XDecomposerResult`: stable product response, independent of upstream tensor
  objects
- `XDecomposerClient`: HTTP or subprocess client for isolated backend
- `XDecomposerPreprocessor`: explicit 2-theta interpolation and intensity
  normalization to the model axis
- `XDecomposerBackendProtocol`: separate protocol from current CPICANN
  classification backend

The existing CPICANN backend and `PredictionService` should not be widened to
pretend decomposition is ordinary single-phase classification. A separate
service boundary avoids confusing rank/probability semantics.

## Security and Operational Risks

- Upstream evaluation code uses `torch.load` for checkpoints/reference caches.
  Any product integration must only load authorized, hash-verified local files.
- Component order is not stable without downstream alignment or deterministic
  ranking rules.
- Reference-bank IDs are dataset-local IDs. Mapping them to COD IDs, formulas,
  or space groups requires a licensed, versioned database.
- CPICANN input uses 4500 points and scales intensities around 100 before the
  current model. XDecomposer reviewed paths use 3500 points and max-normalized
  vectors. Shared preprocessing would be incorrect.
- Real reported weights/proportions require scientific validation. Intensity
  mass estimates should be labelled approximate until checked.
- Upstream scripts are training/evaluation oriented, not a production inference
  API.

## XD-0 to XD-5 Acceptance Standards

### XD-0: Feasibility Precheck

Acceptance:

- `docs/xdecomposer_feasibility.md` exists.
- The document records source links, input/output contract, asset inventory,
  dependency conflicts, integration options, proposed schemas, and XD-0 to XD-5
  criteria.
- No CPICANN inference, CLI, Web, API, or report code is changed.
- No XDecomposer weights or datasets are committed.

### XD-1: Asset and License Gate

Acceptance:

- A product asset manifest schema exists with required SHA-256 fields.
- Local model directories contain only placeholders/examples in git.
- Real checkpoint, dataset, and reference-bank licenses are documented before
  use.
- Manifest verification fails closed on missing or mismatched hashes.
- No automatic download runs by default.

### XD-2: Isolated Runtime Spike

Acceptance:

- `Dockerfile.xdecomposer` or equivalent environment builds independently from
  the CPICANN CPU image.
- The runtime imports the reviewed XDecomposer model modules and validates
  dependency versions.
- The container does not bake in real weights.
- Health check reports environment, CUDA availability, and asset presence.
- CPICANN default app behavior is unchanged when this container is absent.

### XD-3: Single-Sample Adapter

Acceptance:

- An isolated adapter accepts one product request and returns the proposed JSON
  response shape.
- Preprocessing from raw two-theta/intensity to `[1, 1, 3500]` is deterministic
  and covered by tests.
- With fixtures or a stub backend, outputs are stable and do not require real
  checkpoints.
- With locally authorized checkpoints, smoke testing is allowed but must be
  reported separately with exact commands and hashes.

### XD-4: Opt-In Product Integration

Acceptance:

- Web/API/CLI exposure is behind an explicit opt-in flag or separate endpoint.
- CPICANN remains the default backend.
- Decomposition outputs are stored as separate artifacts and are not mixed with
  single-phase CPICANN ranking semantics.
- Timeout, backend unavailable, and asset validation errors produce structured
  diagnostics.
- Tests cover the opt-in integration with a stub XDecomposer service.

### XD-5: Real Checkpoint Validation

Acceptance:

- Authorized checkpoints, MAE pretrain weights, and reference data are present
  only through local mounts or approved artifact storage.
- All asset SHA-256 values are recorded in the manifest.
- A real smoke test runs on representative multiphase samples and records exact
  commands, hardware, device, dependency versions, and output artifacts.
- CPU and GPU behavior are documented, including expected performance and any
  unsupported modes.
- Scientific review confirms whether component activity, reference matches, and
  estimated weights are acceptable for user-facing display.
