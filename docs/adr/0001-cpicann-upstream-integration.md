# ADR 0001: CPICANN Upstream Integration

## Status

Accepted for Phase 1. Implementation is deferred to Phase 4.

## Context

Phase 1 audited these upstream artifacts:

| Artifact | URL | Revision / version | Evidence |
|---|---|---:|---|
| GitHub project | `https://github.com/WPEM/CPICANN` | `01e915697f544e5a4fcd0f03f76da0e38ee00ece` | README and MIT `LICENSE` from a local upstream checkout |
| User fork | `git@github.com:madokaiskami/CPICANN.git` | `01e915697f544e5a4fcd0f03f76da0e38ee00ece` on branch `packaging` | local checkout remote metadata |
| HF source mirror | `https://huggingface.co/AI4Cryst/CPICANN` | `3dbfaeab51d272e013d211c7f957760b46ab41cc` | `src/model/CPICANN.py`, `src/data_format.py`, `src/annotation/anno_struc.csv` |
| HF pretrained models | `https://huggingface.co/caobin/pretrainCPICANN` | `85b7e2ce9060286c84efc459cfde875ad5ca30ec` | Git LFS pointers and README |
| PyPI wrapper | `WPEMPhase==0.1.1` | wheel SHA-256 `ce15cfb5b783aa5fe8c2788c02d387d98d1d97fe39dafd26a1189cfb4f48a182` | unpacked wheel only, not installed |

The GitHub README points source code to the HF source mirror and pretrained models to the HF pretrained repository. The local user fork and upstream GitHub repository contain project metadata, not the full model source tree.

The HF source mirror is not a Python package: it contains training, validation, model, dataset and data-format scripts under `src/`, with relative imports and training-oriented side effects. The PyPI `WPEMPhase` wrapper exposes a `PhaseIdentifier` function, but that function downloads a Figshare `SystemFiles.zip` on first run, writes result CSV/figures in the current directory, loads pickle checkpoints directly with `torch.load`, and includes filtering semantics that do not match this product plan's `include_must` / `allowed_elements` rules.

## Decision

Do not call `WPEMPhase.PhaseIdentifier` as the product inference implementation.

Use an isolated CPICANN backend in Phase 4. The backend should be based on the audited CPICANN single-phase network definition from HF source commit `3dbfaeab51d272e013d211c7f957760b46ab41cc`, paired with the version-locked model and catalog manifests. Because the audited HF source is not packageable as-is, Phase 4 may copy only the minimal network definition into the product adapter if a clean Git dependency or submodule cannot be established by then.

If code is copied, preserve upstream MIT license attribution, original path `src/model/CPICANN.py`, source commit, and a short list of product-local changes. Do not copy training scripts, datasets, notebooks, or unrelated model variants into the product core.

Use the HF pretrained repository file `CPICANNsingle_phase_D1.pth` as the Phase 4 candidate for model ID `cpicann-single-d1`, subject to authorization and checkpoint verification.

## Consequences

- Ordinary CI remains independent of real weights.
- Product code can implement the plan's stricter diagnostics, element filtering, masked softmax and metadata rules.
- Phase 4 must still verify actual checkpoint top-level keys and state-dict compatibility after authorized weight access.
- Phase 5 must build a catalog manifest from `anno_struc.csv`/`strucs.csv` evidence and manually verify sampled `class_index -> COD ID` mappings.

## Open Items

- `CPICANNsingle_phase_D1.pth` actual checkpoint contents were not downloaded in Phase 1. Git LFS pointer metadata was inspected only.
- Permission to redistribute real weights is UNKNOWN. The product must not publish weights in Git, public wheels, images or releases.
- The exact status of the Figshare bundle used by `WPEMPhase` is not part of the selected integration path and was not downloaded.
