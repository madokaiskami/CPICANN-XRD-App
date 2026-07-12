# CPICANN Single-Phase Model Contract

## Scope

This contract covers the CPICANN single-phase classifier targeted by product model ID `cpicann-single-d1`. It is based on read-only Phase 1 audit evidence and must be revalidated in Phase 4 with authorized weights.

## Locked Upstream Evidence

| Item | Value |
|---|---|
| GitHub project | `https://github.com/WPEM/CPICANN` |
| GitHub commit | `01e915697f544e5a4fcd0f03f76da0e38ee00ece` |
| User fork | `git@github.com:madokaiskami/CPICANN.git` |
| User fork branch / commit | `packaging` / `01e915697f544e5a4fcd0f03f76da0e38ee00ece` |
| HF source repository | `https://huggingface.co/AI4Cryst/CPICANN` |
| HF source commit | `3dbfaeab51d272e013d211c7f957760b46ab41cc` |
| HF pretrained repository | `https://huggingface.co/caobin/pretrainCPICANN` |
| HF pretrained commit | `85b7e2ce9060286c84efc459cfde875ad5ca30ec` |
| PyPI wrapper audited | `WPEMPhase==0.1.1`, wheel SHA-256 `ce15cfb5b783aa5fe8c2788c02d387d98d1d97fe39dafd26a1189cfb4f48a182` |
| License evidence | GitHub project has MIT `LICENSE`; PyPI wrapper metadata says MIT; HF source/pretrain repos do not provide stronger redistribution evidence in Phase 1 |

## Network Definition

The single-phase network definition is `src/model/CPICANN.py` in the HF source mirror.

Confirmed constructor for training and validation:

```python
CPICANN(embed_dim=128, num_classes=23073)
```

Default architecture values from source:

| Parameter | Value |
|---|---:|
| `embed_dim` | `128` for the audited single-phase scripts |
| `nhead` | `8` |
| `num_encoder_layers` | `6` |
| `dim_feedforward` | `1024` |
| `dropout` | `0.1` |
| `activation` | `relu` |
| output classes | `23073` |

The model returns raw logits from `cls_head(feats[0])`. Softmax is not part of `CPICANN.forward`.

## Input Tensor Contract

| Question | Contract |
|---|---|
| dtype | `torch.float32` |
| product inference shape | `(batch, 1, 4500)` |
| tolerated training/data-loader shape | `(batch, 2, 4500)` is accepted by `forward`; the first channel is discarded and only channel index `1` is used |
| 2theta range represented by preprocessing | `10.0` to `80.0` degrees |
| fixed point count | `4500` |
| model-internal scaling | `forward` divides input by `100` |
| packaged inference scaling | `WPEMPhase` first normalizes intensity to max `100`, reshapes to `(1, 1, -1)`, then `forward` divides by `100` |

Product preprocessing should therefore produce a finite intensity vector of length `4500`, normalized consistently with the legacy wrapper before tensor conversion.

## Output Contract

| Question | Contract |
|---|---|
| output type | logits tensor |
| output shape | `(batch, 23073)` |
| class index range | `0..23072` |
| single-phase class count | `23073` |
| unfiltered probability | compute `softmax(logits, dim=-1)` only when a probability distribution is needed |
| filtered confidence | must be computed from masked/filtered logits, not by softmaxing already-softmaxed probabilities |

## Checkpoint Contract

| Question | Contract |
|---|---|
| selected candidate file | `CPICANNsingle_phase_D1.pth` from HF pretrained repository |
| LFS oid | `sha256:d2e898bb4b7482cd7b14953feac437b053617815746f025a6b88ca014e51be98` |
| LFS size | `172863787` bytes |
| access requirements | HF pretrain README describes personal verification and approval before download |
| top-level key expected by source | `model` |
| top-level keys actually present in `CPICANNsingle_phase_D1.pth` | UNKNOWN until authorized download and `torch.load` inspection |
| `state_dict` prefix conversion | UNKNOWN until authorized checkpoint inspection; source validation calls `model.load_state_dict(loaded["model"])` directly |
| CPU inference | Source and PyPI wrapper support CPU via `torch.load(..., map_location=torch.device("cpu"))` and no CUDA-only model operation was found |
| redistribution | UNKNOWN; do not redistribute weights |

Other pretrained LFS pointers observed:

| File | LFS SHA-256 | Size bytes |
|---|---|---:|
| `CPICANNsingle_phase_D1.pth` | `d2e898bb4b7482cd7b14953feac437b053617815746f025a6b88ca014e51be98` | `172863787` |
| `CPICANNsingle_phase_D2.pth` | `8141adcc601b59dca06632eb959949937e2e5edb092f8c757ea150f8abc34ff6` | `172900697` |
| `CPICANNsingle_phase_D3.pth` | `1ba8681e858da8200d5241f674c7dcf089881d75cb9539c8c19dbb321e701bfc` | `172900697` |
| `CPICANNsingle_phase_D4.pth` | `1ba8681e858da8200d5241f674c7dcf089881d75cb9539c8c19dbb321e701bfc` | `172900697` |
| `CPICANNbi_phase_D1.pth` | `0e783a2d0e21810240ae7eb7f30446f720887f6226932f94ed3a0597434cec43` | `170927579` |

## Class Catalog Contract

The class-index mapping source is `src/annotation/anno_struc.csv` in the HF source mirror. The PyPI wheel contains `WPEMPhase/config/strucs.csv` with the same visible schema and contiguous class indexes, but `dataId` values are serialized as floats there.

Confirmed `anno_struc.csv` properties:

| Property | Value |
|---|---:|
| data rows | `23073` |
| `No` min | `0` |
| `No` max | `23072` |
| `No` uniqueness | unique |
| `No` continuity | contiguous and equal to row order |

Catalog columns:

```text
dataId, No, formula, symbolSet, spaceGroup, spaceGroupNo, crystalSys,
a, b, c, alpha, beta, gamma
```

Mapping examples:

| class index | COD/data ID | formula | elements | space group | space group No |
|---:|---:|---|---|---|---:|
| `0` | `1522982` | `Mn4 Ni8 Sn4` | `Mn Ni Sn` | `F-43m` | `216` |
| `3378` | `9004484` | `Pb4 S4 O16` | `O Pb S` | `Pnma` | `62` |
| `23072` | `4031642` | `Zr8 V4 Ni12` | `Ni V Zr` | `Fd-3m` | `227` |

Phase 5 must preserve `No` as the authoritative `class_index` and must not assume row order without validating continuity and uniqueness.

## Required Phase 4 Verification

Before implementing real inference, Phase 4 must:

- download weights only with explicit authorization/configuration;
- verify the full file SHA-256 against the LFS oid;
- inspect checkpoint top-level keys;
- verify whether `state_dict` keys need prefix conversion;
- instantiate `CPICANN(embed_dim=128, num_classes=23073)` and validate output shape `(batch, 23073)`;
- confirm deterministic CPU inference on a fixed tensor;
- record final model and catalog hashes in a manifest.
