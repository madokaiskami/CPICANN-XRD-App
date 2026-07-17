# XDecomposer Assets

This directory is reserved for local XDecomposer asset manifests and operator
notes. Real checkpoints, datasets, reference banks, and generated caches must
not be committed to Git or baked into public Docker images.

Expected runtime layout for authorized local decomposition-only smoke tests:

```text
models/xdecomposer/
├── manifest.yaml
├── checkpoints/
│   ├── pretrain/
│   │   └── checkpoint_latest.pt
│   └── xdecomposer/
│       └── latest.pt
```

Use `reference_bank_required: false` in `manifest.yaml` for this mode. The
worker can run decomposition without reference-bank matching.

Expected runtime layout for reference matching:

```text
models/xdecomposer/
├── manifest.yaml
├── checkpoints/
│   ├── pretrain/
│   │   └── checkpoint_latest.pt
│   └── xdecomposer/
│       └── latest.pt
└── reference_bank.pt
```

Only `README.md` and `manifest.example.yaml` are intended to be tracked. The
example manifest is parseable documentation, not a usable production manifest.
Before production use, a human must confirm checkpoint licenses. Dataset and
reference-bank licenses are also required when `reference_bank_required` is true
or a `reference_bank` entry is present. Replace every placeholder path/hash with
the locally verified asset values.
