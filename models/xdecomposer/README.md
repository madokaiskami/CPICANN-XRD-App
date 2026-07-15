# XDecomposer Assets

This directory is reserved for local XDecomposer asset manifests and operator
notes. Real checkpoints, datasets, reference banks, and generated caches must
not be committed to Git or baked into public Docker images.

Expected runtime layout for an authorized local deployment:

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
Before production use, a human must confirm checkpoint, dataset, and reference
bank licenses and replace every placeholder path/hash with the locally verified
asset values.
