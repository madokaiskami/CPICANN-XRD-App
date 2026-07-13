# Known Limitations

- CPICANN-XRD-App reports model ranking confidence, not quantitative phase
  fraction, composition, or refinement output.
- The real backend currently supports the audited CPICANN single-phase D1 model
  contract only.
- Real model inference requires local authorized weights under `models/`; weights
  are not committed, bundled in releases, or baked into Docker images.
- The committed catalog is tied to the configured model SHA-256 and class order.
  Updating either the model weights or catalog requires regenerating golden
  baselines and manual review.
- Web and API execution are synchronous and intended for small local batches.
  Queueing, authentication, database persistence, and distributed workers are out
  of scope for `v0.1.0-rc1`.
- Current golden samples are regression baselines for interface and model
  contract stability. They are not a broad scientific validation set.
- Space-group metadata comes from the catalog. Missing or incorrect upstream
  catalog fields will propagate to reports.
