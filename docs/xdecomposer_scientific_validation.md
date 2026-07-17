# XDecomposer Scientific Validation Draft

Status: experimental

This document defines a reproducible validation framework for the
CPICANN-XRD-App XDecomposer workflow. It does not claim that the workflow is
scientifically validated. Scientific conclusions require material/XRD domain
review and signed acceptance of the data boundary.

## Scope

The validation framework records raw metrics, plots and failed cases for
decomposition and component identification. It must not delete outliers or
failed samples to improve aggregate metrics.

The current Codex-deliverable framework covers:

- permutation-invariant component matching;
- active source count accuracy;
- reconstruction RMSE;
- component-target Pearson correlation;
- spectral-angle/cosine similarity;
- estimated weight MAE;
- false active slot count;
- missed minor phase count;
- optional component Top-1 accuracy and Top-5 recall fields;
- raw metrics CSV;
- failed cases CSV;
- summary Markdown;
- metric plots.

## Dataset Blueprint

Minimum case families:

| Case family | Human ground truth required |
| --- | --- |
| synthetic_two_phase | no |
| synthetic_three_phase | no |
| ratio_90_10 | no |
| ratio_70_30 | no |
| ratio_50_50 | no |
| ratio_30_70 | no |
| minor_impurity_phase | no |
| noise_sweep | no |
| background_shift | no |
| peak_position_shift | no |
| peak_width_variation | no |
| experimental_multiphase | yes |
| out_of_catalog_component | yes |
| correct_element_constraints | no |
| incorrect_element_constraints | no |
| single_phase_in_multiphase_mode | no |
| sources_exceed_checkpoint_num_sources | yes |
| amorphous_background | yes |

Synthetic spectra are useful for reproducibility and regression testing, but
they do not replace experimental multiphase samples.

## Metrics

Decomposition metrics:

- active source count accuracy;
- reconstruction RMSE;
- component-target correlation;
- permutation-invariant component matching;
- spectral angle/cosine similarity;
- estimated weight MAE;
- false active slot rate;
- missed minor phase rate.

Identification metrics:

- component Top-1 accuracy;
- component Top-5 recall;
- fraction of true phases entering the candidate set;
- recall before and after element filtering;
- high-confidence error rate for out-of-catalog samples;
- relationship between decomposition quality and CPICANN conditional
  confidence.

Performance metrics to collect in protected environments:

- model cold-start;
- single-sample latency;
- 10-sample batch latency;
- GPU memory;
- CPU memory;
- concurrent throughput;
- result artifact size.

## Artifact Contract

The validation helper writes:

- `raw_metrics.csv`;
- `failed_cases.csv`;
- `scientific_validation_summary.md`;
- `reconstruction_rmse.png`;
- `component_correlation.png`.

`failed_cases.csv` is mandatory even when empty. The summary must keep
`Status: experimental` until scientific review is complete.

## Human Review Gate

Domain reviewers must decide:

- whether sample truth labels are reliable;
- whether synthetic mixing is physically meaningful;
- whether intensity ratios can stand in for phase ratios;
- acceptable thresholds for minor phase detection;
- acceptable behavior for unknown and out-of-catalog phases;
- whether estimated weights may be shown to users;
- which input conditions require explicit warnings;
- whether deployment is public, internal-only or research-only.

Before review is signed:

```text
status = experimental
```

After review, and only for a defined data boundary:

```text
status = validated-for-defined-domain
```
