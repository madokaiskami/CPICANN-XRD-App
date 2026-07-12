# CPICANN Legacy Preprocessing Contract

## Scope

This document captures the upstream preprocessing behavior that Phase 3 must reproduce under protocol name `legacy-cpicann-v1`, unless a later ADR explicitly changes the default.

Evidence comes from:

- HF source `src/data_format.py` at commit `3dbfaeab51d272e013d211c7f957760b46ab41cc`;
- PyPI wrapper `WPEMPhase==0.1.1`, whose `WPEMPhase/data_format.py` matches the same logic;
- PyPI wrapper `WPEMPhase/CPICANN.py` inference code.

## Supported Input Extensions

Legacy code accepts exactly these lowercase suffix strings:

```text
txt
csv
xy
```

The legacy implementation is case-sensitive. Product code must remain case-insensitive per `DEVELOPMENT_PLAN.md`, but should record this as a product hardening over legacy behavior.

Unsupported files return `None` from `convert_file`. The legacy warning is not emitted correctly because it constructs `Warning(...)` without raising or logging it. Product code must produce a diagnostic record instead.

## TXT and XY Behavior

`txt` and `xy` share the same parser:

1. read all lines as text;
2. strip newline;
3. replace tabs with spaces;
4. split on spaces and remove empty tokens;
5. if a row has exactly 3 tokens, use `[token0, float(token1) - float(token2)]`;
6. if a row has fewer than 2 or more than 3 tokens, skip it;
7. if a row has exactly 2 tokens, append it without numeric conversion at parse time;
8. pass collected rows to interpolation.

Important edge cases:

- two-column non-numeric header rows are not skipped during parsing and later fail when converting the whole array to `float32`;
- three-column non-numeric rows are skipped only if subtraction raises `ValueError`;
- rows are not sorted;
- duplicate angles are not handled;
- non-finite values are not checked explicitly.

## CSV Behavior

Legacy CSV parsing uses:

```python
pd.read_csv(file_path).values
```

Consequences:

- the first row is treated as a header by pandas;
- CSV files without a header lose their first data row;
- the parser does not validate column count before interpolation;
- all columns are passed through to interpolation, but interpolation uses only column `0` as angle and column `1` as intensity.

Phase 3 product code must document and test any compatibility hardening relative to this behavior.

## Interpolation Behavior

Legacy interpolation:

```python
rows.insert(0, ["10", rows[0][1]]) if float(rows[0][0]) > 10 else None
rows.append(["80", rows[-1][1]]) if float(rows[-1][0]) < 80 else None
rowsData = np.array(rows, dtype=np.float32)
f = interpolate.interp1d(rowsData[:, 0], rowsData[:, 1], kind="slinear")
xnew = np.linspace(10, 80, 4500)
ynew = f(xnew)
```

Contract:

| Item | Value |
|---|---|
| output angle range | `10.0` to `80.0` degrees |
| output point count | `4500` |
| interpolation kind | SciPy `interp1d(..., kind="slinear")` |
| left boundary behavior | if first raw angle is greater than `10`, prepend angle `10` with the first intensity |
| right boundary behavior | if last raw angle is less than `80`, append angle `80` with the last intensity |
| output passed to model | intensity vector only |
| tensor conversion | `torch.tensor(v, dtype=torch.float32).reshape(1, 1, -1)` |
| pre-model scaling | `v / v.max() * 100` in packaged inference |
| model-internal scaling | `x / 100` inside `forward` |

The net packaged inference input to the convolution is therefore approximately intensity normalized to max `1.0`, assuming the maximum intensity is positive and finite.

## Known Missing Fixtures

Phase 1 did not find these files in the local CPICANN checkout, HF source mirror, HF pretrained repository, or product repository:

```text
0-norm.txt
1-norm.txt
3-norm.txt
```

Their exact row counts, angle ranges and output hashes are UNKNOWN. Phase 3 must add these fixtures before creating golden preprocessing assertions.

## Product Requirements Derived From Legacy

For `legacy-cpicann-v1`, product code should:

- accept `.txt`, `.csv`, `.xy` case-insensitively per product rules;
- parse and interpolate to exactly `4500` points across `10..80`;
- return a `float32` tensor with shape `(batch, 1, 4500)`;
- record every hardening decision as a diagnostic or warning;
- never pass `.png`, `.rar` or other unsupported files to the model;
- fail clearly instead of silently returning `None` where legacy code would drop or crash.

Phase 3 must decide, with tests, whether compatibility mode should reproduce legacy CSV header loss exactly or preserve no-header CSV data while recording that this differs from upstream legacy behavior.
