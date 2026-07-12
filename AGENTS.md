# AGENTS.md

## Project mission
Build a reproducible, offline-capable XRD phase-identification application around the pretrained CPICANN single-phase model.

## Source of truth
- Read DEVELOPMENT_PLAN.md before starting work.
- Execute only the phase explicitly requested by the user.
- Do not start a later phase until current acceptance commands pass.
- Business rules in DEVELOPMENT_PLAN.md are immutable unless the user approves a change.

## Architecture constraints
- Keep inference core independent from CLI, FastAPI, and Streamlit.
- CLI, API, and Web must call the same service layer.
- Use an InferenceBackend protocol with FakeBackend and CPICANNBackend.
- Never duplicate softmax/filtering logic across interfaces.
- Never treat PNG images as spectra.
- Never interpret filtered confidence as phase fraction.

## Model and data constraints
- Do not commit model weights, secrets, raw licensed datasets, or generated run directories.
- Pin model source revision and verify SHA-256.
- Keep model weights and class catalog version-locked.
- Ordinary CI must run without real pretrained weights.
- Real-model tests must be marked and skipped unless the model is configured.

## Quality gates
Before claiming completion, run the commands required by the current phase, plus:
- git diff --check
- uv run ruff check .
- uv run pytest -q for the available test suite

## Coding conventions
- Python 3.11.
- Type annotations on public functions.
- Pydantic models for external schemas and configuration.
- pathlib instead of string path manipulation.
- Structured exceptions with stable error codes.
- No bare except.
- No silent data correction: every correction or ignored input must be diagnosable.
- User-facing output and reports are Chinese; code identifiers and docstrings may be English.

## Change discipline
- Keep changes limited to the requested phase.
- Do not reformat unrelated files.
- Do not commit or push unless explicitly asked.
- Summarize files changed, commands run, results, and remaining risks.
