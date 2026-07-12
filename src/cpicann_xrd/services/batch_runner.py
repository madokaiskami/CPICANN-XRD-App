"""Batch prediction orchestration and standard output generation."""

from __future__ import annotations

import platform
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from cpicann_xrd.catalog.catalog import PhaseCatalog
from cpicann_xrd.core.preprocessing import PreprocessedSpectrum, preprocess_spectrum
from cpicann_xrd.core.spectrum_io import read_spectrum_file
from cpicann_xrd.exceptions import CpicannXrdError, ErrorCode
from cpicann_xrd.model.protocol import InferenceBackend
from cpicann_xrd.reports.exporters import (
    prediction_rows,
    write_csv_atomic,
    write_json_atomic,
    write_observed_png,
    write_preprocessed_csv,
    write_text_atomic,
    write_zip_bundle,
)
from cpicann_xrd.reports.markdown_report import render_summary_report
from cpicann_xrd.schemas import (
    DiagnosticRecord,
    FilterSpec,
    ModelInfo,
    PreprocessingConfig,
    RunMetadata,
    SamplePrediction,
)
from cpicann_xrd.services.predictor import PredictionService
from cpicann_xrd.services.run_context import RunContext
from cpicann_xrd.version import __version__

PREDICTION_FIELDNAMES = [
    "sample_id",
    "filtered_rank",
    "global_rank",
    "class_index",
    "cod_id",
    "formula",
    "reduced_formula",
    "elements",
    "space_group",
    "space_group_number",
    "raw_logit",
    "unfiltered_probability",
    "filtered_confidence",
    "candidate_count_after_filter",
]

SUMMARY_FIELDNAMES = [
    "sample_id",
    "source_filename",
    "status",
    "candidate_count",
    "returned_top_k",
    "top1_cod_id",
    "top1_formula",
    "top1_confidence",
    "top2_cod_id",
    "top2_formula",
    "top2_confidence",
    "top3_cod_id",
    "top3_formula",
    "top3_confidence",
    "top4_cod_id",
    "top4_formula",
    "top4_confidence",
    "top5_cod_id",
    "top5_formula",
    "top5_confidence",
    "warning",
]

DIAGNOSTIC_FIELDNAMES = [
    "source_filename",
    "status",
    "stage",
    "error_code",
    "message",
    "rows_read",
    "rows_invalid",
    "angle_min",
    "angle_max",
    "ignored_reason",
]

INPUT_MANIFEST_FIELDNAMES = [
    "source_filename",
    "status",
    "sha256",
    "sample_id",
]


@dataclass(frozen=True)
class BatchRunResult:
    """Result metadata for one batch run."""

    run_id: str
    run_dir: Path
    counts: dict[str, int]
    predictions: list[SamplePrediction]
    diagnostics: list[DiagnosticRecord]


def run_batch(
    *,
    input_paths: list[Path],
    output_root: Path,
    backend: InferenceBackend,
    catalog: PhaseCatalog | None = None,
    filter_spec: FilterSpec | None = None,
    top_k: int = 5,
    run_id: str | None = None,
) -> BatchRunResult:
    """Run batch inference and write the standard Phase 6 output bundle."""
    active_filter_spec = filter_spec or FilterSpec()
    context = RunContext.create(output_root, run_id=run_id)
    service = PredictionService(backend, catalog=catalog)
    diagnostics: list[DiagnosticRecord] = []
    predictions: list[SamplePrediction] = []
    summary_rows: list[dict[str, object]] = []
    input_rows: list[dict[str, object]] = []

    for input_path in _discover_inputs(input_paths):
        source_filename = input_path.name
        sample_id = context.allocate_sample_id(source_filename)
        read_result = read_spectrum_file(input_path, sample_id=sample_id)
        diagnostics.extend(read_result.diagnostics)
        input_rows.append(
            {
                "source_filename": source_filename,
                "status": read_result.status,
                "sha256": read_result.spectrum.sha256 if read_result.spectrum else "",
                "sample_id": sample_id if read_result.status == "success" else "",
            }
        )
        if read_result.status != "success" or read_result.spectrum is None:
            summary_rows.append(
                _non_success_summary_row(sample_id, source_filename, read_result.status)
            )
            continue

        try:
            preprocessed = preprocess_spectrum(read_result.spectrum)
            prediction = service.predict_tensor(
                sample_id=sample_id,
                source_filename=source_filename,
                tensor=torch.from_numpy(preprocessed.model_input),
                top_k=top_k,
                filter_spec=active_filter_spec,
            )
            sample_dir = context.sample_dir(sample_id)
            _write_sample_outputs(sample_dir, prediction, read_result.spectrum, preprocessed)
        except CpicannXrdError as exc:
            diagnostics.append(_error_diagnostic(source_filename, "prediction", exc))
            summary_rows.append(_non_success_summary_row(sample_id, source_filename, "failed"))
            continue
        except Exception as exc:
            diagnostics.append(
                DiagnosticRecord(
                    source_filename=source_filename,
                    status="failed",
                    stage="batch",
                    error_code=ErrorCode.INFERENCE_FAILED,
                    message=str(exc),
                )
            )
            summary_rows.append(_non_success_summary_row(sample_id, source_filename, "failed"))
            continue

        predictions.append(prediction)
        summary_rows.append(_success_summary_row(prediction))

    counts_counter = Counter(str(row["status"]) for row in summary_rows)
    counts = {
        "success": counts_counter.get("success", 0),
        "failed": counts_counter.get("failed", 0),
        "ignored": counts_counter.get("ignored", 0),
    }
    model_info = backend.model_info
    write_csv_atomic(context.run_dir / "summary.csv", SUMMARY_FIELDNAMES, summary_rows)
    write_csv_atomic(
        context.run_dir / "diagnostics.csv",
        DIAGNOSTIC_FIELDNAMES,
        [_diagnostic_row(diagnostic) for diagnostic in diagnostics],
    )
    write_csv_atomic(context.run_dir / "input_manifest.csv", INPUT_MANIFEST_FIELDNAMES, input_rows)
    metadata = _run_metadata(
        context=context,
        model_info=model_info,
        filter_spec=active_filter_spec,
        top_k=top_k,
        input_rows=input_rows,
        counts=counts,
    )
    write_json_atomic(context.run_dir / "run_metadata.json", metadata.model_dump(mode="json"))
    report = render_summary_report(
        run_id=context.run_id,
        model_info=model_info,
        preprocessing_version=model_info.preprocessing_version,
        filter_spec=active_filter_spec,
        top_k=top_k,
        predictions=predictions,
        diagnostics=diagnostics,
        counts=counts,
    )
    write_text_atomic(context.run_dir / "summary_report.md", report)
    write_zip_bundle(context.run_dir, context.run_dir / "result_bundle.zip")
    return BatchRunResult(
        run_id=context.run_id,
        run_dir=context.run_dir,
        counts=counts,
        predictions=predictions,
        diagnostics=diagnostics,
    )


def _discover_inputs(input_paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in input_paths:
        if path.is_dir():
            files.extend(
                child
                for child in sorted(path.iterdir())
                if child.is_file() and not _is_dotfile(child)
            )
        elif path.is_file() and not _is_dotfile(path):
            files.append(path)
    return sorted(files, key=lambda item: item.name)


def _is_dotfile(path: Path) -> bool:
    return path.name.startswith(".")


def _write_sample_outputs(
    sample_dir: Path,
    prediction: SamplePrediction,
    spectrum: Any,
    preprocessed: PreprocessedSpectrum,
) -> None:
    write_csv_atomic(
        sample_dir / "prediction_top5.csv",
        PREDICTION_FIELDNAMES,
        prediction_rows(prediction),
    )
    write_observed_png(sample_dir / "observed_xrd.png", spectrum)
    write_preprocessed_csv(sample_dir / "preprocessed_xrd.csv", preprocessed)


def _success_summary_row(prediction: SamplePrediction) -> dict[str, object]:
    row: dict[str, object] = {
        "sample_id": prediction.sample_id,
        "source_filename": prediction.source_filename,
        "status": prediction.status,
        "candidate_count": prediction.candidate_count_after_filter,
        "returned_top_k": prediction.returned_top_k,
        "warning": "；".join(prediction.warnings),
    }
    for rank in range(1, 6):
        if rank <= len(prediction.predictions):
            item = prediction.predictions[rank - 1]
            row[f"top{rank}_cod_id"] = item.phase.cod_id
            row[f"top{rank}_formula"] = item.phase.formula
            row[f"top{rank}_confidence"] = item.filtered_confidence
        else:
            row[f"top{rank}_cod_id"] = ""
            row[f"top{rank}_formula"] = ""
            row[f"top{rank}_confidence"] = ""
    return row


def _non_success_summary_row(
    sample_id: str,
    source_filename: str,
    status: str,
) -> dict[str, object]:
    row: dict[str, object] = {
        "sample_id": sample_id,
        "source_filename": source_filename,
        "status": status,
        "candidate_count": 0,
        "returned_top_k": 0,
        "warning": "",
    }
    for rank in range(1, 6):
        row[f"top{rank}_cod_id"] = ""
        row[f"top{rank}_formula"] = ""
        row[f"top{rank}_confidence"] = ""
    return row


def _diagnostic_row(diagnostic: DiagnosticRecord) -> dict[str, object]:
    return {
        "source_filename": diagnostic.source_filename,
        "status": diagnostic.status,
        "stage": diagnostic.stage,
        "error_code": diagnostic.error_code.value if diagnostic.error_code else "",
        "message": diagnostic.message,
        "rows_read": diagnostic.rows_read or "",
        "rows_invalid": diagnostic.rows_invalid or "",
        "angle_min": diagnostic.angle_min or "",
        "angle_max": diagnostic.angle_max or "",
        "ignored_reason": diagnostic.ignored_reason or "",
    }


def _error_diagnostic(
    source_filename: str,
    stage: str,
    exc: CpicannXrdError,
) -> DiagnosticRecord:
    return DiagnosticRecord(
        source_filename=source_filename,
        status="failed",
        stage=stage,
        error_code=exc.error_code,
        message=exc.message,
    )


def _run_metadata(
    *,
    context: RunContext,
    model_info: ModelInfo,
    filter_spec: FilterSpec,
    top_k: int,
    input_rows: list[dict[str, object]],
    counts: dict[str, int],
) -> RunMetadata:
    return RunMetadata(
        run_id=context.run_id,
        application={
            "name": "cpicann-xrd",
            "version": __version__,
            "git_commit": "UNKNOWN",
        },
        model=model_info,
        runtime={
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "success_count": str(counts["success"]),
            "failed_count": str(counts["failed"]),
            "ignored_count": str(counts["ignored"]),
        },
        preprocessing=PreprocessingConfig(name=model_info.preprocessing_version),
        filter=filter_spec,
        top_k=top_k,
        inputs=[
            {
                "source_filename": str(row["source_filename"]),
                "status": str(row["status"]),
                "sha256": str(row["sha256"]),
            }
            for row in input_rows
        ],
    )
