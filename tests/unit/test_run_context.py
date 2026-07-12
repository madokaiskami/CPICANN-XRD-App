from __future__ import annotations

from datetime import UTC, datetime

from cpicann_xrd.services.run_context import RunContext, create_run_id, safe_sample_id


def test_create_run_id_is_sortable_and_has_random_suffix() -> None:
    run_id = create_run_id(datetime(2026, 7, 12, 9, 8, 7, tzinfo=UTC))

    assert run_id.startswith("20260712T090807Z-")
    assert len(run_id.split("-")[-1]) == 8


def test_safe_sample_id_normalizes_and_deduplicates(tmp_path) -> None:
    context = RunContext.create(tmp_path, run_id="run-001")

    assert safe_sample_id("  0 norm.txt  ") == "0_norm.txt"
    assert safe_sample_id("中文样品") == "sample"
    assert context.allocate_sample_id("same.txt") == "same"
    assert context.allocate_sample_id("same.csv") == "same_2"
    assert context.sample_dir("same").exists()
