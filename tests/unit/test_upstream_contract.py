from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INSPECT_SCRIPT = PROJECT_ROOT / "scripts" / "inspect_upstream.py"


def load_inspect_module():
    spec = importlib.util.spec_from_file_location("inspect_upstream", INSPECT_SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_model_contract_records_required_phase1_answers() -> None:
    text = (PROJECT_ROOT / "docs" / "model_contract.md").read_text(encoding="utf-8")

    required_fragments = [
        "torch.float32",
        "(batch, 1, 4500)",
        "10.0",
        "80.0",
        "23073",
        "raw logits",
        "CPICANNsingle_phase_D1.pth",
        "d2e898bb4b7482cd7b14953feac437b053617815746f025a6b88ca014e51be98",
        "top-level keys actually present",
        "UNKNOWN",
        "anno_struc.csv",
        "01e915697f544e5a4fcd0f03f76da0e38ee00ece",
        "3dbfaeab51d272e013d211c7f957760b46ab41cc",
        "85b7e2ce9060286c84efc459cfde875ad5ca30ec",
    ]
    for fragment in required_fragments:
        assert fragment in text


def test_preprocessing_contract_records_legacy_behavior_and_missing_norm_fixtures() -> None:
    text = (PROJECT_ROOT / "docs" / "preprocessing_contract.md").read_text(encoding="utf-8")

    required_fragments = [
        "legacy-cpicann-v1",
        "txt",
        "csv",
        "xy",
        "np.linspace(10, 80, 4500)",
        'kind="slinear"',
        "v / v.max() * 100",
        "0-norm.txt",
        "1-norm.txt",
        "3-norm.txt",
        "UNKNOWN",
    ]
    for fragment in required_fragments:
        assert fragment in text


def test_adr_rejects_direct_wpemphase_phaseidentifier() -> None:
    text = (PROJECT_ROOT / "docs" / "adr" / "0001-cpicann-upstream-integration.md").read_text(
        encoding="utf-8"
    )

    assert "Do not call `WPEMPhase.PhaseIdentifier`" in text
    assert "minimal network definition" in text
    assert "Phase 4 must still verify actual checkpoint top-level keys" in text


def test_lfs_pointer_parser(tmp_path: Path) -> None:
    module = load_inspect_module()
    pointer = tmp_path / "model.pth"
    pointer.write_text(
        "\n".join(
            [
                "version https://git-lfs.github.com/spec/v1",
                "oid sha256:" + "a" * 64,
                "size 123",
            ]
        ),
        encoding="utf-8",
    )

    parsed = module.parse_lfs_pointer(pointer)

    assert parsed.exists is True
    assert parsed.is_lfs_pointer is True
    assert parsed.oid_sha256 == "a" * 64
    assert parsed.size_bytes == 123


def test_catalog_inspection_validates_contiguous_class_indexes(tmp_path: Path) -> None:
    module = load_inspect_module()
    catalog = tmp_path / "anno_struc.csv"
    with catalog.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataId", "No", "formula"])
        writer.writeheader()
        writer.writerow({"dataId": "100", "No": "0", "formula": "Li2 O1"})
        writer.writerow({"dataId": "101", "No": "1", "formula": "Zr1 O2"})

    inspected = module.inspect_catalog(catalog)

    assert inspected.exists is True
    assert inspected.row_count == 2
    assert inspected.no_min == 0
    assert inspected.no_max == 1
    assert inspected.no_unique is True
    assert inspected.no_contiguous is True
