"""Read-only helpers for inspecting CPICANN upstream artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class GitInspection:
    path: str
    exists: bool
    head: str | None
    remotes: list[str]


@dataclass(frozen=True)
class CatalogInspection:
    path: str
    exists: bool
    row_count: int | None
    no_min: int | None
    no_max: int | None
    no_unique: bool | None
    no_contiguous: bool | None


@dataclass(frozen=True)
class LfsPointer:
    path: str
    exists: bool
    is_lfs_pointer: bool
    oid_sha256: str | None
    size_bytes: int | None


@dataclass(frozen=True)
class SourceInspection:
    root: str
    exists: bool
    git: GitInspection
    has_model_definition: bool
    has_data_format: bool
    has_catalog: bool


@dataclass(frozen=True)
class PretrainInspection:
    root: str
    exists: bool
    git: GitInspection
    weights: list[LfsPointer]


def _run_git(path: Path, args: list[str]) -> list[str]:
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    return [line for line in completed.stdout.splitlines() if line]


def inspect_git(path: Path) -> GitInspection:
    if not path.exists():
        return GitInspection(path=str(path), exists=False, head=None, remotes=[])

    head_lines = _run_git(path, ["rev-parse", "HEAD"])
    remotes = _run_git(path, ["remote", "-v"])
    return GitInspection(
        path=str(path),
        exists=True,
        head=head_lines[0] if head_lines else None,
        remotes=remotes,
    )


def inspect_catalog(path: Path) -> CatalogInspection:
    if not path.exists():
        return CatalogInspection(
            path=str(path),
            exists=False,
            row_count=None,
            no_min=None,
            no_max=None,
            no_unique=None,
            no_contiguous=None,
        )

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    class_indexes = [int(row["No"]) for row in rows]
    return CatalogInspection(
        path=str(path),
        exists=True,
        row_count=len(rows),
        no_min=min(class_indexes) if class_indexes else None,
        no_max=max(class_indexes) if class_indexes else None,
        no_unique=len(set(class_indexes)) == len(class_indexes),
        no_contiguous=class_indexes == list(range(len(class_indexes))),
    )


def parse_lfs_pointer(path: Path) -> LfsPointer:
    if not path.exists():
        return LfsPointer(
            path=str(path),
            exists=False,
            is_lfs_pointer=False,
            oid_sha256=None,
            size_bytes=None,
        )

    oid_sha256: str | None = None
    size_bytes: int | None = None
    is_lfs_pointer = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line == "version https://git-lfs.github.com/spec/v1":
            is_lfs_pointer = True
        elif line.startswith("oid sha256:"):
            oid_sha256 = line.removeprefix("oid sha256:")
        elif line.startswith("size "):
            size_bytes = int(line.removeprefix("size "))

    return LfsPointer(
        path=str(path),
        exists=True,
        is_lfs_pointer=is_lfs_pointer,
        oid_sha256=oid_sha256,
        size_bytes=size_bytes,
    )


def inspect_source(root: Path) -> SourceInspection:
    model_path = root / "src" / "model" / "CPICANN.py"
    data_format_path = root / "src" / "data_format.py"
    catalog_path = root / "src" / "annotation" / "anno_struc.csv"
    return SourceInspection(
        root=str(root),
        exists=root.exists(),
        git=inspect_git(root),
        has_model_definition=model_path.exists(),
        has_data_format=data_format_path.exists(),
        has_catalog=catalog_path.exists(),
    )


def inspect_pretrain(root: Path) -> PretrainInspection:
    weights = (
        [parse_lfs_pointer(path) for path in sorted(root.glob("*.pth"))] if root.exists() else []
    )
    return PretrainInspection(
        root=str(root),
        exists=root.exists(),
        git=inspect_git(root),
        weights=weights,
    )


def build_report(source_root: Path, pretrain_root: Path) -> dict[str, object]:
    catalog = inspect_catalog(source_root / "src" / "annotation" / "anno_struc.csv")
    source = inspect_source(source_root)
    pretrain = inspect_pretrain(pretrain_root)
    return {
        "source": asdict(source),
        "pretrain": asdict(pretrain),
        "catalog": asdict(catalog),
    }


def print_human_report(report: dict[str, object]) -> None:
    print("CPICANN upstream inspection")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect local CPICANN upstream clones without downloading model weights.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/tmp/cpicann_phase1_src"),
        help="Local clone of https://huggingface.co/AI4Cryst/CPICANN.",
    )
    parser.add_argument(
        "--pretrain-root",
        type=Path,
        default=Path("/tmp/cpicann_phase1_pretrain"),
        help="Local clone of https://huggingface.co/caobin/pretrainCPICANN.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args.source_root, args.pretrain_root)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_human_report(report)


if __name__ == "__main__":
    main()
