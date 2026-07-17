"""Console environment check for Docker smoke tests."""

from __future__ import annotations

from xdecomposer_service.health import build_ready
from xdecomposer_service.settings import load_settings


def main() -> None:
    """Print readiness JSON without requiring assets to be present."""
    print(build_ready(load_settings()).model_dump_json())


if __name__ == "__main__":
    main()
