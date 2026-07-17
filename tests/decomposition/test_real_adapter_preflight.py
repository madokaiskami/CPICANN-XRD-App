from __future__ import annotations

import os
from pathlib import Path

import pytest

from cpicann_xrd.decomposition.assets import verify_xdecomposer_assets


@pytest.mark.xdecomposer_model
def test_xdecomposer_real_asset_gate_from_environment() -> None:
    manifest_value = os.environ.get("XDECOMPOSER_MANIFEST")
    if not manifest_value:
        pytest.skip("XDECOMPOSER_MANIFEST is not configured")

    result = verify_xdecomposer_assets(Path(manifest_value), production=True)

    assert result.status == "ok"
    assert result.assets
