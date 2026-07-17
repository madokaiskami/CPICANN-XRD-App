from cpicann_xrd import __version__


def test_version_is_release_candidate() -> None:
    assert __version__ == "0.2.0rc1"
