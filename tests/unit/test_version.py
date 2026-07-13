from cpicann_xrd import __version__


def test_version_is_release_candidate() -> None:
    assert __version__ == "0.1.0rc1"
