from cpicann_xrd import __version__


def test_version_is_development_release() -> None:
    assert __version__ == "0.1.0.dev0"
