from rl_training import __version__


def test_package_exports_version() -> None:
    assert __version__ == "0.1.0"
