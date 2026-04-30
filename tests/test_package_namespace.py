import importlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_project_uses_axiomrl_as_import_package() -> None:
    package = importlib.import_module("axiomrl")

    assert package.__name__ == "axiomrl"
    assert (REPO_ROOT / "src" / "axiomrl").is_dir()
    assert not (REPO_ROOT / "src" / "rl_training").exists()


def test_pyproject_entrypoints_target_axiomrl_package() -> None:
    pyproject_text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'axiomrl = "axiomrl.cli:main"' in pyproject_text
    assert 'axiomrl-zoo = "axiomrl.zoo_cli:main"' in pyproject_text
    assert "rl_training" not in pyproject_text
