from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_declares_release_metadata_and_optional_installs() -> None:
    pyproject_text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'version = "1.0.0"' in pyproject_text
    assert 'Development Status :: 4 - Beta' in pyproject_text
    assert 'Intended Audience :: Developers' in pyproject_text
    assert 'project.urls' in pyproject_text
    assert 'dev = [' in pyproject_text
    assert 'experimental = [' in pyproject_text
    assert '"build"' in pyproject_text
    assert '"twine"' in pyproject_text


def test_repository_declares_ci_and_publish_workflows() -> None:
    ci_workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml")
    publish_workflow = (REPO_ROOT / ".github" / "workflows" / "publish.yml")

    assert ci_workflow.exists()
    assert publish_workflow.exists()

    ci_text = ci_workflow.read_text(encoding="utf-8")
    publish_text = publish_workflow.read_text(encoding="utf-8")

    assert "strategy:" in ci_text
    assert "python-version" in ci_text
    assert "python -m build" in ci_text
    assert "pytest -q" in ci_text
    assert "twine check" in ci_text
    assert "pip install dist/*.whl" in ci_text
    assert 'import rl_training' in ci_text
    assert "axiomrl --help" in ci_text
    assert "workflow_dispatch" in publish_text
    assert "gh-action-pypi-publish" in publish_text


def test_readme_documents_stable_core_and_version_policy() -> None:
    readme_text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "Stable Core API" in readme_text
    assert "rl_training.core" in readme_text
    assert "rl_training.experimental" in readme_text
    assert "Semantic Versioning" in readme_text
    assert "deprecated" in readme_text.lower()


def test_repository_includes_compatibility_policy_and_changelog() -> None:
    compatibility_doc = REPO_ROOT / "docs" / "compatibility.md"
    changelog = REPO_ROOT / "CHANGELOG.md"

    assert compatibility_doc.exists()
    assert changelog.exists()

    compatibility_text = compatibility_doc.read_text(encoding="utf-8")
    changelog_text = changelog.read_text(encoding="utf-8")

    assert "stable" in compatibility_text.lower()
    assert "experimental" in compatibility_text.lower()
    assert "minor" in compatibility_text.lower()
    assert "# Changelog" in changelog_text
