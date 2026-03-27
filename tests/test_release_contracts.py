from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_declares_release_metadata_and_optional_installs() -> None:
    pyproject_text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'version = "1.0.0"' in pyproject_text
    assert "Development Status :: 4 - Beta" in pyproject_text
    assert "Intended Audience :: Developers" in pyproject_text
    assert "project.urls" in pyproject_text
    assert "dev = [" in pyproject_text
    assert "experimental = [" in pyproject_text
    assert '"build"' in pyproject_text
    assert '"twine"' in pyproject_text


def test_repository_declares_ci_and_publish_workflows() -> None:
    ci_workflow = REPO_ROOT / ".github" / "workflows" / "ci.yml"
    publish_workflow = REPO_ROOT / ".github" / "workflows" / "publish.yml"

    assert ci_workflow.exists()
    assert publish_workflow.exists()

    ci_text = ci_workflow.read_text(encoding="utf-8")
    publish_text = publish_workflow.read_text(encoding="utf-8")

    assert "strategy:" in ci_text
    assert "python-version" in ci_text
    assert "ruff check" in ci_text
    assert "python -m mypy" in ci_text
    assert 'pytest -q -m "unit and not slow"' in ci_text
    assert 'pytest -q -m "integration and not slow"' in ci_text
    assert 'pytest -q -m "smoke and not slow"' in ci_text
    assert "python -m build" in ci_text
    assert "pytest -q" in ci_text
    assert "twine check" in ci_text
    assert "pip install dist/*.whl" in ci_text
    assert "import rl_training" in ci_text
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


def test_repository_includes_development_tooling_contracts() -> None:
    pyproject_text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert '"pre-commit"' in pyproject_text
    assert '"ruff"' in pyproject_text
    assert '"mypy"' in pyproject_text
    assert "[tool.ruff]" in pyproject_text
    assert "[tool.mypy]" in pyproject_text
    assert "unit: fast isolated tests" in pyproject_text
    assert "integration: cross-module workflow tests" in pyproject_text
    assert "smoke: representative runtime smoke tests" in pyproject_text
    assert "slow: tests excluded from the default fast path" in pyproject_text

    assert (REPO_ROOT / ".pre-commit-config.yaml").exists()
    assert (REPO_ROOT / "Makefile").exists()
    assert (REPO_ROOT / "CONTRIBUTING.md").exists()
    assert (REPO_ROOT / "docs" / "development.md").exists()


def test_repository_includes_split_cli_test_modules() -> None:
    expected_modules = [
        REPO_ROOT / "tests" / "test_cli_config.py",
        REPO_ROOT / "tests" / "test_cli_zoo_report.py",
        REPO_ROOT / "tests" / "test_cli_zoo_leaderboard.py",
        REPO_ROOT / "tests" / "test_cli_workflows.py",
    ]

    for module_path in expected_modules:
        assert module_path.exists(), f"missing split CLI test module: {module_path.name}"


def test_repository_includes_split_checkpoint_workflow_test_modules() -> None:
    expected_modules = [
        REPO_ROOT / "tests" / "support" / "checkpoint_workflows.py",
        REPO_ROOT / "tests" / "test_checkpoint_evaluate.py",
        REPO_ROOT / "tests" / "test_checkpoint_resume.py",
    ]

    for module_path in expected_modules:
        assert module_path.exists(), f"missing split checkpoint workflow module: {module_path.name}"


def test_repository_includes_split_runtime_foundation_test_modules() -> None:
    expected_modules = [
        REPO_ROOT / "tests" / "support" / "runtime_foundation.py",
        REPO_ROOT / "tests" / "test_runtime_training_session_integration.py",
        REPO_ROOT / "tests" / "test_runtime_evaluation_support_integration.py",
    ]

    for module_path in expected_modules:
        assert module_path.exists(), f"missing split runtime foundation module: {module_path.name}"


def test_repository_includes_split_experiment_manager_test_modules() -> None:
    monolith = REPO_ROOT / "tests" / "test_experiment_manager.py"
    expected_modules = [
        REPO_ROOT / "tests" / "test_algorithm_registry_contracts.py",
        REPO_ROOT / "tests" / "test_experiment_manager_workflows.py",
    ]

    assert not monolith.exists(), "legacy experiment manager test monolith should be removed"

    for module_path in expected_modules:
        assert module_path.exists(), f"missing split experiment manager module: {module_path.name}"


def test_quality_gates_cover_split_experiment_manager_test_modules() -> None:
    expected_names = (
        "tests/test_algorithm_registry_contracts.py",
        "tests/test_experiment_manager_workflows.py",
    )

    quality_gate_texts = {
        "pyproject": (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"),
        "makefile": (REPO_ROOT / "Makefile").read_text(encoding="utf-8"),
        "pre-commit": (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8"),
        "ci": (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"),
    }

    for expected_name in expected_names:
        for gate_name, gate_text in quality_gate_texts.items():
            assert expected_name in gate_text, f"{expected_name} missing from {gate_name} quality gate"


def test_repository_includes_split_public_api_test_modules() -> None:
    monolith = REPO_ROOT / "tests" / "test_public_api.py"
    expected_modules = [
        REPO_ROOT / "tests" / "support" / "public_api.py",
        REPO_ROOT / "tests" / "test_public_api_continuous_control.py",
        REPO_ROOT / "tests" / "test_public_api_policy_gradient.py",
        REPO_ROOT / "tests" / "test_public_api_visual_control.py",
        REPO_ROOT / "tests" / "test_public_api_off_policy_suite.py",
    ]

    assert not monolith.exists(), "legacy public API test monolith should be removed"

    for module_path in expected_modules:
        assert module_path.exists(), f"missing split public API module: {module_path.name}"


def test_quality_gates_cover_split_public_api_test_modules() -> None:
    expected_names = (
        "tests/support/public_api.py",
        "tests/test_public_api_continuous_control.py",
        "tests/test_public_api_policy_gradient.py",
        "tests/test_public_api_visual_control.py",
        "tests/test_public_api_off_policy_suite.py",
    )

    quality_gate_texts = {
        "pyproject": (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"),
        "makefile": (REPO_ROOT / "Makefile").read_text(encoding="utf-8"),
        "pre-commit": (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8"),
        "ci": (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"),
    }

    for expected_name in expected_names:
        for gate_name, gate_text in quality_gate_texts.items():
            assert expected_name in gate_text, f"{expected_name} missing from {gate_name} quality gate"


def test_repository_includes_split_registry_dqn_loader_module() -> None:
    module_path = REPO_ROOT / "src" / "rl_training" / "experiment" / "registry_dqn_loaders.py"

    assert module_path.exists(), "missing split registry DQN loader module"


def test_quality_gates_cover_split_registry_dqn_loader_module() -> None:
    expected_name = "src/rl_training/experiment/registry_dqn_loaders.py"
    quality_gate_texts = {
        "pyproject": (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"),
        "makefile": (REPO_ROOT / "Makefile").read_text(encoding="utf-8"),
        "pre-commit": (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8"),
        "ci": (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"),
    }

    for gate_name, gate_text in quality_gate_texts.items():
        assert expected_name in gate_text, f"{expected_name} missing from {gate_name} quality gate"


def test_repository_includes_split_registry_continuous_loader_module() -> None:
    module_path = REPO_ROOT / "src" / "rl_training" / "experiment" / "registry_continuous_loaders.py"

    assert module_path.exists(), "missing split registry continuous loader module"


def test_quality_gates_cover_split_registry_continuous_loader_module() -> None:
    expected_name = "src/rl_training/experiment/registry_continuous_loaders.py"
    quality_gate_texts = {
        "pyproject": (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"),
        "makefile": (REPO_ROOT / "Makefile").read_text(encoding="utf-8"),
        "pre-commit": (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8"),
        "ci": (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"),
    }

    for gate_name, gate_text in quality_gate_texts.items():
        assert expected_name in gate_text, f"{expected_name} missing from {gate_name} quality gate"


def test_repository_includes_split_registry_recurrent_loader_module() -> None:
    module_path = REPO_ROOT / "src" / "rl_training" / "experiment" / "registry_recurrent_loaders.py"

    assert module_path.exists(), "missing split registry recurrent loader module"


def test_quality_gates_cover_split_registry_recurrent_loader_module() -> None:
    expected_name = "src/rl_training/experiment/registry_recurrent_loaders.py"
    quality_gate_texts = {
        "pyproject": (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"),
        "makefile": (REPO_ROOT / "Makefile").read_text(encoding="utf-8"),
        "pre-commit": (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8"),
        "ci": (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"),
    }

    for gate_name, gate_text in quality_gate_texts.items():
        assert expected_name in gate_text, f"{expected_name} missing from {gate_name} quality gate"


def test_repository_includes_split_registry_policy_loader_module() -> None:
    module_path = REPO_ROOT / "src" / "rl_training" / "experiment" / "registry_policy_loaders.py"

    assert module_path.exists(), "missing split registry policy loader module"


def test_quality_gates_cover_split_registry_policy_loader_module() -> None:
    expected_name = "src/rl_training/experiment/registry_policy_loaders.py"
    quality_gate_texts = {
        "pyproject": (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"),
        "makefile": (REPO_ROOT / "Makefile").read_text(encoding="utf-8"),
        "pre-commit": (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8"),
        "ci": (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"),
    }

    for gate_name, gate_text in quality_gate_texts.items():
        assert expected_name in gate_text, f"{expected_name} missing from {gate_name} quality gate"


def test_repository_includes_split_registry_specialized_loader_module() -> None:
    module_path = REPO_ROOT / "src" / "rl_training" / "experiment" / "registry_specialized_loaders.py"

    assert module_path.exists(), "missing split registry specialized loader module"


def test_quality_gates_cover_split_registry_specialized_loader_module() -> None:
    expected_name = "src/rl_training/experiment/registry_specialized_loaders.py"
    quality_gate_texts = {
        "pyproject": (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"),
        "makefile": (REPO_ROOT / "Makefile").read_text(encoding="utf-8"),
        "pre-commit": (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8"),
        "ci": (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"),
    }

    for gate_name, gate_text in quality_gate_texts.items():
        assert expected_name in gate_text, f"{expected_name} missing from {gate_name} quality gate"


def test_repository_includes_split_registry_offline_loader_module() -> None:
    module_path = REPO_ROOT / "src" / "rl_training" / "experiment" / "registry_offline_loaders.py"

    assert module_path.exists(), "missing split registry offline loader module"


def test_quality_gates_cover_split_registry_offline_loader_module() -> None:
    expected_name = "src/rl_training/experiment/registry_offline_loaders.py"
    quality_gate_texts = {
        "pyproject": (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"),
        "makefile": (REPO_ROOT / "Makefile").read_text(encoding="utf-8"),
        "pre-commit": (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8"),
        "ci": (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"),
    }

    for gate_name, gate_text in quality_gate_texts.items():
        assert expected_name in gate_text, f"{expected_name} missing from {gate_name} quality gate"


def test_repository_includes_split_registry_on_policy_specs_module() -> None:
    module_path = REPO_ROOT / "src" / "rl_training" / "experiment" / "registry_on_policy_specs.py"

    assert module_path.exists(), "missing split registry on-policy specs module"


def test_quality_gates_cover_split_registry_on_policy_specs_module() -> None:
    expected_name = "src/rl_training/experiment/registry_on_policy_specs.py"
    quality_gate_texts = {
        "pyproject": (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"),
        "makefile": (REPO_ROOT / "Makefile").read_text(encoding="utf-8"),
        "pre-commit": (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8"),
        "ci": (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"),
    }

    for gate_name, gate_text in quality_gate_texts.items():
        assert expected_name in gate_text, f"{expected_name} missing from {gate_name} quality gate"


def test_repository_includes_split_registry_offline_specs_module() -> None:
    module_path = REPO_ROOT / "src" / "rl_training" / "experiment" / "registry_offline_specs.py"

    assert module_path.exists(), "missing split registry offline specs module"


def test_quality_gates_cover_split_registry_offline_specs_module() -> None:
    expected_name = "src/rl_training/experiment/registry_offline_specs.py"
    quality_gate_texts = {
        "pyproject": (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"),
        "makefile": (REPO_ROOT / "Makefile").read_text(encoding="utf-8"),
        "pre-commit": (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8"),
        "ci": (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"),
    }

    for gate_name, gate_text in quality_gate_texts.items():
        assert expected_name in gate_text, f"{expected_name} missing from {gate_name} quality gate"


def test_repository_includes_split_registry_world_model_specs_module() -> None:
    module_path = REPO_ROOT / "src" / "rl_training" / "experiment" / "registry_world_model_specs.py"

    assert module_path.exists(), "missing split registry world-model specs module"


def test_quality_gates_cover_split_registry_world_model_specs_module() -> None:
    expected_name = "src/rl_training/experiment/registry_world_model_specs.py"
    quality_gate_texts = {
        "pyproject": (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"),
        "makefile": (REPO_ROOT / "Makefile").read_text(encoding="utf-8"),
        "pre-commit": (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8"),
        "ci": (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"),
    }

    for gate_name, gate_text in quality_gate_texts.items():
        assert expected_name in gate_text, f"{expected_name} missing from {gate_name} quality gate"


def test_repository_includes_split_registry_actor_critic_specs_module() -> None:
    module_path = REPO_ROOT / "src" / "rl_training" / "experiment" / "registry_actor_critic_specs.py"

    assert module_path.exists(), "missing split registry actor-critic specs module"


def test_quality_gates_cover_split_registry_actor_critic_specs_module() -> None:
    expected_name = "src/rl_training/experiment/registry_actor_critic_specs.py"
    quality_gate_texts = {
        "pyproject": (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"),
        "makefile": (REPO_ROOT / "Makefile").read_text(encoding="utf-8"),
        "pre-commit": (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8"),
        "ci": (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"),
    }

    for gate_name, gate_text in quality_gate_texts.items():
        assert expected_name in gate_text, f"{expected_name} missing from {gate_name} quality gate"


def test_repository_includes_split_registry_value_based_specs_module() -> None:
    module_path = REPO_ROOT / "src" / "rl_training" / "experiment" / "registry_value_based_specs.py"

    assert module_path.exists(), "missing split registry value-based specs module"


def test_quality_gates_cover_split_registry_value_based_specs_module() -> None:
    expected_name = "src/rl_training/experiment/registry_value_based_specs.py"
    quality_gate_texts = {
        "pyproject": (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"),
        "makefile": (REPO_ROOT / "Makefile").read_text(encoding="utf-8"),
        "pre-commit": (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8"),
        "ci": (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"),
    }

    for gate_name, gate_text in quality_gate_texts.items():
        assert expected_name in gate_text, f"{expected_name} missing from {gate_name} quality gate"
