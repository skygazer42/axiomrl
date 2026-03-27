PYTHON ?= python

RUFF_TARGETS = \
	src/rl_training/cli.py \
	src/rl_training/cli_config.py \
	src/rl_training/cli_doctor.py \
	src/rl_training/cli_zoo.py \
	src/rl_training/zoo/reporting.py \
	src/rl_training/zoo/reporting_render.py \
	src/rl_training/zoo/reporting_runs.py \
	src/rl_training/zoo/reporting_stats.py \
	src/rl_training/experiment/registry_actor_critic_specs.py \
	src/rl_training/experiment/registry_core.py \
	src/rl_training/experiment/registry_continuous_loaders.py \
	src/rl_training/experiment/registry_dqn_loaders.py \
	src/rl_training/experiment/registry_evaluators.py \
	src/rl_training/experiment/registry_offline_loaders.py \
	src/rl_training/experiment/registry_offline_specs.py \
	src/rl_training/experiment/registry_on_policy_specs.py \
	src/rl_training/experiment/registry_policy_loaders.py \
	src/rl_training/experiment/registry_predictors.py \
	src/rl_training/experiment/registry_recurrent_loaders.py \
	src/rl_training/experiment/registry_specialized_loaders.py \
	src/rl_training/experiment/registry_value_based_specs.py \
	src/rl_training/experiment/registry_world_model_specs.py \
	tests/conftest.py \
	tests/support/checkpoint_workflows.py \
	tests/support/markers.py \
	tests/support/public_api.py \
	tests/support/runtime_foundation.py \
	tests/test_algorithm_registry_contracts.py \
	tests/test_checkpoint_evaluate.py \
	tests/test_checkpoint_resume.py \
	tests/test_cli_config.py \
	tests/test_cli_workflows.py \
	tests/test_cli_zoo_leaderboard.py \
	tests/test_cli_zoo_report.py \
	tests/test_experiment_manager_workflows.py \
	tests/test_public_api_continuous_control.py \
	tests/test_public_api_off_policy_suite.py \
	tests/test_public_api_policy_gradient.py \
	tests/test_public_api_visual_control.py \
	tests/test_registry_internal_split.py \
	tests/test_registry_providers.py \
	tests/test_release_contracts.py \
	tests/test_runtime_evaluation_support_integration.py \
	tests/test_runtime_training_session_integration.py \
	tests/test_test_markers.py \
	tests/test_zoo_modules.py \
	tests/test_zoo_reporting_split.py

TYPECHECK_TARGETS = \
	src/rl_training/cli.py \
	src/rl_training/cli_config.py \
	src/rl_training/cli_doctor.py \
	src/rl_training/cli_zoo.py \
	src/rl_training/zoo/reporting.py \
	src/rl_training/zoo/reporting_render.py \
	src/rl_training/zoo/reporting_runs.py \
	src/rl_training/zoo/reporting_stats.py \
	src/rl_training/zoo/core.py \
	src/rl_training/experiment/registry_actor_critic_specs.py \
	src/rl_training/experiment/registry_core.py \
	src/rl_training/experiment/registry_continuous_loaders.py \
	src/rl_training/experiment/registry_dqn_loaders.py \
	src/rl_training/experiment/registry_evaluators.py \
	src/rl_training/experiment/registry_offline_loaders.py \
	src/rl_training/experiment/registry_offline_specs.py \
	src/rl_training/experiment/registry_on_policy_specs.py \
	src/rl_training/experiment/registry_policy_loaders.py \
	src/rl_training/experiment/registry_predictors.py \
	src/rl_training/experiment/registry_recurrent_loaders.py \
	src/rl_training/experiment/registry_specialized_loaders.py \
	src/rl_training/experiment/registry_value_based_specs.py \
	src/rl_training/experiment/registry_world_model_specs.py \
	src/rl_training/experiment/registry_types.py \
	tests/conftest.py \
	tests/support/checkpoint_workflows.py \
	tests/support/markers.py \
	tests/support/public_api.py \
	tests/support/runtime_foundation.py \
	tests/test_algorithm_registry_contracts.py \
	tests/test_checkpoint_evaluate.py \
	tests/test_checkpoint_resume.py \
	tests/test_cli_config.py \
	tests/test_cli_workflows.py \
	tests/test_cli_zoo_leaderboard.py \
	tests/test_cli_zoo_report.py \
	tests/test_experiment_manager_workflows.py \
	tests/test_public_api_continuous_control.py \
	tests/test_public_api_off_policy_suite.py \
	tests/test_public_api_policy_gradient.py \
	tests/test_public_api_visual_control.py \
	tests/test_registry_internal_split.py \
	tests/test_registry_providers.py \
	tests/test_release_contracts.py \
	tests/test_runtime_evaluation_support_integration.py \
	tests/test_runtime_training_session_integration.py \
	tests/test_test_markers.py \
	tests/test_zoo_modules.py \
	tests/test_zoo_reporting_split.py

.PHONY: install-dev lint typecheck test test-fast test-integration test-smoke build precommit verify

install-dev:
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	$(PYTHON) -m ruff check $(RUFF_TARGETS)
	$(PYTHON) -m ruff format --check $(RUFF_TARGETS)

typecheck:
	$(PYTHON) -m mypy $(TYPECHECK_TARGETS)

test:
	$(PYTHON) -m pytest -q

test-fast:
	$(PYTHON) -m pytest -q -m "unit and not slow"

test-integration:
	$(PYTHON) -m pytest -q -m "integration and not slow"

test-smoke:
	$(PYTHON) -m pytest -q -m "smoke and not slow"

build:
	$(PYTHON) -m build --sdist --wheel

precommit:
	$(PYTHON) -m pre_commit run --all-files

verify:
	$(MAKE) lint
	$(MAKE) typecheck
	$(MAKE) test-fast
	$(MAKE) test-integration
	$(MAKE) test-smoke
	$(MAKE) build
