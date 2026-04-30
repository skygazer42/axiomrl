from pathlib import Path

import pytest

from axiomrl.tuning.config import load_study_config


def _write_base_train_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "ppo-cartpole.yaml"
    config_path.write_text(
        "\n".join(
            [
                "algo: ppo",
                "env_id: CartPole-v1",
                "seed: 7",
                "total_timesteps: 32",
                f"output_dir: {tmp_path / 'runs'}",
                "num_envs: 1",
                "eval_episodes: 1",
                "algo_kwargs:",
                "  num_steps: 32",
                "  update_epochs: 1",
                "  minibatch_size: 32",
                "  hidden_sizes: [16, 16]",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def test_load_study_config_reads_yaml_and_normalizes_search_space(tmp_path: Path) -> None:
    base_config = _write_base_train_config(tmp_path)
    study_config_path = tmp_path / "study.yaml"
    study_config_path.write_text(
        "\n".join(
            [
                f"base_config: {base_config}",
                f"output_dir: {tmp_path / 'studies'}",
                "study:",
                "  name: ppo_cartpole_tune",
                "  backend: native",
                "  sampler: grid",
                "  seed: 123",
                "  objective:",
                "    metric: global_step",
                "    mode: max",
                "search_space:",
                "  total_timesteps:",
                "    type: int",
                "    low: 32",
                "    high: 64",
                "    step: 32",
                "  algo_kwargs.num_steps:",
                "    type: categorical",
                "    values: [32, 64]",
            ]
        ),
        encoding="utf-8",
    )

    config = load_study_config(study_config_path)

    assert config.base_config_path == base_config.resolve()
    assert config.output_dir == (tmp_path / "studies").resolve()
    assert config.study.name == "ppo_cartpole_tune"
    assert config.study.backend == "native"
    assert config.study.sampler == "grid"
    assert config.study.num_trials is None
    assert config.study.objective.metric == "global_step"
    assert config.study.objective.mode == "max"
    assert config.search_space["total_timesteps"].kind == "int"
    assert config.search_space["total_timesteps"].step == 32
    assert config.search_space["algo_kwargs.num_steps"].kind == "categorical"
    assert config.search_space["algo_kwargs.num_steps"].values == (32, 64)


def test_load_study_config_rejects_forbidden_search_path(tmp_path: Path) -> None:
    base_config = _write_base_train_config(tmp_path)
    study_config_path = tmp_path / "study-invalid.yaml"
    study_config_path.write_text(
        "\n".join(
            [
                f"base_config: {base_config}",
                f"output_dir: {tmp_path / 'studies'}",
                "study:",
                "  name: invalid_tune",
                "  backend: native",
                "  sampler: random",
                "  num_trials: 2",
                "  objective:",
                "    metric: eval_return_mean",
                "    mode: max",
                "search_space:",
                "  benchmark.seeds:",
                "    type: categorical",
                "    values: [1, 2]",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="benchmark\\.seeds"):
        load_study_config(study_config_path)
