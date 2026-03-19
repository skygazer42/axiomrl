from pathlib import Path

import pytest

from rl_training.experiment.config import TrainConfig
from rl_training.experiment.sweeps import resolve_benchmark_seeds


def _base_config(tmp_path: Path, *, benchmark: dict[str, object]) -> TrainConfig:
    return TrainConfig(
        algo="ppo",
        env_id="CartPole-v1",
        seed=7,
        total_timesteps=64,
        output_dir=tmp_path,
        benchmark=benchmark,
    )


def test_resolve_benchmark_seeds_returns_empty_tuple_when_unset(tmp_path: Path) -> None:
    config = _base_config(tmp_path, benchmark={})

    assert resolve_benchmark_seeds(config) == ()


def test_resolve_benchmark_seeds_normalizes_distinct_integers(tmp_path: Path) -> None:
    config = _base_config(tmp_path, benchmark={"seeds": [3, 5, 3, 7]})

    assert resolve_benchmark_seeds(config) == (3, 5, 7)


def test_resolve_benchmark_seeds_rejects_empty_seed_list(tmp_path: Path) -> None:
    config = _base_config(tmp_path, benchmark={"seeds": []})

    with pytest.raises(ValueError, match="benchmark\\['seeds'\\] must not be empty"):
        resolve_benchmark_seeds(config)


def test_resolve_benchmark_seeds_rejects_non_sequence_seed_values(tmp_path: Path) -> None:
    config = _base_config(tmp_path, benchmark={"seeds": "1,2,3"})

    with pytest.raises(TypeError, match="benchmark\\['seeds'\\] must be a sequence of integers"):
        resolve_benchmark_seeds(config)


def test_resolve_benchmark_seeds_rejects_non_integer_entries(tmp_path: Path) -> None:
    config = _base_config(tmp_path, benchmark={"seeds": [1, "bad"]})

    with pytest.raises(TypeError, match="benchmark\\['seeds'\\] entries must be integers"):
        resolve_benchmark_seeds(config)


def test_resolve_benchmark_seeds_rejects_negative_seed_values(tmp_path: Path) -> None:
    config = _base_config(tmp_path, benchmark={"seeds": [1, -3]})

    with pytest.raises(ValueError, match="benchmark\\['seeds'\\] entries must be non-negative integers"):
        resolve_benchmark_seeds(config)
