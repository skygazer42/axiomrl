import json
from pathlib import Path
from unittest.mock import patch

import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from axiomrl.experiment.checkpointing import load_checkpoint
from axiomrl.experiment.config import TrainConfig
from axiomrl.runtime.run_utils import (
    create_training_run,
    resolve_device,
    save_training_checkpoint,
    serialize_train_config,
)


def test_serialize_train_config_normalizes_paths_and_tags(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="ppo",
        env_id="CartPole-v1",
        seed=5,
        total_timesteps=128,
        output_dir=tmp_path,
        execution_backend="local_async",
        tags=("baseline", "phase1"),
    )

    payload = serialize_train_config(config)

    assert payload["output_dir"] == str(tmp_path)
    assert payload["execution_backend"] == "local_async"
    assert payload["tags"] == ["baseline", "phase1"]


def test_create_training_run_writes_config_and_metadata(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="dqn",
        env_id="CartPole-v1",
        seed=7,
        total_timesteps=256,
        output_dir=tmp_path,
    )

    artifacts = create_training_run(config, run_suffix="shared")
    try:
        assert artifacts.run_context.config_path.exists()
        assert artifacts.run_context.metadata_path.exists()
        assert artifacts.run_context.run_dir.exists()
        metadata_payload = json.loads(artifacts.run_context.metadata_path.read_text(encoding="utf-8"))
        assert metadata_payload["algo"] == "dqn"
        assert metadata_payload["env_id"] == "CartPole-v1"
        assert metadata_payload["seed"] == 7
        assert metadata_payload["output_dir"] == str(artifacts.run_context.run_dir)
        assert isinstance(metadata_payload.get("benchmark"), dict)

        assert isinstance(metadata_payload.get("created_at_utc"), str)

        command = metadata_payload.get("command")
        assert isinstance(command, list)
        assert all(isinstance(token, str) for token in command)
        assert command

        system_payload = metadata_payload.get("system")
        assert isinstance(system_payload, dict)
        assert isinstance(system_payload.get("python_version"), str)
        assert isinstance(system_payload.get("platform"), str)

        versions_payload = metadata_payload.get("versions")
        assert isinstance(versions_payload, dict)
        assert isinstance(versions_payload.get("torch"), str)
        assert isinstance(versions_payload.get("gymnasium"), str)
        assert isinstance(versions_payload.get("numpy"), str)

        git_payload = metadata_payload.get("git")
        assert isinstance(git_payload, dict)
        assert "commit" in git_payload
        assert "is_dirty" in git_payload
    finally:
        artifacts.close()


def test_create_training_run_emits_tensorboard_event_file(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="ppo",
        env_id="CartPole-v1",
        seed=13,
        total_timesteps=64,
        output_dir=tmp_path,
    )

    artifacts = create_training_run(config, run_suffix="tensorboard")
    try:
        artifacts.logger.log_metrics({"loss": 1.25}, step=7)
    finally:
        artifacts.close()

    accumulator = EventAccumulator(str(artifacts.run_context.tensorboard_dir))
    accumulator.Reload()
    scalars = accumulator.Scalars("loss")

    assert scalars[-1].step == 7
    assert scalars[-1].value == pytest.approx(1.25)


def test_save_training_checkpoint_writes_loadable_checkpoint(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="sac",
        env_id="Pendulum-v1",
        seed=11,
        total_timesteps=64,
        output_dir=tmp_path,
    )

    artifacts = create_training_run(config, run_suffix="checkpoint")
    try:
        checkpoint_path = save_training_checkpoint(
            run_context=artifacts.run_context,
            config=config,
            algorithm_state={"model": {"weights": [1, 2, 3]}},
            buffer_state={"size": 8},
            trainer_state={"global_step": 64},
            metrics={"loss": 0.5},
        )
    finally:
        artifacts.close()

    loaded = load_checkpoint(checkpoint_path)

    assert checkpoint_path.exists()
    assert loaded.algorithm_state == {"model": {"weights": [1, 2, 3]}}
    assert loaded.buffer_state == {"size": 8}
    assert loaded.trainer_state == {"global_step": 64}
    assert loaded.metadata["metrics"].keys() == {"loss"}
    assert loaded.metadata["metrics"]["loss"] == pytest.approx(0.5)


def test_resolve_device_auto_falls_back_to_cpu_when_cuda_probe_fails() -> None:
    with patch("axiomrl.runtime.run_utils.torch.cuda.is_available", return_value=True):
        with patch("axiomrl.runtime.run_utils.torch.empty", side_effect=RuntimeError("oom")):
            device = resolve_device("auto")

    assert str(device) == "cpu"
