from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from rl_training.experiment.checkpointing import CheckpointState, load_checkpoint, save_checkpoint
from rl_training.experiment.config import TrainConfig
from rl_training.experiment.runs import create_run_context


def test_create_run_context_creates_expected_paths(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="ppo",
        env_id="CartPole-v1",
        seed=7,
        total_timesteps=1024,
        output_dir=tmp_path,
    )

    context = create_run_context(config, run_suffix="manual")

    assert context.run_id == "ppo__CartPole-v1__seed7__manual"
    assert context.run_dir == tmp_path / context.run_id
    assert context.run_dir.exists()
    assert context.checkpoints_dir.exists()
    assert context.tensorboard_dir.exists()
    assert context.config_path == context.run_dir / "config.yaml"
    assert context.metadata_path == context.run_dir / "metadata.json"


def test_create_run_context_sanitizes_env_id_for_filesystem_paths(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="ppo",
        env_id="ALE/Breakout-v5",
        seed=7,
        total_timesteps=1024,
        output_dir=tmp_path,
    )

    context = create_run_context(config, run_suffix="manual")

    assert context.run_id == "ppo__ALE-Breakout-v5__seed7__manual"
    assert context.run_dir == tmp_path / context.run_id
    assert context.run_dir.exists()


def test_create_run_context_auto_suffixes_are_unique(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="ppo",
        env_id="CartPole-v1",
        seed=9,
        total_timesteps=256,
        output_dir=tmp_path,
    )

    first = create_run_context(config)
    second = create_run_context(config)

    assert first.run_id != second.run_id
    assert first.run_dir != second.run_dir
    assert first.run_dir.exists()
    assert second.run_dir.exists()


def test_create_run_context_auto_suffix_retries_when_timestamp_collides(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="ppo",
        env_id="CartPole-v1",
        seed=9,
        total_timesteps=256,
        output_dir=tmp_path,
    )
    fixed_now = datetime(2026, 3, 19, 12, 34, 56, 123456, tzinfo=timezone.utc)

    with patch("rl_training.experiment.runs.datetime") as mocked_datetime:
        mocked_datetime.now.return_value = fixed_now
        first = create_run_context(config)
        second = create_run_context(config)

    assert first.run_id != second.run_id
    assert first.run_dir != second.run_dir
    assert first.run_dir.exists()
    assert second.run_dir.exists()


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    state = CheckpointState(
        algorithm_state={"policy": {"weights": [1, 2, 3]}},
        buffer_state={"size": 32},
        trainer_state={"global_step": 64},
        config={"algo": "ppo", "env_id": "CartPole-v1"},
        metadata={"seed": 7},
    )

    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, state)
    loaded = load_checkpoint(path)

    assert loaded.algorithm_state == state.algorithm_state
    assert loaded.buffer_state == state.buffer_state
    assert loaded.trainer_state == state.trainer_state
    assert loaded.config == state.config
    assert loaded.metadata == state.metadata


def test_load_checkpoint_uses_torch_weights_only_mode(tmp_path: Path) -> None:
    state = CheckpointState(
        algorithm_state={"policy": {"weights": [1, 2, 3]}},
        buffer_state=None,
        trainer_state={"global_step": 8},
        config={"algo": "ppo", "env_id": "CartPole-v1"},
        metadata={"seed": 3},
    )

    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, state)

    with patch("rl_training.experiment.checkpointing.torch.load") as mocked_load:
        mocked_load.return_value = {
            "algorithm_state": state.algorithm_state,
            "buffer_state": state.buffer_state,
            "trainer_state": state.trainer_state,
            "config": state.config,
            "metadata": state.metadata,
        }
        loaded = load_checkpoint(path)

    mocked_load.assert_called_once()
    assert mocked_load.call_args.kwargs["weights_only"] is True
    assert loaded.algorithm_state == state.algorithm_state
