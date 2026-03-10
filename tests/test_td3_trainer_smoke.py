from pathlib import Path

import pytest
import torch

import rl_training.runtime.td3_trainer as td3_trainer
from rl_training.experiment.config import TrainConfig
from rl_training.runtime.td3_trainer import train_td3


def test_train_td3_writes_checkpoint_and_metrics(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="td3",
        env_id="Pendulum-v1",
        seed=79,
        total_timesteps=128,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "buffer_capacity": 512,
            "batch_size": 32,
            "learning_starts": 32,
            "train_frequency": 1,
            "hidden_sizes": (32, 32),
            "tau": 0.005,
            "policy_noise": 0.2,
            "noise_clip": 0.5,
            "policy_delay": 2,
        },
    )

    result = train_td3(config, run_suffix="smoke")

    assert result.run_dir.exists()
    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()
    assert result.metrics["global_step"] >= 128
    assert "eval_return_mean" in result.metrics


def test_train_td3_uses_exploration_noise(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_actor(self, obs):  # type: ignore[no-untyped-def]
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        return torch.zeros((int(obs_tensor.shape[0]), int(self.action_dim)), dtype=torch.float32, device=obs_tensor.device)

    monkeypatch.setattr(td3_trainer.MLPTD3Model, "actor", fake_actor)

    original_scale_actions = td3_trainer._scale_actions

    def spy_scale_actions(normalized_actions: torch.Tensor, *, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        captured["normalized_actions"] = normalized_actions.detach().cpu().clone()
        return original_scale_actions(normalized_actions, low=low, high=high)

    monkeypatch.setattr(td3_trainer, "_scale_actions", spy_scale_actions)

    def fake_randn_like(tensor: torch.Tensor, *args, **kwargs) -> torch.Tensor:  # type: ignore[no-untyped-def]
        del args, kwargs
        return torch.ones_like(tensor)

    monkeypatch.setattr(torch, "randn_like", fake_randn_like)

    config = TrainConfig(
        algo="td3",
        env_id="Pendulum-v1",
        seed=83,
        total_timesteps=1,
        output_dir=tmp_path,
        eval_episodes=0,
        algo_kwargs={
            "buffer_capacity": 128,
            "batch_size": 32,
            "learning_starts": 32,
            "train_frequency": 1,
            "hidden_sizes": (32, 32),
            "tau": 0.005,
            "exploration_noise": 0.1,
            "policy_noise": 0.2,
            "noise_clip": 0.5,
            "policy_delay": 2,
        },
    )

    train_td3(config, run_suffix="exploration-noise-smoke")

    assert "normalized_actions" in captured
    normalized_actions = captured["normalized_actions"]
    assert isinstance(normalized_actions, torch.Tensor)
    assert normalized_actions.shape == (1, 1)
    assert float(normalized_actions[0, 0].item()) == pytest.approx(0.1)
