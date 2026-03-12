from pathlib import Path
from types import SimpleNamespace

import rl_training.examples.dqn_breakout_atari_reference as dqn_reference
import rl_training.examples.ppo_breakout_atari_reference as ppo_reference


def test_dqn_breakout_reference_script_builds_atari_config(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_train(config, *args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        captured["config"] = config
        return SimpleNamespace(
            run_dir=tmp_path / "run",
            checkpoint_path=tmp_path / "checkpoint.pt",
            metrics={"global_step": 64.0},
        )

    monkeypatch.setattr(dqn_reference, "train_dqn", fake_train)

    exit_code = dqn_reference.main(
        [
            "--total-timesteps",
            "64",
            "--output-dir",
            str(tmp_path / "dqn-runs"),
            "--eval-episodes",
            "1",
        ]
    )

    assert exit_code == 0
    config = captured["config"]
    assert config.algo == "dqn"
    assert config.env_id == "ALE/Breakout-v5"
    assert config.tags == ("atari",)
    assert config.env_kwargs["wrappers"]["atari"]["frame_stack"] == 4


def test_ppo_breakout_reference_script_builds_atari_config(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_train(config, *args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        captured["config"] = config
        return SimpleNamespace(
            run_dir=tmp_path / "run",
            checkpoint_path=tmp_path / "checkpoint.pt",
            metrics={"global_step": 64.0},
        )

    monkeypatch.setattr(ppo_reference, "train_ppo", fake_train)

    exit_code = ppo_reference.main(
        [
            "--total-timesteps",
            "64",
            "--output-dir",
            str(tmp_path / "ppo-runs"),
            "--num-envs",
            "2",
            "--eval-episodes",
            "1",
        ]
    )

    assert exit_code == 0
    config = captured["config"]
    assert config.algo == "ppo"
    assert config.env_id == "ALE/Breakout-v5"
    assert config.num_envs == 2
    assert config.tags == ("atari",)
    assert config.env_kwargs["wrappers"]["atari"]["channel_first"] is True
