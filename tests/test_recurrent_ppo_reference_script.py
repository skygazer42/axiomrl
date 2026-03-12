from pathlib import Path
from types import SimpleNamespace

import rl_training.examples.recurrent_ppo_breakout_atari_reference as recurrent_reference


def test_recurrent_ppo_reference_script_builds_atari_config(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class FakeRecurrentPPO:
        def __init__(self, config) -> None:  # type: ignore[no-untyped-def]
            captured["config"] = config

        def learn(self):  # type: ignore[no-untyped-def]
            return SimpleNamespace(
                run_dir=tmp_path / "run",
                checkpoint_path=tmp_path / "checkpoint.pt",
                metrics={"global_step": 64.0},
            )

    monkeypatch.setattr(recurrent_reference, "RecurrentPPO", FakeRecurrentPPO)

    exit_code = recurrent_reference.main(
        [
            "--total-timesteps",
            "64",
            "--output-dir",
            str(tmp_path / "recurrent-runs"),
            "--num-envs",
            "2",
            "--eval-episodes",
            "1",
        ]
    )

    assert exit_code == 0
    config = captured["config"]
    assert config.algo == "recurrent_ppo"
    assert config.env_id == "ALE/Breakout-v5"
    assert config.num_envs == 2
    assert config.tags == ("atari",)
    assert config.algo_kwargs["sequence_length"] == 8
