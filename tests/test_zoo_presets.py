from __future__ import annotations

from pathlib import Path

import yaml

from rl_training.cli import load_config
from rl_training.zoo_cli import main as zoo_main


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_atari_benchmark_manifest_points_to_existing_configs() -> None:
    manifest_path = REPO_ROOT / "zoo" / "atari" / "benchmark.yaml"
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

    assert payload["suite"] == "atari"
    presets = payload["presets"]
    assert isinstance(presets, list)
    assert presets

    for preset in presets:
        config_path = REPO_ROOT / preset["config"]
        assert config_path.exists()


def test_each_atari_zoo_preset_resolves_to_a_train_config() -> None:
    preset_paths = sorted((REPO_ROOT / "zoo" / "atari").glob("*.yaml"))
    preset_paths = [path for path in preset_paths if path.name != "benchmark.yaml"]

    assert preset_paths

    expected_algorithms = {"dqn", "ppo", "recurrent_ppo"}
    for preset_path in preset_paths:
        config = load_config(preset_path)
        assert config.algo in expected_algorithms
        assert config.env_id == "ALE/Breakout-v5"
        assert "atari" in config.tags


def test_packaged_zoo_preset_can_be_loaded_outside_repo_root(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    config = load_config("zoo/atari/dqn_breakout.yaml")

    assert config.algo == "dqn"
    assert config.env_id == "ALE/Breakout-v5"
    assert "atari" in config.tags


def test_zoo_cli_can_resolve_packaged_manifest_outside_repo_root(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    exit_code = zoo_main(["--manifest", "zoo/atari/benchmark.yaml", "--format", "commands"])

    assert exit_code == 0
