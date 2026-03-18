from pathlib import Path

from rl_training import __version__
from rl_training import contrib as root_contrib
from rl_training.resources import find_packaged_asset


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_package_exports_version() -> None:
    assert __version__ == "0.1.0"


def test_package_exposes_contrib_namespace() -> None:
    assert root_contrib.RecurrentPPO.__name__ == "RecurrentPPO"
    assert root_contrib.RecurrentPPOAlgorithm.__name__ == "RecurrentPPOAlgorithm"


def test_pyproject_declares_cli_entrypoints_and_package_assets() -> None:
    pyproject_text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'axiomrl = "rl_training.cli:main"' in pyproject_text
    assert 'axiomrl-zoo = "rl_training.zoo_cli:main"' in pyproject_text
    assert 'offline = [' in pyproject_text
    assert '"minari"' in pyproject_text
    assert 'assets/configs/*/*.yaml' in pyproject_text
    assert 'assets/zoo/*/*.yaml' in pyproject_text
    assert 'assets/zoo/README.md' in pyproject_text


def test_readme_describes_core_contrib_and_zoo_workflows() -> None:
    readme_text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "core + contrib + zoo" in readme_text
    assert "axiomrl train --config configs/ppo/cartpole.yaml" in readme_text
    assert "axiomrl train --config configs/ars/pendulum.yaml" in readme_text
    assert "axiomrl train --config configs/openai_es/pendulum.yaml" in readme_text
    assert "axiomrl train --config configs/pets/pendulum.yaml" in readme_text
    assert "axiomrl train --config configs/crr/pendulum.yaml" in readme_text
    assert "axiomrl train --config configs/awr/pendulum.yaml" in readme_text
    assert "axiomrl train --config configs/marwil/pendulum.yaml" in readme_text
    assert "axiomrl train --config configs/cal_ql/pendulum.yaml" in readme_text
    assert "axiomrl train --config configs/edac/pendulum.yaml" in readme_text
    assert "axiomrl train --config configs/rlpd/pendulum.yaml" in readme_text
    assert "axiomrl train --config configs/xql/pendulum.yaml" in readme_text
    assert "axiomrl train --config configs/rebrac/pendulum.yaml" in readme_text
    assert "axiomrl train --config zoo/atari/dqn_breakout.yaml" in readme_text
    assert "axiomrl zoo --format commands" in readme_text
    assert 'pip install -e ".[offline]"' in readme_text
    assert "configs/awac/pendulum.yaml" in readme_text
    assert "configs/bear/pendulum.yaml" in readme_text
    assert "configs/her/point_goal.yaml" in readme_text
    assert "configs/bcq/pendulum.yaml" in readme_text
    assert "configs/cal_ql/pendulum.yaml" in readme_text
    assert "configs/crr/pendulum.yaml" in readme_text
    assert "configs/awr/pendulum.yaml" in readme_text
    assert "configs/marwil/pendulum.yaml" in readme_text
    assert "configs/edac/pendulum.yaml" in readme_text
    assert "configs/rlpd/pendulum.yaml" in readme_text
    assert "configs/xql/pendulum.yaml" in readme_text
    assert "configs/rebrac/pendulum.yaml" in readme_text
    assert "configs/crossq/pendulum.yaml" in readme_text
    assert "configs/decision_transformer/pendulum.yaml" in readme_text
    assert "configs/impala/cartpole.yaml" in readme_text
    assert "configs/appo/cartpole.yaml" in readme_text
    assert "configs/mopo/pendulum.yaml" in readme_text
    assert "configs/pets/pendulum.yaml" in readme_text
    assert "configs/curl/pendulum_pixels.yaml" in readme_text
    assert "configs/drq/pendulum_pixels.yaml" in readme_text
    assert "configs/drqv2/pendulum_pixels.yaml" in readme_text
    assert "configs/ppg/cartpole.yaml" in readme_text
    assert "configs/discrete_sac/cartpole.yaml" in readme_text
    assert "configs/trpo/cartpole.yaml" in readme_text
    assert "render_mode: rgb_array" in readme_text
    assert "dataset_kind: npz" in readme_text
    assert "dataset_kind: minari" in readme_text
    assert "wrappers:" in readme_text
    assert "early_stopping:" in readme_text
    assert "python -m rl_training.examples.dqn_breakout_atari_reference" in readme_text
    assert "python -m rl_training train --config configs/dqn/cartpole.yaml" in readme_text
    assert "from rl_training.contrib import RecurrentPPO" in readme_text


def test_packaged_assets_include_core_configs_and_zoo_manifest() -> None:
    awac_config = find_packaged_asset("configs/awac/pendulum.yaml")
    ars_config = find_packaged_asset("configs/ars/pendulum.yaml")
    openai_es_config = find_packaged_asset("configs/openai_es/pendulum.yaml")
    awr_config = find_packaged_asset("configs/awr/pendulum.yaml")
    marwil_config = find_packaged_asset("configs/marwil/pendulum.yaml")
    bear_config = find_packaged_asset("configs/bear/pendulum.yaml")
    bc_config = find_packaged_asset("configs/bc/pendulum.yaml")
    bcq_config = find_packaged_asset("configs/bcq/pendulum.yaml")
    cal_ql_config = find_packaged_asset("configs/cal_ql/pendulum.yaml")
    crr_config = find_packaged_asset("configs/crr/pendulum.yaml")
    edac_config = find_packaged_asset("configs/edac/pendulum.yaml")
    rlpd_config = find_packaged_asset("configs/rlpd/pendulum.yaml")
    xql_config = find_packaged_asset("configs/xql/pendulum.yaml")
    rebrac_config = find_packaged_asset("configs/rebrac/pendulum.yaml")
    crossq_config = find_packaged_asset("configs/crossq/pendulum.yaml")
    decision_transformer_config = find_packaged_asset("configs/decision_transformer/pendulum.yaml")
    impala_config = find_packaged_asset("configs/impala/cartpole.yaml")
    appo_config = find_packaged_asset("configs/appo/cartpole.yaml")
    mopo_config = find_packaged_asset("configs/mopo/pendulum.yaml")
    pets_config = find_packaged_asset("configs/pets/pendulum.yaml")
    curl_config = find_packaged_asset("configs/curl/pendulum_pixels.yaml")
    drq_config = find_packaged_asset("configs/drq/pendulum_pixels.yaml")
    drqv2_config = find_packaged_asset("configs/drqv2/pendulum_pixels.yaml")
    ppg_config = find_packaged_asset("configs/ppg/cartpole.yaml")
    her_config = find_packaged_asset("configs/her/point_goal.yaml")
    discrete_sac_config = find_packaged_asset("configs/discrete_sac/cartpole.yaml")
    ppo_config = find_packaged_asset("configs/ppo/cartpole.yaml")
    trpo_config = find_packaged_asset("configs/trpo/cartpole.yaml")
    zoo_manifest = find_packaged_asset("zoo/atari/benchmark.yaml")
    zoo_readme = find_packaged_asset("zoo/README.md")

    assert awac_config is not None
    assert ars_config is not None
    assert openai_es_config is not None
    assert awr_config is not None
    assert marwil_config is not None
    assert bear_config is not None
    assert bc_config is not None
    assert bcq_config is not None
    assert cal_ql_config is not None
    assert crr_config is not None
    assert edac_config is not None
    assert rlpd_config is not None
    assert xql_config is not None
    assert rebrac_config is not None
    assert crossq_config is not None
    assert decision_transformer_config is not None
    assert impala_config is not None
    assert appo_config is not None
    assert mopo_config is not None
    assert pets_config is not None
    assert curl_config is not None
    assert drq_config is not None
    assert drqv2_config is not None
    assert ppg_config is not None
    assert her_config is not None
    assert discrete_sac_config is not None
    assert ppo_config is not None
    assert trpo_config is not None
    assert zoo_manifest is not None
    assert zoo_readme is not None
    assert awac_config.exists()
    assert ars_config.exists()
    assert openai_es_config.exists()
    assert awr_config.exists()
    assert marwil_config.exists()
    assert bear_config.exists()
    assert bc_config.exists()
    assert bcq_config.exists()
    assert cal_ql_config.exists()
    assert crr_config.exists()
    assert edac_config.exists()
    assert rlpd_config.exists()
    assert xql_config.exists()
    assert rebrac_config.exists()
    assert crossq_config.exists()
    assert decision_transformer_config.exists()
    assert impala_config.exists()
    assert appo_config.exists()
    assert mopo_config.exists()
    assert pets_config.exists()
    assert curl_config.exists()
    assert drq_config.exists()
    assert drqv2_config.exists()
    assert ppg_config.exists()
    assert her_config.exists()
    assert discrete_sac_config.exists()
    assert ppo_config.exists()
    assert trpo_config.exists()
    assert zoo_manifest.exists()
    assert zoo_readme.exists()


def test_package_includes_atari_reference_example_modules() -> None:
    example_root = REPO_ROOT / "src" / "rl_training" / "examples"

    assert (example_root / "__init__.py").exists()
    assert (example_root / "dqn_breakout_atari_reference.py").exists()
    assert (example_root / "ppo_breakout_atari_reference.py").exists()
    assert (example_root / "recurrent_ppo_breakout_atari_reference.py").exists()


def test_package_includes_module_entrypoint() -> None:
    assert (REPO_ROOT / "src" / "rl_training" / "__main__.py").exists()
