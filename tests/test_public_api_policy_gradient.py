from __future__ import annotations

from pathlib import Path

from rl_training.api import APPO, IMPALA, PPG, PPO, TRPO, DiscreteSAC
from rl_training.contrib import RecurrentPPO
from rl_training.experiment.config import TrainConfig


def test_impala_public_api_supports_learn_save_load_and_evaluate(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="impala",
        env_id="CartPole-v1",
        seed=52,
        total_timesteps=64,
        output_dir=tmp_path / "impala-runs",
        num_envs=2,
        eval_episodes=1,
        algo_kwargs={
            "num_steps": 32,
            "hidden_sizes": (16, 16),
            "learning_rate": 3e-4,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "gamma": 0.99,
            "rho_clip": 1.0,
            "c_clip": 1.0,
            "pg_rho_clip": 1.0,
        },
    )

    algo = IMPALA(config)
    result = algo.learn()
    exported = algo.save(tmp_path / "exports" / "impala.pt")
    loaded = IMPALA.load(exported)
    action = loaded.predict([0.0, 0.0, 0.0, 0.0])
    metrics = loaded.evaluate(num_episodes=1)

    assert result.checkpoint_path is not None
    assert exported.exists()
    assert isinstance(action, int)
    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_appo_public_api_supports_learn_save_load_and_evaluate(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="appo",
        env_id="CartPole-v1",
        seed=53,
        total_timesteps=64,
        output_dir=tmp_path / "appo-runs",
        num_envs=2,
        eval_episodes=1,
        algo_kwargs={
            "num_steps": 32,
            "hidden_sizes": (16, 16),
            "learning_rate": 3e-4,
            "clip_coef": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "gamma": 0.99,
            "rho_clip": 1.0,
            "c_clip": 1.0,
            "pg_rho_clip": 1.0,
        },
    )

    algo = APPO(config)
    result = algo.learn()
    exported = algo.save(tmp_path / "exports" / "appo.pt")
    loaded = APPO.load(exported)
    action = loaded.predict([0.0, 0.0, 0.0, 0.0])
    metrics = loaded.evaluate(num_episodes=1)

    assert result.checkpoint_path is not None
    assert exported.exists()
    assert isinstance(action, int)
    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_ppg_public_api_supports_learn_save_load_and_evaluate(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="ppg",
        env_id="CartPole-v1",
        seed=56,
        total_timesteps=128,
        output_dir=tmp_path / "ppg-runs",
        num_envs=2,
        eval_episodes=1,
        algo_kwargs={
            "num_steps": 32,
            "update_epochs": 1,
            "minibatch_size": 32,
            "hidden_sizes": (32, 32),
            "aux_frequency": 1,
            "aux_epochs": 1,
            "aux_minibatch_size": 32,
            "aux_buffer_rollouts": 2,
        },
    )

    algo = PPG(config)
    result = algo.learn()
    exported = algo.save(tmp_path / "exports" / "ppg.pt")
    loaded = PPG.load(exported)
    action = loaded.predict([0.0, 0.0, 0.0, 0.0])
    metrics = loaded.evaluate(num_episodes=1)

    assert result.checkpoint_path is not None
    assert exported.exists()
    assert isinstance(action, int)
    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_ppo_public_api_supports_learn_save_load_and_evaluate(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="ppo",
        env_id="CartPole-v1",
        seed=53,
        total_timesteps=64,
        output_dir=tmp_path / "ppo-runs",
        num_envs=2,
        eval_episodes=1,
        algo_kwargs={
            "num_steps": 32,
            "update_epochs": 1,
            "minibatch_size": 32,
            "hidden_sizes": (16, 16),
        },
    )

    algo = PPO(config)
    result = algo.learn()
    exported = algo.save(tmp_path / "exports" / "ppo.pt")
    loaded = PPO.load(exported)
    action = loaded.predict([0.0, 0.0, 0.0, 0.0])
    metrics = loaded.evaluate(num_episodes=1)

    assert result.checkpoint_path is not None
    assert exported.exists()
    assert isinstance(action, int)
    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_discrete_sac_public_api_supports_learn_save_load_and_evaluate(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="discrete_sac",
        env_id="CartPole-v1",
        seed=54,
        total_timesteps=128,
        output_dir=tmp_path / "discrete-sac-runs",
        num_envs=2,
        eval_episodes=1,
        algo_kwargs={
            "buffer_capacity": 512,
            "batch_size": 32,
            "learning_starts": 32,
            "train_frequency": 1,
            "hidden_sizes": (32, 32),
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "alpha": 0.2,
            "tau": 0.005,
        },
    )

    algo = DiscreteSAC(config)
    result = algo.learn()
    exported = algo.save(tmp_path / "exports" / "discrete-sac.pt")
    loaded = DiscreteSAC.load(exported)
    action = loaded.predict([0.0, 0.0, 0.0, 0.0])
    metrics = loaded.evaluate(num_episodes=1)

    assert result.checkpoint_path is not None
    assert exported.exists()
    assert isinstance(action, int)
    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_trpo_public_api_supports_learn_save_load_and_evaluate(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="trpo",
        env_id="CartPole-v1",
        seed=54,
        total_timesteps=64,
        output_dir=tmp_path / "trpo-runs",
        num_envs=2,
        eval_episodes=1,
        algo_kwargs={
            "num_steps": 32,
            "hidden_sizes": (16, 16),
            "learning_rate": 1e-3,
            "value_updates": 3,
            "max_kl": 0.01,
            "cg_iterations": 5,
            "cg_damping": 0.1,
            "line_search_steps": 5,
            "line_search_shrink": 0.8,
        },
    )

    algo = TRPO(config)
    result = algo.learn()
    exported = algo.save(tmp_path / "exports" / "trpo.pt")
    loaded = TRPO.load(exported)
    action = loaded.predict([0.0, 0.0, 0.0, 0.0])
    metrics = loaded.evaluate(num_episodes=1)

    assert result.checkpoint_path is not None
    assert exported.exists()
    assert isinstance(action, int)
    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_recurrent_ppo_contrib_public_api_supports_learn_save_load_and_evaluate(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="recurrent_ppo",
        env_id="CartPole-v1",
        seed=67,
        total_timesteps=64,
        output_dir=tmp_path / "recurrent-ppo-runs",
        num_envs=2,
        eval_episodes=1,
        algo_kwargs={
            "num_steps": 32,
            "update_epochs": 1,
            "minibatch_size": 32,
            "sequence_length": 8,
            "sequences_per_batch": 4,
            "learning_rate": 3e-4,
            "clip_coef": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "encoder_hidden_sizes": (32,),
            "head_hidden_sizes": (32,),
            "features_dim": 64,
            "recurrent_hidden_size": 64,
            "recurrent_num_layers": 1,
        },
    )

    algo = RecurrentPPO(config)
    result = algo.learn()
    exported = algo.save(tmp_path / "exports" / "recurrent-ppo.pt")
    loaded = RecurrentPPO.load(exported)
    action = loaded.predict([0.0, 0.0, 0.0, 0.0])
    metrics = loaded.evaluate(num_episodes=1)

    assert result.checkpoint_path is not None
    assert exported.exists()
    assert isinstance(action, int)
    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}
