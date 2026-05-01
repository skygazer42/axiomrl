from pathlib import Path

from axiomrl.api import (
    ARS,
    AWAC,
    AWR,
    BC,
    BCQ,
    BEAR,
    CRR,
    D4PG,
    EDAC,
    HER,
    MARWIL,
    MOPO,
    NAF,
    PETS,
    RLPD,
    XQL,
    CalQL,
    CrossQ,
    DecisionTransformer,
    OpenAIES,
    ReBRAC,
)
from axiomrl.envs import POINT_GOAL_ENV_ID
from axiomrl.experiment.config import TrainConfig


def test_bc_public_api_supports_learn_save_load_and_evaluate(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="bc",
        env_id="Pendulum-v1",
        seed=49,
        total_timesteps=16,
        output_dir=tmp_path / "bc-runs",
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 64,
            "dataset_seed": 17,
            "batch_size": 16,
            "hidden_sizes": (16, 16),
            "eval_interval": 8,
        },
    )

    algo = BC(config)
    result = algo.learn()
    exported = algo.save(tmp_path / "exports" / "bc.pt")
    loaded = BC.load(exported)
    action = loaded.predict([0.0, 0.0, 0.0])
    metrics = loaded.evaluate(num_episodes=1)

    assert result.checkpoint_path is not None
    assert exported.exists()
    assert len(action) == 1
    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_ars_public_api_supports_learn_save_load_and_evaluate(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="ars",
        env_id="Pendulum-v1",
        seed=48,
        total_timesteps=100,
        output_dir=tmp_path / "ars-runs",
        eval_episodes=1,
        algo_kwargs={
            "hidden_sizes": (32, 32),
            "step_size": 0.02,
            "noise_std": 0.03,
            "num_directions": 2,
            "num_top_directions": 2,
        },
        env_kwargs={
            "max_episode_steps": 25,
        },
    )

    algo = ARS(config)
    result = algo.learn()
    exported = algo.save(tmp_path / "exports" / "ars.pt")
    loaded = ARS.load(exported)
    action = loaded.predict([0.0, 0.0, 0.0])
    metrics = loaded.evaluate(num_episodes=1)

    assert result.checkpoint_path is not None
    assert exported.exists()
    assert len(action) == 1
    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_openai_es_public_api_supports_learn_save_load_and_evaluate(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="openai_es",
        env_id="Pendulum-v1",
        seed=49,
        total_timesteps=100,
        output_dir=tmp_path / "openai-es-runs",
        eval_episodes=1,
        algo_kwargs={
            "hidden_sizes": (32, 32),
            "step_size": 0.02,
            "noise_std": 0.03,
            "num_directions": 2,
        },
        env_kwargs={
            "max_episode_steps": 25,
        },
    )

    algo = OpenAIES(config)
    result = algo.learn()
    exported = algo.save(tmp_path / "exports" / "openai_es.pt")
    loaded = OpenAIES.load(exported)
    action = loaded.predict([0.0, 0.0, 0.0])
    metrics = loaded.evaluate(num_episodes=1)

    assert result.checkpoint_path is not None
    assert exported.exists()
    assert len(action) == 1
    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_decision_transformer_public_api_supports_learn_save_load_and_evaluate(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="decision_transformer",
        env_id="Pendulum-v1",
        seed=50,
        total_timesteps=16,
        output_dir=tmp_path / "decision-transformer-runs",
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 64,
            "dataset_seed": 17,
            "batch_size": 8,
            "context_length": 4,
            "hidden_size": 32,
            "num_layers": 1,
            "num_heads": 2,
            "dropout": 0.0,
            "learning_rate": 1e-4,
            "gamma": 0.99,
            "target_return": 0.0,
            "max_timestep": 64,
        },
    )

    algo = DecisionTransformer(config)
    result = algo.learn()
    exported = algo.save(tmp_path / "exports" / "decision_transformer.pt")
    loaded = DecisionTransformer.load(exported)
    action = loaded.predict([0.0, 0.0, 0.0])
    metrics = loaded.evaluate(num_episodes=1)

    assert result.checkpoint_path is not None
    assert exported.exists()
    assert len(action) == 1
    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_mopo_public_api_supports_learn_save_load_and_evaluate(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="mopo",
        env_id="Pendulum-v1",
        seed=51,
        total_timesteps=16,
        output_dir=tmp_path / "mopo-runs",
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 64,
            "dataset_seed": 17,
            "batch_size": 8,
            "hidden_sizes": (32, 32),
            "model_hidden_sizes": (32, 32),
            "num_ensembles": 3,
            "model_batch_size": 16,
            "model_updates": 4,
            "rollout_batch_size": 16,
            "rollout_horizon": 2,
            "rollout_refresh_interval": 4,
            "synthetic_buffer_capacity": 256,
            "synthetic_batch_ratio": 0.5,
            "policy_learning_rate": 1e-4,
            "model_learning_rate": 1e-3,
            "gamma": 0.99,
            "alpha": 0.2,
            "tau": 0.005,
            "penalty_coef": 1.0,
        },
    )

    algo = MOPO(config)
    result = algo.learn()
    exported = algo.save(tmp_path / "exports" / "mopo.pt")
    loaded = MOPO.load(exported)
    action = loaded.predict([0.0, 0.0, 0.0])
    metrics = loaded.evaluate(num_episodes=1)

    assert result.checkpoint_path is not None
    assert exported.exists()
    assert len(action) == 1
    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_pets_public_api_supports_learn_save_load_and_evaluate(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="pets",
        env_id="Pendulum-v1",
        seed=52,
        total_timesteps=64,
        output_dir=tmp_path / "pets-runs",
        eval_episodes=1,
        algo_kwargs={
            "buffer_capacity": 512,
            "batch_size": 16,
            "learning_starts": 8,
            "train_frequency": 1,
            "model_hidden_sizes": (32, 32),
            "model_learning_rate": 1e-3,
            "num_ensembles": 3,
            "model_updates_per_step": 1,
            "planning_horizon": 3,
            "planning_candidates": 64,
            "planning_topk": 8,
            "planning_iterations": 2,
            "planning_particles": 4,
            "initial_random_steps": 8,
        },
        env_kwargs={
            "max_episode_steps": 25,
        },
    )

    algo = PETS(config)
    result = algo.learn()
    exported = algo.save(tmp_path / "exports" / "pets.pt")
    loaded = PETS.load(exported)
    action = loaded.predict([0.0, 0.0, 0.0])
    metrics = loaded.evaluate(num_episodes=1)

    assert result.checkpoint_path is not None
    assert exported.exists()
    assert len(action) == 1
    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_awr_public_api_supports_learn_save_load_and_evaluate(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="awr",
        env_id="Pendulum-v1",
        seed=50,
        total_timesteps=16,
        output_dir=tmp_path / "awr-runs",
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 64,
            "dataset_seed": 18,
            "batch_size": 16,
            "hidden_sizes": (16, 16),
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "beta": 1.0,
            "max_weight": 20.0,
            "eval_interval": 8,
        },
    )

    algo = AWR(config)
    result = algo.learn()
    exported = algo.save(tmp_path / "exports" / "awr.pt")
    loaded = AWR.load(exported)
    action = loaded.predict([0.0, 0.0, 0.0])
    metrics = loaded.evaluate(num_episodes=1)

    assert result.checkpoint_path is not None
    assert exported.exists()
    assert len(action) == 1
    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_marwil_public_api_supports_learn_save_load_and_evaluate(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="marwil",
        env_id="Pendulum-v1",
        seed=50,
        total_timesteps=16,
        output_dir=tmp_path / "marwil-runs",
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 64,
            "dataset_seed": 18,
            "batch_size": 16,
            "hidden_sizes": (16, 16),
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "beta": 1.0,
            "vf_coeff": 1.0,
            "moving_average_sqd_adv_norm_start": 100.0,
            "moving_average_sqd_adv_norm_update_rate": 0.05,
            "eval_interval": 8,
        },
    )

    algo = MARWIL(config)
    result = algo.learn()
    exported = algo.save(tmp_path / "exports" / "marwil.pt")
    loaded = MARWIL.load(exported)
    action = loaded.predict([0.0, 0.0, 0.0])
    metrics = loaded.evaluate(num_episodes=1)

    assert result.checkpoint_path is not None
    assert exported.exists()
    assert len(action) == 1
    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_naf_public_api_supports_learn_save_load_and_evaluate(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="naf",
        env_id="Pendulum-v1",
        seed=58,
        total_timesteps=32,
        output_dir=tmp_path / "naf-runs",
        eval_episodes=1,
        algo_kwargs={
            "buffer_capacity": 256,
            "batch_size": 16,
            "learning_starts": 16,
            "train_frequency": 1,
            "hidden_sizes": (16, 16),
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "tau": 0.005,
            "exploration_noise": 0.1,
            "eval_interval": 16,
        },
    )

    algo = NAF(config)
    result = algo.learn()
    exported = algo.save(tmp_path / "exports" / "naf.pt")
    loaded = NAF.load(exported)
    action = loaded.predict([0.0, 0.0, 0.0])
    metrics = loaded.evaluate(num_episodes=1)

    assert result.checkpoint_path is not None
    assert exported.exists()
    assert len(action) == 1
    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_d4pg_public_api_supports_learn_save_load_and_evaluate(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="d4pg",
        env_id="Pendulum-v1",
        seed=59,
        total_timesteps=32,
        output_dir=tmp_path / "d4pg-runs",
        eval_episodes=1,
        algo_kwargs={
            "buffer_capacity": 256,
            "batch_size": 16,
            "learning_starts": 16,
            "train_frequency": 1,
            "hidden_sizes": (16, 16),
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "tau": 0.005,
            "exploration_noise": 0.1,
            "v_min": -50.0,
            "v_max": 10.0,
            "num_atoms": 21,
            "eval_interval": 16,
        },
    )

    algo = D4PG(config)
    result = algo.learn()
    exported = algo.save(tmp_path / "exports" / "d4pg.pt")
    loaded = D4PG.load(exported)
    action = loaded.predict([0.0, 0.0, 0.0])
    metrics = loaded.evaluate(num_episodes=1)

    assert result.checkpoint_path is not None
    assert exported.exists()
    assert len(action) == 1
    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_awac_public_api_supports_learn_save_load_and_evaluate(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="awac",
        env_id="Pendulum-v1",
        seed=51,
        total_timesteps=16,
        output_dir=tmp_path / "awac-runs",
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 64,
            "dataset_seed": 23,
            "batch_size": 16,
            "hidden_sizes": (16, 16),
            "eval_interval": 8,
        },
    )

    algo = AWAC(config)
    result = algo.learn()
    exported = algo.save(tmp_path / "exports" / "awac.pt")
    loaded = AWAC.load(exported)
    action = loaded.predict([0.0, 0.0, 0.0])
    metrics = loaded.evaluate(num_episodes=1)

    assert result.checkpoint_path is not None
    assert exported.exists()
    assert len(action) == 1
    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_crr_public_api_supports_learn_save_load_and_evaluate(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="crr",
        env_id="Pendulum-v1",
        seed=52,
        total_timesteps=16,
        output_dir=tmp_path / "crr-runs",
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 64,
            "dataset_seed": 24,
            "batch_size": 16,
            "hidden_sizes": (16, 16),
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "tau": 0.005,
            "beta": 1.0,
            "n_action_samples": 4,
            "max_weight": 20.0,
            "advantage_type": "mean",
            "weight_type": "exp",
            "eval_interval": 8,
        },
    )

    algo = CRR(config)
    result = algo.learn()
    exported = algo.save(tmp_path / "exports" / "crr.pt")
    loaded = CRR.load(exported)
    action = loaded.predict([0.0, 0.0, 0.0])
    metrics = loaded.evaluate(num_episodes=1)

    assert result.checkpoint_path is not None
    assert exported.exists()
    assert len(action) == 1
    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_rebrac_public_api_supports_learn_save_load_and_evaluate(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="rebrac",
        env_id="Pendulum-v1",
        seed=53,
        total_timesteps=16,
        output_dir=tmp_path / "rebrac-runs",
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 64,
            "dataset_seed": 25,
            "batch_size": 16,
            "hidden_sizes": (16, 16),
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "tau": 0.005,
            "policy_noise": 0.2,
            "noise_clip": 0.5,
            "policy_delay": 2,
            "actor_bc_weight": 1.0,
            "critic_bc_weight": 1.0,
            "actor_q_weight": 1.0,
            "eval_interval": 8,
        },
    )

    algo = ReBRAC(config)
    result = algo.learn()
    exported = algo.save(tmp_path / "exports" / "rebrac.pt")
    loaded = ReBRAC.load(exported)
    action = loaded.predict([0.0, 0.0, 0.0])
    metrics = loaded.evaluate(num_episodes=1)

    assert result.checkpoint_path is not None
    assert exported.exists()
    assert len(action) == 1
    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_cal_ql_public_api_supports_learn_save_load_and_evaluate(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="cal_ql",
        env_id="Pendulum-v1",
        seed=54,
        total_timesteps=16,
        output_dir=tmp_path / "cal-ql-runs",
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 64,
            "dataset_seed": 26,
            "batch_size": 16,
            "hidden_sizes": (16, 16),
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "alpha": 0.2,
            "tau": 0.005,
            "cql_alpha": 5.0,
            "num_cql_samples": 10,
            "eval_interval": 8,
        },
    )

    algo = CalQL(config)
    result = algo.learn()
    exported = algo.save(tmp_path / "exports" / "cal_ql.pt")
    loaded = CalQL.load(exported)
    action = loaded.predict([0.0, 0.0, 0.0])
    metrics = loaded.evaluate(num_episodes=1)

    assert result.checkpoint_path is not None
    assert exported.exists()
    assert len(action) == 1
    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_edac_public_api_supports_learn_save_load_and_evaluate(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="edac",
        env_id="Pendulum-v1",
        seed=55,
        total_timesteps=16,
        output_dir=tmp_path / "edac-runs",
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 64,
            "dataset_seed": 27,
            "batch_size": 16,
            "hidden_sizes": (16, 16),
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "alpha": 0.2,
            "tau": 0.005,
            "num_critics": 4,
            "eta": 1.0,
            "eval_interval": 8,
        },
    )

    algo = EDAC(config)
    result = algo.learn()
    exported = algo.save(tmp_path / "exports" / "edac.pt")
    loaded = EDAC.load(exported)
    action = loaded.predict([0.0, 0.0, 0.0])
    metrics = loaded.evaluate(num_episodes=1)

    assert result.checkpoint_path is not None
    assert exported.exists()
    assert len(action) == 1
    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_rlpd_public_api_supports_learn_save_load_and_evaluate(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="rlpd",
        env_id="Pendulum-v1",
        seed=56,
        total_timesteps=16,
        output_dir=tmp_path / "rlpd-runs",
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 64,
            "dataset_seed": 28,
            "buffer_capacity": 256,
            "batch_size": 16,
            "learning_starts": 8,
            "train_frequency": 1,
            "gradient_updates_per_step": 2,
            "offline_pretrain_updates": 4,
            "offline_batch_ratio": 0.5,
            "hidden_sizes": (16, 16),
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "alpha": 0.2,
            "tau": 0.005,
            "eval_interval": 8,
        },
    )

    algo = RLPD(config)
    result = algo.learn()
    exported = algo.save(tmp_path / "exports" / "rlpd.pt")
    loaded = RLPD.load(exported)
    action = loaded.predict([0.0, 0.0, 0.0])
    metrics = loaded.evaluate(num_episodes=1)

    assert result.checkpoint_path is not None
    assert exported.exists()
    assert len(action) == 1
    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_xql_public_api_supports_learn_save_load_and_evaluate(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="xql",
        env_id="Pendulum-v1",
        seed=55,
        total_timesteps=16,
        output_dir=tmp_path / "xql-runs",
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 64,
            "dataset_seed": 27,
            "batch_size": 16,
            "hidden_sizes": (16, 16),
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "tau": 0.005,
            "beta": 3.0,
            "loss_temperature": 1.0,
            "max_advantage_weight": 100.0,
            "max_value_diff_exp": 5.0,
            "eval_interval": 8,
        },
    )

    algo = XQL(config)
    result = algo.learn()
    exported = algo.save(tmp_path / "exports" / "xql.pt")
    loaded = XQL.load(exported)
    action = loaded.predict([0.0, 0.0, 0.0])
    metrics = loaded.evaluate(num_episodes=1)

    assert result.checkpoint_path is not None
    assert exported.exists()
    assert len(action) == 1
    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_bear_public_api_supports_learn_save_load_and_evaluate(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="bear",
        env_id="Pendulum-v1",
        seed=52,
        total_timesteps=16,
        output_dir=tmp_path / "bear-runs",
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 64,
            "dataset_seed": 25,
            "batch_size": 16,
            "hidden_sizes": (16, 16),
            "latent_dim": 2,
            "mmd_sigma": 20.0,
            "mmd_alpha": 10.0,
            "num_mmd_action_samples": 10,
            "eval_interval": 8,
        },
    )

    algo = BEAR(config)
    result = algo.learn()
    exported = algo.save(tmp_path / "exports" / "bear.pt")
    loaded = BEAR.load(exported)
    action = loaded.predict([0.0, 0.0, 0.0])
    metrics = loaded.evaluate(num_episodes=1)

    assert result.checkpoint_path is not None
    assert exported.exists()
    assert len(action) == 1
    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_bcq_public_api_supports_learn_save_load_and_evaluate(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="bcq",
        env_id="Pendulum-v1",
        seed=52,
        total_timesteps=16,
        output_dir=tmp_path / "bcq-runs",
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 64,
            "dataset_seed": 27,
            "batch_size": 16,
            "hidden_sizes": (16, 16),
            "latent_dim": 2,
            "num_action_samples": 10,
            "perturbation_scale": 0.05,
            "eval_interval": 8,
        },
    )

    algo = BCQ(config)
    result = algo.learn()
    exported = algo.save(tmp_path / "exports" / "bcq.pt")
    loaded = BCQ.load(exported)
    action = loaded.predict([0.0, 0.0, 0.0])
    metrics = loaded.evaluate(num_episodes=1)

    assert result.checkpoint_path is not None
    assert exported.exists()
    assert len(action) == 1
    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_crossq_public_api_supports_learn_save_load_and_evaluate(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="crossq",
        env_id="Pendulum-v1",
        seed=53,
        total_timesteps=128,
        output_dir=tmp_path / "crossq-runs",
        eval_episodes=1,
        algo_kwargs={
            "buffer_capacity": 512,
            "batch_size": 32,
            "learning_starts": 32,
            "train_frequency": 1,
            "hidden_sizes": (32, 32),
            "critic_hidden_sizes": (32, 32),
            "learning_rate": 1e-3,
            "gamma": 0.99,
            "alpha": 0.1,
            "policy_delay": 1,
            "adam_beta1": 0.5,
            "bn_momentum": 0.99,
        },
    )

    algo = CrossQ(config)
    result = algo.learn()
    exported = algo.save(tmp_path / "exports" / "crossq.pt")
    loaded = CrossQ.load(exported)
    action = loaded.predict([0.0, 0.0, 0.0])
    metrics = loaded.evaluate(num_episodes=1)

    assert result.checkpoint_path is not None
    assert exported.exists()
    assert len(action) == 1
    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_her_public_api_supports_learn_save_load_and_evaluate(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="her",
        env_id=POINT_GOAL_ENV_ID,
        seed=52,
        total_timesteps=32,
        output_dir=tmp_path / "her-runs",
        eval_episodes=1,
        algo_kwargs={
            "buffer_capacity": 512,
            "batch_size": 16,
            "learning_starts": 8,
            "hidden_sizes": (16, 16),
            "eval_interval": 8,
        },
    )

    algo = HER(config)
    result = algo.learn()
    exported = algo.save(tmp_path / "exports" / "her.pt")
    loaded = HER.load(exported)
    action = loaded.predict(
        {
            "observation": [0.0],
            "achieved_goal": [0.0],
            "desired_goal": [0.5],
        }
    )
    metrics = loaded.evaluate(num_episodes=1)

    assert result.checkpoint_path is not None
    assert exported.exists()
    assert len(action) == 1
    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes", "eval_success_rate"}
