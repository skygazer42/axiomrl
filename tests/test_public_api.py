from pathlib import Path

import gymnasium as gym
import numpy as np

from rl_training.api import (
    A2C,
    ARS,
    OpenAIES,
    AWR,
    AWAC,
    MARWIL,
    BEAR,
    BC,
    DecisionTransformer,
    BCQ,
    C51DQN,
    CalQL,
    CURL,
    CrossQ,
    CRR,
    D4PG,
    DDPG,
    DRQN,
    R2D2,
    EDAC,
    DrQ,
    DrQv2,
    DiscreteSAC,
    DQN,
    DoubleDQN,
    DuelingDQN,
    CQL,
    HER,
    IQL,
    IQN,
    IMPALA,
    APPO,
    MOPO,
    PETS,
    NAF,
    PPG,
    XQL,
    NoisyDQN,
    NStepDQN,
    PPO,
    PrioritizedDQN,
    QRDQN,
    RainbowDQN,
    REDQ,
    RLPD,
    ReBRAC,
    SAC,
    TD3,
    TD3BC,
    TRPO,
    TQC,
)
from rl_training.contrib import RecurrentPPO
from rl_training.envs import POINT_GOAL_ENV_ID
from rl_training.experiment.config import TrainConfig


class TinyRenderContinuousEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self._step = 0
        self._state = np.zeros(3, dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        del options
        self._step = 0
        self._state.fill(0.0)
        return self._state.copy(), {}

    def step(self, action: np.ndarray):
        action_value = float(np.asarray(action).reshape(-1)[0])
        self._step += 1
        self._state = np.array([action_value, self._step / 4.0, -action_value], dtype=np.float32)
        terminated = self._step >= 4
        truncated = False
        reward = 1.0 - abs(action_value)
        return self._state.copy(), reward, terminated, truncated, {}

    def render(self) -> np.ndarray:
        canvas = np.zeros((96, 96, 3), dtype=np.uint8)
        action_intensity = int(np.clip((self._state[0] + 1.0) * 127.5, 0, 255))
        canvas[..., 0] = np.uint8(self._step * 32)
        canvas[16:80, 16:80, 1] = np.uint8(action_intensity)
        canvas[32:64, 32:64, 2] = np.uint8(255 - action_intensity)
        return canvas


def _register_tiny_render_env() -> str:
    env_id = "RLTrainingTest/PublicAPIDrQv2-v0"
    try:
        gym.spec(env_id)
    except gym.error.Error:
        gym.register(id=env_id, entry_point="tests.support.envs:TinyRenderContinuousEnv")
    return env_id


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


def test_drqn_public_api_supports_learn_save_load_and_evaluate(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="drqn",
        env_id="CartPole-v1",
        seed=60,
        total_timesteps=64,
        output_dir=tmp_path / "drqn-runs",
        num_envs=2,
        eval_episodes=1,
        algo_kwargs={
            "buffer_capacity": 128,
            "batch_size": 4,
            "learning_starts": 4,
            "train_frequency": 1,
            "target_update_interval": 8,
            "hidden_sizes": (16,),
            "head_hidden_sizes": (16,),
            "features_dim": 32,
            "recurrent_hidden_size": 32,
            "recurrent_num_layers": 1,
            "sequence_length": 8,
            "epsilon_start": 1.0,
            "epsilon_end": 0.05,
            "exploration_fraction": 0.2,
            "eval_interval": 16,
        },
    )

    algo = DRQN(config)
    result = algo.learn()
    exported = algo.save(tmp_path / "exports" / "drqn.pt")
    loaded = DRQN.load(exported)
    action = loaded.predict([0.0, 0.0, 0.0, 0.0])
    metrics = loaded.evaluate(num_episodes=1)

    assert result.checkpoint_path is not None
    assert exported.exists()
    assert isinstance(action, int)
    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_r2d2_public_api_supports_learn_save_load_and_evaluate(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="r2d2",
        env_id="CartPole-v1",
        seed=61,
        total_timesteps=64,
        output_dir=tmp_path / "r2d2-runs",
        num_envs=2,
        eval_episodes=1,
        algo_kwargs={
            "buffer_capacity": 128,
            "batch_size": 4,
            "learning_starts": 4,
            "train_frequency": 1,
            "target_update_interval": 8,
            "hidden_sizes": (16,),
            "head_hidden_sizes": (16,),
            "features_dim": 32,
            "recurrent_hidden_size": 32,
            "recurrent_num_layers": 1,
            "sequence_length": 8,
            "epsilon_start": 1.0,
            "epsilon_end": 0.05,
            "exploration_fraction": 0.2,
            "prioritized_alpha": 0.6,
            "prioritized_beta_start": 0.4,
            "prioritized_beta_end": 1.0,
            "prioritized_beta_fraction": 1.0,
            "priority_eta": 0.9,
            "n_step": 3,
            "eval_interval": 16,
        },
    )

    algo = R2D2(config)
    result = algo.learn()
    exported = algo.save(tmp_path / "exports" / "r2d2.pt")
    loaded = R2D2.load(exported)
    action = loaded.predict([0.0, 0.0, 0.0, 0.0])
    metrics = loaded.evaluate(num_episodes=1)

    assert result.checkpoint_path is not None
    assert exported.exists()
    assert isinstance(action, int)
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


def test_drqv2_public_api_supports_learn_save_load_and_evaluate(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="drqv2",
        env_id=_register_tiny_render_env(),
        seed=54,
        total_timesteps=96,
        output_dir=tmp_path / "drqv2-runs",
        eval_episodes=1,
        algo_kwargs={
            "buffer_capacity": 512,
            "batch_size": 32,
            "learning_starts": 32,
            "train_frequency": 1,
            "features_dim": 64,
            "actor_hidden_sizes": (32,),
            "critic_hidden_sizes": (32,),
            "learning_rate": 1e-4,
            "gamma": 0.99,
            "tau": 0.01,
            "policy_delay": 2,
            "augmentation_pad": 4,
            "exploration_noise": 0.1,
            "exploration_noise_clip": 0.3,
        },
        env_kwargs={
            "render_mode": "rgb_array",
            "wrappers": {
                "pixels": {
                    "resize_shape": [84, 84],
                    "frame_stack": 3,
                    "channel_first": True,
                }
            },
        },
    )

    algo = DrQv2(config)
    result = algo.learn()
    exported = algo.save(tmp_path / "exports" / "drqv2.pt")
    loaded = DrQv2.load(exported)
    action = loaded.predict(np.zeros((9, 84, 84), dtype=np.uint8))
    metrics = loaded.evaluate(num_episodes=1)

    assert result.checkpoint_path is not None
    assert exported.exists()
    assert len(action) == 1
    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_drq_public_api_supports_learn_save_load_and_evaluate(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="drq",
        env_id=_register_tiny_render_env(),
        seed=55,
        total_timesteps=96,
        output_dir=tmp_path / "drq-runs",
        eval_episodes=1,
        algo_kwargs={
            "buffer_capacity": 512,
            "batch_size": 32,
            "learning_starts": 32,
            "train_frequency": 1,
            "features_dim": 64,
            "actor_hidden_sizes": (32,),
            "critic_hidden_sizes": (32,),
            "learning_rate": 1e-4,
            "gamma": 0.99,
            "alpha": 0.1,
            "tau": 0.01,
            "augmentation_pad": 4,
        },
        env_kwargs={
            "render_mode": "rgb_array",
            "wrappers": {
                "pixels": {
                    "resize_shape": [84, 84],
                    "frame_stack": 3,
                    "channel_first": True,
                }
            },
        },
    )

    algo = DrQ(config)
    result = algo.learn()
    exported = algo.save(tmp_path / "exports" / "drq.pt")
    loaded = DrQ.load(exported)
    action = loaded.predict(np.zeros((9, 84, 84), dtype=np.uint8))
    metrics = loaded.evaluate(num_episodes=1)

    assert result.checkpoint_path is not None
    assert exported.exists()
    assert len(action) == 1
    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_curl_public_api_supports_learn_save_load_and_evaluate(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="curl",
        env_id=_register_tiny_render_env(),
        seed=57,
        total_timesteps=96,
        output_dir=tmp_path / "curl-runs",
        eval_episodes=1,
        algo_kwargs={
            "buffer_capacity": 512,
            "batch_size": 32,
            "learning_starts": 32,
            "train_frequency": 1,
            "features_dim": 64,
            "actor_hidden_sizes": (32,),
            "critic_hidden_sizes": (32,),
            "projection_dim": 32,
            "learning_rate": 1e-4,
            "gamma": 0.99,
            "alpha": 0.1,
            "tau": 0.01,
            "augmentation_pad": 4,
            "curl_temperature": 0.1,
            "curl_coef": 1.0,
        },
        env_kwargs={
            "render_mode": "rgb_array",
            "wrappers": {
                "pixels": {
                    "resize_shape": [84, 84],
                    "frame_stack": 3,
                    "channel_first": True,
                }
            },
        },
    )

    algo = CURL(config)
    result = algo.learn()
    exported = algo.save(tmp_path / "exports" / "curl.pt")
    loaded = CURL.load(exported)
    action = loaded.predict(np.zeros((9, 84, 84), dtype=np.uint8))
    metrics = loaded.evaluate(num_episodes=1)

    assert result.checkpoint_path is not None
    assert exported.exists()
    assert len(action) == 1
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


def test_off_policy_public_apis_support_learn_and_evaluate(tmp_path: Path) -> None:
    a2c = A2C(
        TrainConfig(
            algo="a2c",
            env_id="CartPole-v1",
            seed=57,
            total_timesteps=64,
            output_dir=tmp_path / "a2c-runs",
            num_envs=2,
            eval_episodes=1,
            algo_kwargs={
                "num_steps": 32,
                "hidden_sizes": (16, 16),
                "learning_rate": 3e-4,
                "ent_coef": 0.01,
                "vf_coef": 0.5,
                "gamma": 0.99,
                "gae_lambda": 0.95,
            },
        )
    )
    dqn = DQN(
        TrainConfig(
            algo="dqn",
            env_id="CartPole-v1",
            seed=59,
            total_timesteps=96,
            output_dir=tmp_path / "dqn-runs",
            eval_episodes=1,
            algo_kwargs={
                "buffer_capacity": 256,
                "batch_size": 32,
                "learning_starts": 32,
                "train_frequency": 1,
                "target_update_interval": 16,
                "hidden_sizes": (16, 16),
            },
        )
    )
    double_dqn = DoubleDQN(
        TrainConfig(
            algo="double_dqn",
            env_id="CartPole-v1",
            seed=60,
            total_timesteps=96,
            output_dir=tmp_path / "double-dqn-runs",
            eval_episodes=1,
            algo_kwargs={
                "buffer_capacity": 256,
                "batch_size": 32,
                "learning_starts": 32,
                "train_frequency": 1,
                "target_update_interval": 16,
                "hidden_sizes": (16, 16),
            },
        )
    )
    dueling_dqn = DuelingDQN(
        TrainConfig(
            algo="dueling_dqn",
            env_id="CartPole-v1",
            seed=60,
            total_timesteps=96,
            output_dir=tmp_path / "dueling-dqn-runs",
            eval_episodes=1,
            algo_kwargs={
                "buffer_capacity": 256,
                "batch_size": 32,
                "learning_starts": 32,
                "train_frequency": 1,
                "target_update_interval": 16,
                "hidden_sizes": (16, 16),
            },
        )
    )
    noisy_dqn = NoisyDQN(
        TrainConfig(
            algo="noisy_dqn",
            env_id="CartPole-v1",
            seed=60,
            total_timesteps=96,
            output_dir=tmp_path / "noisy-dqn-runs",
            eval_episodes=1,
            algo_kwargs={
                "buffer_capacity": 256,
                "batch_size": 32,
                "learning_starts": 32,
                "train_frequency": 1,
                "target_update_interval": 16,
                "hidden_sizes": (16, 16),
            },
        )
    )
    prioritized_dqn = PrioritizedDQN(
        TrainConfig(
            algo="prioritized_dqn",
            env_id="CartPole-v1",
            seed=60,
            total_timesteps=96,
            output_dir=tmp_path / "prioritized-dqn-runs",
            eval_episodes=1,
            algo_kwargs={
                "buffer_capacity": 256,
                "batch_size": 32,
                "learning_starts": 32,
                "train_frequency": 1,
                "target_update_interval": 16,
                "hidden_sizes": (16, 16),
                "prioritized_alpha": 0.6,
                "prioritized_beta_start": 0.4,
            },
        )
    )
    rainbow_dqn = RainbowDQN(
        TrainConfig(
            algo="rainbow_dqn",
            env_id="CartPole-v1",
            seed=60,
            total_timesteps=96,
            output_dir=tmp_path / "rainbow-dqn-runs",
            eval_episodes=1,
            algo_kwargs={
                "buffer_capacity": 256,
                "batch_size": 32,
                "learning_starts": 32,
                "train_frequency": 1,
                "target_update_interval": 16,
                "hidden_sizes": (16, 16),
                "epsilon_start": 0.0,
                "epsilon_end": 0.0,
                "exploration_fraction": 0.0,
                "prioritized_alpha": 0.6,
                "prioritized_beta_start": 0.4,
            },
        )
    )
    c51_dqn = C51DQN(
        TrainConfig(
            algo="c51_dqn",
            env_id="CartPole-v1",
            seed=60,
            total_timesteps=96,
            output_dir=tmp_path / "c51-dqn-runs",
            eval_episodes=1,
            algo_kwargs={
                "buffer_capacity": 256,
                "batch_size": 32,
                "learning_starts": 32,
                "train_frequency": 1,
                "target_update_interval": 16,
                "hidden_sizes": (16, 16),
                "v_min": 0.0,
                "v_max": 200.0,
                "num_atoms": 51,
            },
        )
    )
    n_step_dqn = NStepDQN(
        TrainConfig(
            algo="n_step_dqn",
            env_id="CartPole-v1",
            seed=60,
            total_timesteps=96,
            output_dir=tmp_path / "n-step-dqn-runs",
            eval_episodes=1,
            algo_kwargs={
                "n_step": 3,
                "gamma": 0.99,
                "buffer_capacity": 256,
                "batch_size": 32,
                "learning_starts": 32,
                "train_frequency": 1,
                "target_update_interval": 16,
                "hidden_sizes": (16, 16),
            },
        )
    )
    qr_dqn = QRDQN(
        TrainConfig(
            algo="qr_dqn",
            env_id="CartPole-v1",
            seed=60,
            total_timesteps=96,
            output_dir=tmp_path / "qr-dqn-runs",
            eval_episodes=1,
            algo_kwargs={
                "buffer_capacity": 256,
                "batch_size": 32,
                "learning_starts": 32,
                "train_frequency": 1,
                "target_update_interval": 16,
                "hidden_sizes": (16, 16),
                "gamma": 0.99,
                "num_quantiles": 51,
                "kappa": 1.0,
            },
        )
    )
    iqn = IQN(
        TrainConfig(
            algo="iqn",
            env_id="CartPole-v1",
            seed=60,
            total_timesteps=96,
            output_dir=tmp_path / "iqn-runs",
            eval_episodes=1,
            algo_kwargs={
                "buffer_capacity": 256,
                "batch_size": 32,
                "learning_starts": 32,
                "train_frequency": 1,
                "target_update_interval": 16,
                "hidden_sizes": (16, 16),
                "gamma": 0.99,
                "num_quantiles": 16,
                "embedding_dim": 32,
                "kappa": 1.0,
            },
        )
    )
    sac = SAC(
        TrainConfig(
            algo="sac",
            env_id="Pendulum-v1",
            seed=61,
            total_timesteps=96,
            output_dir=tmp_path / "sac-runs",
            eval_episodes=1,
            algo_kwargs={
                "buffer_capacity": 512,
                "batch_size": 32,
                "learning_starts": 32,
                "train_frequency": 1,
                "hidden_sizes": (32, 32),
                "alpha": 0.2,
                "tau": 0.005,
            },
        )
    )
    cql = CQL(
        TrainConfig(
            algo="cql",
            env_id="Pendulum-v1",
            seed=62,
            total_timesteps=96,
            output_dir=tmp_path / "cql-runs",
            eval_episodes=1,
            algo_kwargs={
                "dataset_kind": "random",
                "dataset_size": 192,
                "dataset_seed": 17,
                "batch_size": 32,
                "hidden_sizes": (32, 32),
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "alpha": 0.2,
                "tau": 0.005,
                "cql_alpha": 5.0,
                "num_cql_samples": 10,
            },
        )
    )
    cal_ql = CalQL(
        TrainConfig(
            algo="cal_ql",
            env_id="Pendulum-v1",
            seed=63,
            total_timesteps=96,
            output_dir=tmp_path / "cal-ql-runs",
            eval_episodes=1,
            algo_kwargs={
                "dataset_kind": "random",
                "dataset_size": 192,
                "dataset_seed": 18,
                "batch_size": 32,
                "hidden_sizes": (32, 32),
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "alpha": 0.2,
                "tau": 0.005,
                "cql_alpha": 5.0,
                "num_cql_samples": 10,
            },
        )
    )
    edac = EDAC(
        TrainConfig(
            algo="edac",
            env_id="Pendulum-v1",
            seed=64,
            total_timesteps=96,
            output_dir=tmp_path / "edac-runs",
            eval_episodes=1,
            algo_kwargs={
                "dataset_kind": "random",
                "dataset_size": 192,
                "dataset_seed": 19,
                "batch_size": 32,
                "hidden_sizes": (32, 32),
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "alpha": 0.2,
                "tau": 0.005,
                "num_critics": 4,
                "eta": 1.0,
            },
        )
    )
    awr = AWR(
        TrainConfig(
            algo="awr",
            env_id="Pendulum-v1",
            seed=65,
            total_timesteps=96,
            output_dir=tmp_path / "awr-runs",
            eval_episodes=1,
            algo_kwargs={
                "dataset_kind": "random",
                "dataset_size": 192,
                "dataset_seed": 20,
                "batch_size": 32,
                "hidden_sizes": (32, 32),
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "beta": 1.0,
                "max_weight": 20.0,
            },
        )
    )
    marwil = MARWIL(
        TrainConfig(
            algo="marwil",
            env_id="Pendulum-v1",
            seed=66,
            total_timesteps=96,
            output_dir=tmp_path / "marwil-runs",
            eval_episodes=1,
            algo_kwargs={
                "dataset_kind": "random",
                "dataset_size": 192,
                "dataset_seed": 21,
                "batch_size": 32,
                "hidden_sizes": (32, 32),
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "beta": 1.0,
                "vf_coeff": 1.0,
                "moving_average_sqd_adv_norm_start": 100.0,
                "moving_average_sqd_adv_norm_update_rate": 0.05,
            },
        )
    )
    rlpd = RLPD(
        TrainConfig(
            algo="rlpd",
            env_id="Pendulum-v1",
            seed=67,
            total_timesteps=96,
            output_dir=tmp_path / "rlpd-runs",
            eval_episodes=1,
            algo_kwargs={
                "dataset_kind": "random",
                "dataset_size": 192,
                "dataset_seed": 22,
                "buffer_capacity": 512,
                "batch_size": 32,
                "learning_starts": 32,
                "train_frequency": 1,
                "gradient_updates_per_step": 2,
                "offline_pretrain_updates": 8,
                "offline_batch_ratio": 0.5,
                "hidden_sizes": (32, 32),
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "alpha": 0.2,
                "tau": 0.005,
            },
        )
    )
    tqc = TQC(
        TrainConfig(
            algo="tqc",
            env_id="Pendulum-v1",
            seed=62,
            total_timesteps=96,
            output_dir=tmp_path / "tqc-runs",
            eval_episodes=1,
            algo_kwargs={
                "buffer_capacity": 512,
                "batch_size": 32,
                "learning_starts": 32,
                "train_frequency": 1,
                "hidden_sizes": (32, 32),
                "alpha": 0.2,
                "tau": 0.005,
                "num_critics": 3,
                "num_quantiles": 7,
                "top_quantiles_to_drop_per_net": 1,
                "kappa": 1.0,
            },
        )
    )
    redq = REDQ(
        TrainConfig(
            algo="redq",
            env_id="Pendulum-v1",
            seed=63,
            total_timesteps=96,
            output_dir=tmp_path / "redq-runs",
            eval_episodes=1,
            algo_kwargs={
                "buffer_capacity": 512,
                "batch_size": 32,
                "learning_starts": 32,
                "train_frequency": 1,
                "gradient_updates_per_step": 2,
                "hidden_sizes": (32, 32),
                "alpha": 0.2,
                "tau": 0.005,
                "num_critics": 5,
                "subset_size": 2,
            },
        )
    )
    iql = IQL(
        TrainConfig(
            algo="iql",
            env_id="Pendulum-v1",
            seed=64,
            total_timesteps=96,
            output_dir=tmp_path / "iql-runs",
            eval_episodes=1,
            algo_kwargs={
                "dataset_kind": "random",
                "dataset_size": 192,
                "dataset_seed": 21,
                "batch_size": 32,
                "hidden_sizes": (32, 32),
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "tau": 0.005,
                "expectile": 0.7,
                "beta": 3.0,
                "max_advantage_weight": 100.0,
            },
        )
    )
    xql = XQL(
        TrainConfig(
            algo="xql",
            env_id="Pendulum-v1",
            seed=65,
            total_timesteps=96,
            output_dir=tmp_path / "xql-runs",
            eval_episodes=1,
            algo_kwargs={
                "dataset_kind": "random",
                "dataset_size": 192,
                "dataset_seed": 22,
                "batch_size": 32,
                "hidden_sizes": (32, 32),
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "tau": 0.005,
                "beta": 3.0,
                "loss_temperature": 1.0,
                "max_advantage_weight": 100.0,
                "max_value_diff_exp": 5.0,
            },
        )
    )
    td3_bc = TD3BC(
        TrainConfig(
            algo="td3_bc",
            env_id="Pendulum-v1",
            seed=65,
            total_timesteps=96,
            output_dir=tmp_path / "td3-bc-runs",
            eval_episodes=1,
            algo_kwargs={
                "dataset_kind": "random",
                "dataset_size": 192,
                "dataset_seed": 23,
                "batch_size": 32,
                "hidden_sizes": (32, 32),
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "tau": 0.005,
                "policy_noise": 0.2,
                "noise_clip": 0.5,
                "policy_delay": 2,
                "bc_alpha": 2.5,
            },
        )
    )
    ddpg = DDPG(
        TrainConfig(
            algo="ddpg",
            env_id="Pendulum-v1",
            seed=62,
            total_timesteps=96,
            output_dir=tmp_path / "ddpg-runs",
            eval_episodes=1,
            algo_kwargs={
                "buffer_capacity": 512,
                "batch_size": 32,
                "learning_starts": 32,
                "train_frequency": 1,
                "hidden_sizes": (32, 32),
                "tau": 0.005,
            },
        )
    )
    td3 = TD3(
        TrainConfig(
            algo="td3",
            env_id="Pendulum-v1",
            seed=63,
            total_timesteps=96,
            output_dir=tmp_path / "td3-runs",
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
    )

    a2c.learn()
    dqn.learn()
    double_dqn.learn()
    dueling_dqn.learn()
    noisy_dqn.learn()
    prioritized_dqn.learn()
    rainbow_dqn.learn()
    c51_dqn.learn()
    n_step_dqn.learn()
    qr_dqn.learn()
    iqn.learn()
    sac.learn()
    cql.learn()
    cal_ql.learn()
    edac.learn()
    awr.learn()
    marwil.learn()
    rlpd.learn()
    tqc.learn()
    redq.learn()
    iql.learn()
    xql.learn()
    td3_bc.learn()
    ddpg.learn()
    td3.learn()

    a2c_action = a2c.predict([0.0, 0.0, 0.0, 0.0])
    dqn_action = dqn.predict([0.0, 0.0, 0.0, 0.0])
    double_dqn_action = double_dqn.predict([0.0, 0.0, 0.0, 0.0])
    dueling_dqn_action = dueling_dqn.predict([0.0, 0.0, 0.0, 0.0])
    noisy_dqn_action = noisy_dqn.predict([0.0, 0.0, 0.0, 0.0])
    prioritized_dqn_action = prioritized_dqn.predict([0.0, 0.0, 0.0, 0.0])
    rainbow_dqn_action = rainbow_dqn.predict([0.0, 0.0, 0.0, 0.0])
    c51_dqn_action = c51_dqn.predict([0.0, 0.0, 0.0, 0.0])
    n_step_dqn_action = n_step_dqn.predict([0.0, 0.0, 0.0, 0.0])
    qr_dqn_action = qr_dqn.predict([0.0, 0.0, 0.0, 0.0])
    iqn_action = iqn.predict([0.0, 0.0, 0.0, 0.0])
    sac_action = sac.predict([0.0, 0.0, 0.0])
    cql_action = cql.predict([0.0, 0.0, 0.0])
    cal_ql_action = cal_ql.predict([0.0, 0.0, 0.0])
    edac_action = edac.predict([0.0, 0.0, 0.0])
    awr_action = awr.predict([0.0, 0.0, 0.0])
    marwil_action = marwil.predict([0.0, 0.0, 0.0])
    rlpd_action = rlpd.predict([0.0, 0.0, 0.0])
    tqc_action = tqc.predict([0.0, 0.0, 0.0])
    redq_action = redq.predict([0.0, 0.0, 0.0])
    iql_action = iql.predict([0.0, 0.0, 0.0])
    xql_action = xql.predict([0.0, 0.0, 0.0])
    td3_bc_action = td3_bc.predict([0.0, 0.0, 0.0])
    ddpg_action = ddpg.predict([0.0, 0.0, 0.0])
    td3_action = td3.predict([0.0, 0.0, 0.0])

    assert isinstance(a2c_action, int)
    assert isinstance(dqn_action, int)
    assert isinstance(double_dqn_action, int)
    assert isinstance(dueling_dqn_action, int)
    assert isinstance(noisy_dqn_action, int)
    assert isinstance(prioritized_dqn_action, int)
    assert isinstance(rainbow_dqn_action, int)
    assert isinstance(c51_dqn_action, int)
    assert isinstance(n_step_dqn_action, int)
    assert isinstance(qr_dqn_action, int)
    assert isinstance(iqn_action, int)
    assert len(sac_action) == 1
    assert len(cql_action) == 1
    assert len(cal_ql_action) == 1
    assert len(edac_action) == 1
    assert len(awr_action) == 1
    assert len(marwil_action) == 1
    assert len(rlpd_action) == 1
    assert len(tqc_action) == 1
    assert len(redq_action) == 1
    assert len(iql_action) == 1
    assert len(xql_action) == 1
    assert len(td3_bc_action) == 1
    assert len(ddpg_action) == 1
    assert len(td3_action) == 1
    assert "eval_return_mean" in a2c.evaluate(num_episodes=1)
    assert "eval_return_mean" in dqn.evaluate(num_episodes=1)
    assert "eval_return_mean" in double_dqn.evaluate(num_episodes=1)
    assert "eval_return_mean" in dueling_dqn.evaluate(num_episodes=1)
    assert "eval_return_mean" in noisy_dqn.evaluate(num_episodes=1)
    assert "eval_return_mean" in prioritized_dqn.evaluate(num_episodes=1)
    assert "eval_return_mean" in rainbow_dqn.evaluate(num_episodes=1)
    assert "eval_return_mean" in c51_dqn.evaluate(num_episodes=1)
    assert "eval_return_mean" in n_step_dqn.evaluate(num_episodes=1)
    assert "eval_return_mean" in qr_dqn.evaluate(num_episodes=1)
    assert "eval_return_mean" in iqn.evaluate(num_episodes=1)
    assert "eval_return_mean" in sac.evaluate(num_episodes=1)
    assert "eval_return_mean" in cql.evaluate(num_episodes=1)
    assert "eval_return_mean" in cal_ql.evaluate(num_episodes=1)
    assert "eval_return_mean" in edac.evaluate(num_episodes=1)
    assert "eval_return_mean" in awr.evaluate(num_episodes=1)
    assert "eval_return_mean" in marwil.evaluate(num_episodes=1)
    assert "eval_return_mean" in rlpd.evaluate(num_episodes=1)
    assert "eval_return_mean" in tqc.evaluate(num_episodes=1)
    assert "eval_return_mean" in redq.evaluate(num_episodes=1)
    assert "eval_return_mean" in iql.evaluate(num_episodes=1)
    assert "eval_return_mean" in xql.evaluate(num_episodes=1)
    assert "eval_return_mean" in td3_bc.evaluate(num_episodes=1)
    assert "eval_return_mean" in ddpg.evaluate(num_episodes=1)
    assert "eval_return_mean" in td3.evaluate(num_episodes=1)
