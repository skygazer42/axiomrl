from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

from rl_training.envs import POINT_GOAL_ENV_ID
from rl_training.experiment.config import TrainConfig
from rl_training.runtime.ars_trainer import train_ars
from rl_training.runtime.openai_es_trainer import train_openai_es
from rl_training.runtime.awac_trainer import train_awac
from rl_training.runtime.awr_trainer import train_awr
from rl_training.runtime.marwil_trainer import train_marwil
from rl_training.runtime.bear_trainer import train_bear
from rl_training.runtime.bcq_trainer import train_bcq
from rl_training.runtime.cal_ql_trainer import train_cal_ql
from rl_training.runtime.curl_trainer import train_curl
from rl_training.runtime.crossq_trainer import train_crossq
from rl_training.runtime.crr_trainer import train_crr
from rl_training.runtime.drq_trainer import train_drq
from rl_training.runtime.drqv2_trainer import train_drqv2
from rl_training.runtime.r2d2_trainer import train_r2d2
from rl_training.runtime.discrete_sac_trainer import train_discrete_sac
from rl_training.runtime.dqn_trainer import train_dqn
from rl_training.runtime.edac_trainer import train_edac
from rl_training.runtime.her_trainer import train_her
from rl_training.runtime.iql_trainer import train_iql
from rl_training.runtime.decision_transformer_trainer import train_decision_transformer
from rl_training.runtime.impala_trainer import train_impala
from rl_training.runtime.appo_trainer import train_appo
from rl_training.runtime.mopo_trainer import train_mopo
from rl_training.runtime.pets_trainer import train_pets
from rl_training.runtime.ppo_trainer import train_ppo
from rl_training.runtime.ppg_trainer import train_ppg
from rl_training.runtime.recurrent_ppo_trainer import train_recurrent_ppo
from rl_training.runtime.redq_trainer import train_redq
from rl_training.runtime.rlpd_trainer import train_rlpd
from rl_training.runtime.rebrac_trainer import train_rebrac
from rl_training.runtime.cql_trainer import train_cql
from rl_training.runtime.sac_trainer import train_sac
from rl_training.runtime.td3_bc_trainer import train_td3_bc
from rl_training.runtime.tqc_trainer import train_tqc
from rl_training.runtime.trpo_trainer import train_trpo
from rl_training.runtime.workflows import evaluate_checkpoint, resume_training
from rl_training.runtime.xql_trainer import train_xql


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
    env_id = "RLTrainingTest/CheckpointDrQv2-v0"
    try:
        gym.spec(env_id)
    except gym.error.Error:
        gym.register(id=env_id, entry_point=TinyRenderContinuousEnv)
    return env_id


def test_evaluate_checkpoint_returns_metrics_for_ppo(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="ppo",
        env_id="CartPole-v1",
        seed=13,
        total_timesteps=64,
        output_dir=tmp_path,
        num_envs=2,
        eval_episodes=1,
        algo_kwargs={
            "num_steps": 32,
            "update_epochs": 1,
            "minibatch_size": 32,
            "hidden_sizes": (16, 16),
        },
    )

    train_result = train_ppo(config, run_suffix="eval-source")
    metrics = evaluate_checkpoint(train_result.checkpoint_path, num_episodes=1)

    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_evaluate_checkpoint_returns_metrics_for_ars(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="ars",
        env_id="Pendulum-v1",
        seed=12,
        total_timesteps=100,
        output_dir=tmp_path,
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

    train_result = train_ars(config, run_suffix="ars-eval-source")
    metrics = evaluate_checkpoint(train_result.checkpoint_path, num_episodes=1)

    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_evaluate_checkpoint_returns_metrics_for_openai_es(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="openai_es",
        env_id="Pendulum-v1",
        seed=11,
        total_timesteps=100,
        output_dir=tmp_path,
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

    train_result = train_openai_es(config, run_suffix="openai-es-eval-source")
    metrics = evaluate_checkpoint(train_result.checkpoint_path, num_episodes=1)

    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_evaluate_checkpoint_returns_metrics_for_trpo(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="trpo",
        env_id="CartPole-v1",
        seed=14,
        total_timesteps=64,
        output_dir=tmp_path,
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

    train_result = train_trpo(config, run_suffix="eval-source")
    metrics = evaluate_checkpoint(train_result.checkpoint_path, num_episodes=1)

    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_evaluate_checkpoint_returns_metrics_for_decision_transformer(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="decision_transformer",
        env_id="Pendulum-v1",
        seed=15,
        total_timesteps=8,
        output_dir=tmp_path,
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

    train_result = train_decision_transformer(config, run_suffix="dt-eval-source")
    metrics = evaluate_checkpoint(train_result.checkpoint_path, num_episodes=1)

    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_evaluate_checkpoint_returns_metrics_for_impala(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="impala",
        env_id="CartPole-v1",
        seed=16,
        total_timesteps=64,
        output_dir=tmp_path,
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

    train_result = train_impala(config, run_suffix="impala-eval-source")
    metrics = evaluate_checkpoint(train_result.checkpoint_path, num_episodes=1)

    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_evaluate_checkpoint_returns_metrics_for_appo(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="appo",
        env_id="CartPole-v1",
        seed=17,
        total_timesteps=64,
        output_dir=tmp_path,
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

    train_result = train_appo(config, run_suffix="appo-eval-source")
    metrics = evaluate_checkpoint(train_result.checkpoint_path, num_episodes=1)

    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_evaluate_checkpoint_returns_metrics_for_mopo(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="mopo",
        env_id="Pendulum-v1",
        seed=16,
        total_timesteps=8,
        output_dir=tmp_path,
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

    train_result = train_mopo(config, run_suffix="mopo-eval-source")
    metrics = evaluate_checkpoint(train_result.checkpoint_path, num_episodes=1)

    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_evaluate_checkpoint_returns_metrics_for_pets(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="pets",
        env_id="Pendulum-v1",
        seed=17,
        total_timesteps=64,
        output_dir=tmp_path,
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

    train_result = train_pets(config, run_suffix="pets-eval-source")
    metrics = evaluate_checkpoint(train_result.checkpoint_path, num_episodes=1)

    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_evaluate_checkpoint_returns_metrics_for_discrete_sac(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="discrete_sac",
        env_id="CartPole-v1",
        seed=15,
        total_timesteps=128,
        output_dir=tmp_path,
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

    train_result = train_discrete_sac(config, run_suffix="eval-source")
    metrics = evaluate_checkpoint(train_result.checkpoint_path, num_episodes=1)

    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_evaluate_checkpoint_returns_metrics_for_crossq(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="crossq",
        env_id="Pendulum-v1",
        seed=16,
        total_timesteps=128,
        output_dir=tmp_path,
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

    train_result = train_crossq(config, run_suffix="eval-source")
    metrics = evaluate_checkpoint(train_result.checkpoint_path, num_episodes=1)

    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_evaluate_checkpoint_returns_metrics_for_drqv2(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="drqv2",
        env_id=_register_tiny_render_env(),
        seed=17,
        total_timesteps=96,
        output_dir=tmp_path,
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

    train_result = train_drqv2(config, run_suffix="eval-source")
    metrics = evaluate_checkpoint(train_result.checkpoint_path, num_episodes=1)

    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_evaluate_checkpoint_returns_metrics_for_drq(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="drq",
        env_id=_register_tiny_render_env(),
        seed=18,
        total_timesteps=96,
        output_dir=tmp_path,
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

    train_result = train_drq(config, run_suffix="eval-source")
    metrics = evaluate_checkpoint(train_result.checkpoint_path, num_episodes=1)

    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_evaluate_checkpoint_returns_metrics_for_recurrent_ppo(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="recurrent_ppo",
        env_id="CartPole-v1",
        seed=15,
        total_timesteps=64,
        output_dir=tmp_path,
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

    train_result = train_recurrent_ppo(config, run_suffix="recurrent-eval-source")
    metrics = evaluate_checkpoint(train_result.checkpoint_path, num_episodes=1)

    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_evaluate_checkpoint_returns_metrics_for_ppg(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="ppg",
        env_id="CartPole-v1",
        seed=19,
        total_timesteps=128,
        output_dir=tmp_path,
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

    train_result = train_ppg(config, run_suffix="ppg-eval-source")
    metrics = evaluate_checkpoint(train_result.checkpoint_path, num_episodes=1)

    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_evaluate_checkpoint_returns_metrics_for_curl(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="curl",
        env_id=_register_tiny_render_env(),
        seed=119,
        total_timesteps=96,
        output_dir=tmp_path,
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

    train_result = train_curl(config, run_suffix="curl-eval-source")
    metrics = evaluate_checkpoint(train_result.checkpoint_path, num_episodes=1)

    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_evaluate_checkpoint_returns_metrics_for_r2d2(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="r2d2",
        env_id="CartPole-v1",
        seed=16,
        total_timesteps=96,
        output_dir=tmp_path,
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
        },
    )

    train_result = train_r2d2(config, run_suffix="r2d2-eval-source")
    metrics = evaluate_checkpoint(train_result.checkpoint_path, num_episodes=1)

    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_resume_training_advances_global_step_for_dqn(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="dqn",
        env_id="CartPole-v1",
        seed=17,
        total_timesteps=96,
        output_dir=tmp_path,
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

    train_result = train_dqn(config, run_suffix="resume-source")
    resumed = resume_training(
        train_result.checkpoint_path,
        total_timesteps=160,
        run_suffix="resume-target",
    )

    assert resumed.checkpoint_path is not None
    assert resumed.checkpoint_path.exists()
    assert resumed.metrics["global_step"] >= 160


def test_resume_training_advances_global_step_for_ars(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="ars",
        env_id="Pendulum-v1",
        seed=18,
        total_timesteps=100,
        output_dir=tmp_path,
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

    train_result = train_ars(config, run_suffix="ars-resume-source")
    resumed = resume_training(
        train_result.checkpoint_path,
        total_timesteps=200,
        run_suffix="ars-resume-target",
    )

    assert resumed.checkpoint_path is not None
    assert resumed.checkpoint_path.exists()
    assert resumed.metrics["global_step"] >= 200


def test_resume_training_advances_global_step_for_openai_es(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="openai_es",
        env_id="Pendulum-v1",
        seed=19,
        total_timesteps=100,
        output_dir=tmp_path,
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

    train_result = train_openai_es(config, run_suffix="openai-es-resume-source")
    resumed = resume_training(
        train_result.checkpoint_path,
        total_timesteps=200,
        run_suffix="openai-es-resume-target",
    )

    assert resumed.checkpoint_path is not None
    assert resumed.checkpoint_path.exists()
    assert resumed.metrics["global_step"] >= 200


def test_resume_training_advances_global_step_for_recurrent_ppo(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="recurrent_ppo",
        env_id="CartPole-v1",
        seed=19,
        total_timesteps=64,
        output_dir=tmp_path,
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

    train_result = train_recurrent_ppo(config, run_suffix="recurrent-resume-source")
    resumed = resume_training(
        train_result.checkpoint_path,
        total_timesteps=128,
        run_suffix="recurrent-resume-target",
    )

    assert resumed.checkpoint_path is not None
    assert resumed.checkpoint_path.exists()
    assert resumed.metrics["global_step"] >= 128


def test_resume_training_advances_global_step_for_decision_transformer(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="decision_transformer",
        env_id="Pendulum-v1",
        seed=20,
        total_timesteps=8,
        output_dir=tmp_path,
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

    train_result = train_decision_transformer(config, run_suffix="dt-resume-source")
    resumed = resume_training(
        train_result.checkpoint_path,
        total_timesteps=16,
        run_suffix="dt-resume-target",
    )

    assert resumed.checkpoint_path is not None
    assert resumed.checkpoint_path.exists()
    assert resumed.metrics["global_step"] >= 16


def test_resume_training_advances_global_step_for_mopo(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="mopo",
        env_id="Pendulum-v1",
        seed=21,
        total_timesteps=8,
        output_dir=tmp_path,
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

    train_result = train_mopo(config, run_suffix="mopo-resume-source")
    resumed = resume_training(
        train_result.checkpoint_path,
        total_timesteps=16,
        run_suffix="mopo-resume-target",
    )

    assert resumed.checkpoint_path is not None
    assert resumed.checkpoint_path.exists()
    assert resumed.metrics["global_step"] >= 16


def test_resume_training_advances_global_step_for_pets(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="pets",
        env_id="Pendulum-v1",
        seed=22,
        total_timesteps=64,
        output_dir=tmp_path,
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

    train_result = train_pets(config, run_suffix="pets-resume-source")
    resumed = resume_training(
        train_result.checkpoint_path,
        total_timesteps=96,
        run_suffix="pets-resume-target",
    )

    assert resumed.checkpoint_path is not None
    assert resumed.checkpoint_path.exists()
    assert resumed.metrics["global_step"] >= 96


def test_resume_training_advances_global_step_for_impala(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="impala",
        env_id="CartPole-v1",
        seed=22,
        total_timesteps=64,
        output_dir=tmp_path,
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

    train_result = train_impala(config, run_suffix="impala-resume-source")
    resumed = resume_training(
        train_result.checkpoint_path,
        total_timesteps=128,
        run_suffix="impala-resume-target",
    )

    assert resumed.checkpoint_path is not None
    assert resumed.checkpoint_path.exists()
    assert resumed.metrics["global_step"] >= 128


def test_resume_training_advances_global_step_for_appo(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="appo",
        env_id="CartPole-v1",
        seed=23,
        total_timesteps=64,
        output_dir=tmp_path,
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

    train_result = train_appo(config, run_suffix="appo-resume-source")
    resumed = resume_training(
        train_result.checkpoint_path,
        total_timesteps=128,
        run_suffix="appo-resume-target",
    )

    assert resumed.checkpoint_path is not None
    assert resumed.checkpoint_path.exists()
    assert resumed.metrics["global_step"] >= 128


def test_resume_training_advances_global_step_for_drq(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="drq",
        env_id=_register_tiny_render_env(),
        seed=21,
        total_timesteps=64,
        output_dir=tmp_path,
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

    train_result = train_drq(config, run_suffix="drq-resume-source")
    resumed = resume_training(
        train_result.checkpoint_path,
        total_timesteps=128,
        run_suffix="drq-resume-target",
    )

    assert resumed.checkpoint_path is not None
    assert resumed.checkpoint_path.exists()
    assert resumed.metrics["global_step"] >= 128


def test_resume_training_advances_global_step_for_ppg(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="ppg",
        env_id="CartPole-v1",
        seed=22,
        total_timesteps=64,
        output_dir=tmp_path,
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

    train_result = train_ppg(config, run_suffix="ppg-resume-source")
    resumed = resume_training(
        train_result.checkpoint_path,
        total_timesteps=128,
        run_suffix="ppg-resume-target",
    )

    assert resumed.checkpoint_path is not None
    assert resumed.checkpoint_path.exists()
    assert resumed.metrics["global_step"] >= 128


def test_resume_training_advances_global_step_for_curl(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="curl",
        env_id=_register_tiny_render_env(),
        seed=122,
        total_timesteps=64,
        output_dir=tmp_path,
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

    train_result = train_curl(config, run_suffix="curl-resume-source")
    resumed = resume_training(
        train_result.checkpoint_path,
        total_timesteps=128,
        run_suffix="curl-resume-target",
    )

    assert resumed.checkpoint_path is not None
    assert resumed.checkpoint_path.exists()
    assert resumed.metrics["global_step"] >= 128


def test_resume_training_preserves_ppg_auxiliary_phase_schedule(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="ppg",
        env_id="CartPole-v1",
        seed=222,
        total_timesteps=192,
        output_dir=tmp_path,
        num_envs=2,
        eval_episodes=1,
        algo_kwargs={
            "num_steps": 32,
            "update_epochs": 1,
            "minibatch_size": 32,
            "hidden_sizes": (32, 32),
            "aux_frequency": 2,
            "aux_epochs": 1,
            "aux_minibatch_size": 32,
            "aux_buffer_rollouts": 2,
        },
    )

    train_result = train_ppg(config, run_suffix="ppg-aux-schedule-source")
    resumed = resume_training(
        train_result.checkpoint_path,
        total_timesteps=256,
        run_suffix="ppg-aux-schedule-target",
    )

    assert resumed.metrics["global_step"] >= 256
    assert resumed.metrics["auxiliary_phase_ran"] == pytest.approx(1.0)


def test_resume_training_advances_global_step_for_r2d2(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="r2d2",
        env_id="CartPole-v1",
        seed=20,
        total_timesteps=64,
        output_dir=tmp_path,
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
        },
    )

    train_result = train_r2d2(config, run_suffix="r2d2-resume-source")
    resumed = resume_training(
        train_result.checkpoint_path,
        total_timesteps=128,
        run_suffix="r2d2-resume-target",
    )

    assert resumed.checkpoint_path is not None
    assert resumed.checkpoint_path.exists()
    assert resumed.metrics["global_step"] >= 128


def test_resume_training_advances_global_step_for_trpo(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="trpo",
        env_id="CartPole-v1",
        seed=20,
        total_timesteps=64,
        output_dir=tmp_path,
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

    train_result = train_trpo(config, run_suffix="trpo-resume-source")
    resumed = resume_training(
        train_result.checkpoint_path,
        total_timesteps=128,
        run_suffix="trpo-resume-target",
    )

    assert resumed.checkpoint_path is not None
    assert resumed.checkpoint_path.exists()
    assert resumed.metrics["global_step"] >= 128


def test_resume_training_advances_global_step_for_discrete_sac(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="discrete_sac",
        env_id="CartPole-v1",
        seed=21,
        total_timesteps=128,
        output_dir=tmp_path,
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

    train_result = train_discrete_sac(config, run_suffix="discrete-sac-resume-source")
    resumed = resume_training(
        train_result.checkpoint_path,
        total_timesteps=192,
        run_suffix="discrete-sac-resume-target",
    )

    assert resumed.checkpoint_path is not None
    assert resumed.checkpoint_path.exists()
    assert resumed.metrics["global_step"] >= 192


def test_resume_training_advances_global_step_for_crossq(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="crossq",
        env_id="Pendulum-v1",
        seed=22,
        total_timesteps=128,
        output_dir=tmp_path,
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

    train_result = train_crossq(config, run_suffix="crossq-resume-source")
    resumed = resume_training(
        train_result.checkpoint_path,
        total_timesteps=192,
        run_suffix="crossq-resume-target",
    )

    assert resumed.checkpoint_path is not None
    assert resumed.checkpoint_path.exists()
    assert resumed.metrics["global_step"] >= 192


def test_evaluate_checkpoint_returns_metrics_for_sac(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="sac",
        env_id="Pendulum-v1",
        seed=37,
        total_timesteps=96,
        output_dir=tmp_path,
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

    train_result = train_sac(config, run_suffix="sac-eval-source")
    metrics = evaluate_checkpoint(train_result.checkpoint_path, num_episodes=1)

    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_evaluate_checkpoint_returns_metrics_for_her(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="her",
        env_id=POINT_GOAL_ENV_ID,
        seed=41,
        total_timesteps=64,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "buffer_capacity": 512,
            "batch_size": 16,
            "learning_starts": 8,
            "train_frequency": 1,
            "hidden_sizes": (32, 32),
            "eval_interval": 16,
        },
    )

    train_result = train_her(config, run_suffix="her-eval-source")
    metrics = evaluate_checkpoint(train_result.checkpoint_path, num_episodes=1)

    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes", "eval_success_rate"}


def test_evaluate_checkpoint_returns_metrics_for_tqc(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="tqc",
        env_id="Pendulum-v1",
        seed=43,
        total_timesteps=96,
        output_dir=tmp_path,
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

    train_result = train_tqc(config, run_suffix="tqc-eval-source")
    metrics = evaluate_checkpoint(train_result.checkpoint_path, num_episodes=1)

    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_evaluate_checkpoint_returns_metrics_for_redq(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="redq",
        env_id="Pendulum-v1",
        seed=47,
        total_timesteps=96,
        output_dir=tmp_path,
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

    train_result = train_redq(config, run_suffix="redq-eval-source")
    metrics = evaluate_checkpoint(train_result.checkpoint_path, num_episodes=1)

    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_resume_training_advances_global_step_for_redq(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="redq",
        env_id="Pendulum-v1",
        seed=53,
        total_timesteps=96,
        output_dir=tmp_path,
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

    train_result = train_redq(config, run_suffix="redq-resume-source")
    resumed = resume_training(
        train_result.checkpoint_path,
        total_timesteps=160,
        run_suffix="redq-resume-target",
    )

    assert resumed.checkpoint_path is not None
    assert resumed.checkpoint_path.exists()
    assert resumed.metrics["global_step"] >= 160


def test_resume_training_advances_global_step_for_her(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="her",
        env_id=POINT_GOAL_ENV_ID,
        seed=71,
        total_timesteps=64,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "buffer_capacity": 512,
            "batch_size": 16,
            "learning_starts": 8,
            "train_frequency": 1,
            "hidden_sizes": (32, 32),
            "eval_interval": 16,
        },
    )

    train_result = train_her(config, run_suffix="her-resume-source")
    resumed = resume_training(
        train_result.checkpoint_path,
        total_timesteps=128,
        run_suffix="her-resume-target",
    )

    assert resumed.checkpoint_path is not None
    assert resumed.checkpoint_path.exists()
    assert resumed.metrics["global_step"] >= 128


def test_evaluate_checkpoint_returns_metrics_for_iql(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="iql",
        env_id="Pendulum-v1",
        seed=59,
        total_timesteps=96,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 192,
            "dataset_seed": 31,
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

    train_result = train_iql(config, run_suffix="iql-eval-source")
    metrics = evaluate_checkpoint(train_result.checkpoint_path, num_episodes=1)

    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_evaluate_checkpoint_returns_metrics_for_cql(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="cql",
        env_id="Pendulum-v1",
        seed=63,
        total_timesteps=96,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 192,
            "dataset_seed": 37,
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

    train_result = train_cql(config, run_suffix="cql-eval-source")
    metrics = evaluate_checkpoint(train_result.checkpoint_path, num_episodes=1)

    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_evaluate_checkpoint_returns_metrics_for_cal_ql(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="cal_ql",
        env_id="Pendulum-v1",
        seed=64,
        total_timesteps=96,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 192,
            "dataset_seed": 40,
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

    train_result = train_cal_ql(config, run_suffix="cal-ql-eval-source")
    metrics = evaluate_checkpoint(train_result.checkpoint_path, num_episodes=1)

    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_evaluate_checkpoint_returns_metrics_for_edac(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="edac",
        env_id="Pendulum-v1",
        seed=64,
        total_timesteps=96,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 192,
            "dataset_seed": 40,
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

    train_result = train_edac(config, run_suffix="edac-eval-source")
    metrics = evaluate_checkpoint(train_result.checkpoint_path, num_episodes=1)

    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_evaluate_checkpoint_returns_metrics_for_awr(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="awr",
        env_id="Pendulum-v1",
        seed=65,
        total_timesteps=96,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 192,
            "dataset_seed": 41,
            "batch_size": 32,
            "hidden_sizes": (32, 32),
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "beta": 1.0,
            "max_weight": 20.0,
        },
    )

    train_result = train_awr(config, run_suffix="awr-eval-source")
    metrics = evaluate_checkpoint(train_result.checkpoint_path, num_episodes=1)

    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_evaluate_checkpoint_returns_metrics_for_marwil(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="marwil",
        env_id="Pendulum-v1",
        seed=65,
        total_timesteps=96,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 192,
            "dataset_seed": 41,
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

    train_result = train_marwil(config, run_suffix="marwil-eval-source")
    metrics = evaluate_checkpoint(train_result.checkpoint_path, num_episodes=1)

    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_evaluate_checkpoint_returns_metrics_for_rlpd(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="rlpd",
        env_id="Pendulum-v1",
        seed=65,
        total_timesteps=96,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 192,
            "dataset_seed": 41,
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

    train_result = train_rlpd(config, run_suffix="rlpd-eval-source")
    metrics = evaluate_checkpoint(train_result.checkpoint_path, num_episodes=1)

    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_evaluate_checkpoint_returns_metrics_for_xql(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="xql",
        env_id="Pendulum-v1",
        seed=65,
        total_timesteps=96,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 192,
            "dataset_seed": 41,
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

    train_result = train_xql(config, run_suffix="xql-eval-source")
    metrics = evaluate_checkpoint(train_result.checkpoint_path, num_episodes=1)

    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_evaluate_checkpoint_returns_metrics_for_awac(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="awac",
        env_id="Pendulum-v1",
        seed=64,
        total_timesteps=96,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 192,
            "dataset_seed": 38,
            "batch_size": 32,
            "hidden_sizes": (32, 32),
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "tau": 0.005,
            "awac_lambda": 1.0,
            "max_advantage_weight": 20.0,
        },
    )

    train_result = train_awac(config, run_suffix="awac-eval-source")
    metrics = evaluate_checkpoint(train_result.checkpoint_path, num_episodes=1)

    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_evaluate_checkpoint_returns_metrics_for_crr(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="crr",
        env_id="Pendulum-v1",
        seed=65,
        total_timesteps=96,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 192,
            "dataset_seed": 39,
            "batch_size": 32,
            "hidden_sizes": (32, 32),
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "tau": 0.005,
            "beta": 1.0,
            "n_action_samples": 4,
            "max_weight": 20.0,
            "advantage_type": "mean",
            "weight_type": "exp",
        },
    )

    train_result = train_crr(config, run_suffix="crr-eval-source")
    metrics = evaluate_checkpoint(train_result.checkpoint_path, num_episodes=1)

    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_evaluate_checkpoint_returns_metrics_for_rebrac(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="rebrac",
        env_id="Pendulum-v1",
        seed=66,
        total_timesteps=96,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 192,
            "dataset_seed": 40,
            "batch_size": 32,
            "hidden_sizes": (32, 32),
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "tau": 0.005,
            "policy_noise": 0.2,
            "noise_clip": 0.5,
            "policy_delay": 2,
            "actor_bc_weight": 1.0,
            "critic_bc_weight": 1.0,
            "actor_q_weight": 1.0,
        },
    )

    train_result = train_rebrac(config, run_suffix="rebrac-eval-source")
    metrics = evaluate_checkpoint(train_result.checkpoint_path, num_episodes=1)

    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_evaluate_checkpoint_returns_metrics_for_bear(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="bear",
        env_id="Pendulum-v1",
        seed=65,
        total_timesteps=96,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 192,
            "dataset_seed": 38,
            "batch_size": 32,
            "hidden_sizes": (32, 32),
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "tau": 0.005,
            "latent_dim": 2,
            "behavior_kl_weight": 0.5,
            "mmd_sigma": 20.0,
            "mmd_alpha": 10.0,
            "num_mmd_action_samples": 10,
        },
    )

    train_result = train_bear(config, run_suffix="bear-eval-source")
    metrics = evaluate_checkpoint(train_result.checkpoint_path, num_episodes=1)

    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_evaluate_checkpoint_returns_metrics_for_bcq(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="bcq",
        env_id="Pendulum-v1",
        seed=65,
        total_timesteps=96,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 192,
            "dataset_seed": 39,
            "batch_size": 32,
            "hidden_sizes": (32, 32),
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "tau": 0.005,
            "latent_dim": 2,
            "num_action_samples": 10,
            "perturbation_scale": 0.05,
            "vae_kl_weight": 0.5,
        },
    )

    train_result = train_bcq(config, run_suffix="bcq-eval-source")
    metrics = evaluate_checkpoint(train_result.checkpoint_path, num_episodes=1)

    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_resume_training_advances_global_step_for_iql(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="iql",
        env_id="Pendulum-v1",
        seed=61,
        total_timesteps=96,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 192,
            "dataset_seed": 37,
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

    train_result = train_iql(config, run_suffix="iql-resume-source")
    resumed = resume_training(
        train_result.checkpoint_path,
        total_timesteps=160,
        run_suffix="iql-resume-target",
    )

    assert resumed.checkpoint_path is not None
    assert resumed.checkpoint_path.exists()
    assert resumed.metrics["global_step"] >= 160


def test_resume_training_advances_global_step_for_awac(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="awac",
        env_id="Pendulum-v1",
        seed=66,
        total_timesteps=96,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 192,
            "dataset_seed": 40,
            "batch_size": 32,
            "hidden_sizes": (32, 32),
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "tau": 0.005,
            "awac_lambda": 1.0,
            "max_advantage_weight": 20.0,
        },
    )

    train_result = train_awac(config, run_suffix="awac-resume-source")
    resumed = resume_training(
        train_result.checkpoint_path,
        total_timesteps=160,
        run_suffix="awac-resume-target",
    )

    assert resumed.checkpoint_path is not None
    assert resumed.checkpoint_path.exists()
    assert resumed.metrics["global_step"] >= 160


def test_resume_training_advances_global_step_for_crr(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="crr",
        env_id="Pendulum-v1",
        seed=67,
        total_timesteps=96,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 192,
            "dataset_seed": 41,
            "batch_size": 32,
            "hidden_sizes": (32, 32),
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "tau": 0.005,
            "beta": 1.0,
            "n_action_samples": 4,
            "max_weight": 20.0,
            "advantage_type": "mean",
            "weight_type": "exp",
        },
    )

    train_result = train_crr(config, run_suffix="crr-resume-source")
    resumed = resume_training(
        train_result.checkpoint_path,
        total_timesteps=160,
        run_suffix="crr-resume-target",
    )

    assert resumed.checkpoint_path is not None
    assert resumed.checkpoint_path.exists()
    assert resumed.metrics["global_step"] >= 160


def test_resume_training_advances_global_step_for_rebrac(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="rebrac",
        env_id="Pendulum-v1",
        seed=68,
        total_timesteps=96,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 192,
            "dataset_seed": 42,
            "batch_size": 32,
            "hidden_sizes": (32, 32),
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "tau": 0.005,
            "policy_noise": 0.2,
            "noise_clip": 0.5,
            "policy_delay": 2,
            "actor_bc_weight": 1.0,
            "critic_bc_weight": 1.0,
            "actor_q_weight": 1.0,
        },
    )

    train_result = train_rebrac(config, run_suffix="rebrac-resume-source")
    resumed = resume_training(
        train_result.checkpoint_path,
        total_timesteps=160,
        run_suffix="rebrac-resume-target",
    )

    assert resumed.checkpoint_path is not None
    assert resumed.checkpoint_path.exists()
    assert resumed.metrics["global_step"] >= 160


def test_resume_training_advances_global_step_for_bear(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="bear",
        env_id="Pendulum-v1",
        seed=67,
        total_timesteps=96,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 192,
            "dataset_seed": 41,
            "batch_size": 32,
            "hidden_sizes": (32, 32),
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "tau": 0.005,
            "latent_dim": 2,
            "behavior_kl_weight": 0.5,
            "mmd_sigma": 20.0,
            "mmd_alpha": 10.0,
            "num_mmd_action_samples": 10,
        },
    )

    train_result = train_bear(config, run_suffix="bear-resume-source")
    resumed = resume_training(
        train_result.checkpoint_path,
        total_timesteps=160,
        run_suffix="bear-resume-target",
    )

    assert resumed.checkpoint_path is not None
    assert resumed.checkpoint_path.exists()
    assert resumed.metrics["global_step"] >= 160


def test_resume_training_advances_global_step_for_bcq(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="bcq",
        env_id="Pendulum-v1",
        seed=68,
        total_timesteps=96,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 192,
            "dataset_seed": 42,
            "batch_size": 32,
            "hidden_sizes": (32, 32),
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "tau": 0.005,
            "latent_dim": 2,
            "num_action_samples": 10,
            "perturbation_scale": 0.05,
            "vae_kl_weight": 0.5,
        },
    )

    train_result = train_bcq(config, run_suffix="bcq-resume-source")
    resumed = resume_training(
        train_result.checkpoint_path,
        total_timesteps=160,
        run_suffix="bcq-resume-target",
    )

    assert resumed.checkpoint_path is not None
    assert resumed.checkpoint_path.exists()
    assert resumed.metrics["global_step"] >= 160


def test_resume_training_advances_global_step_for_cql(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="cql",
        env_id="Pendulum-v1",
        seed=65,
        total_timesteps=96,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 192,
            "dataset_seed": 39,
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

    train_result = train_cql(config, run_suffix="cql-resume-source")
    resumed = resume_training(
        train_result.checkpoint_path,
        total_timesteps=160,
        run_suffix="cql-resume-target",
    )

    assert resumed.checkpoint_path is not None
    assert resumed.checkpoint_path.exists()
    assert resumed.metrics["global_step"] >= 160


def test_resume_training_advances_global_step_for_cal_ql(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="cal_ql",
        env_id="Pendulum-v1",
        seed=66,
        total_timesteps=96,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 192,
            "dataset_seed": 41,
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

    train_result = train_cal_ql(config, run_suffix="cal-ql-resume-source")
    resumed = resume_training(
        train_result.checkpoint_path,
        total_timesteps=160,
        run_suffix="cal-ql-resume-target",
    )

    assert resumed.checkpoint_path is not None
    assert resumed.checkpoint_path.exists()
    assert resumed.metrics["global_step"] >= 160


def test_resume_training_advances_global_step_for_edac(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="edac",
        env_id="Pendulum-v1",
        seed=66,
        total_timesteps=96,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 192,
            "dataset_seed": 41,
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

    train_result = train_edac(config, run_suffix="edac-resume-source")
    resumed = resume_training(
        train_result.checkpoint_path,
        total_timesteps=160,
        run_suffix="edac-resume-target",
    )

    assert resumed.checkpoint_path is not None
    assert resumed.checkpoint_path.exists()
    assert resumed.metrics["global_step"] >= 160


def test_resume_training_advances_global_step_for_awr(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="awr",
        env_id="Pendulum-v1",
        seed=67,
        total_timesteps=96,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 192,
            "dataset_seed": 43,
            "batch_size": 32,
            "hidden_sizes": (32, 32),
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "beta": 1.0,
            "max_weight": 20.0,
        },
    )

    train_result = train_awr(config, run_suffix="awr-resume-source")
    resumed = resume_training(
        train_result.checkpoint_path,
        total_timesteps=160,
        run_suffix="awr-resume-target",
    )

    assert resumed.checkpoint_path is not None
    assert resumed.checkpoint_path.exists()
    assert resumed.metrics["global_step"] >= 160


def test_resume_training_advances_global_step_for_marwil(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="marwil",
        env_id="Pendulum-v1",
        seed=67,
        total_timesteps=96,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 192,
            "dataset_seed": 43,
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

    train_result = train_marwil(config, run_suffix="marwil-resume-source")
    resumed = resume_training(
        train_result.checkpoint_path,
        total_timesteps=160,
        run_suffix="marwil-resume-target",
    )

    assert resumed.checkpoint_path is not None
    assert resumed.checkpoint_path.exists()
    assert resumed.metrics["global_step"] >= 160


def test_resume_training_advances_global_step_for_rlpd(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="rlpd",
        env_id="Pendulum-v1",
        seed=67,
        total_timesteps=96,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 192,
            "dataset_seed": 42,
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

    train_result = train_rlpd(config, run_suffix="rlpd-resume-source")
    resumed = resume_training(
        train_result.checkpoint_path,
        total_timesteps=160,
        run_suffix="rlpd-resume-target",
    )

    assert resumed.checkpoint_path is not None
    assert resumed.checkpoint_path.exists()
    assert resumed.metrics["global_step"] >= 160


def test_resume_training_advances_global_step_for_xql(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="xql",
        env_id="Pendulum-v1",
        seed=81,
        total_timesteps=96,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 192,
            "dataset_seed": 59,
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

    train_result = train_xql(config, run_suffix="xql-resume-source")
    resumed = resume_training(
        train_result.checkpoint_path,
        total_timesteps=160,
        run_suffix="xql-resume-target",
    )

    assert resumed.checkpoint_path is not None
    assert resumed.checkpoint_path.exists()
    assert resumed.metrics["global_step"] >= 160


def test_evaluate_checkpoint_returns_metrics_for_td3_bc(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="td3_bc",
        env_id="Pendulum-v1",
        seed=67,
        total_timesteps=96,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 192,
            "dataset_seed": 41,
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

    train_result = train_td3_bc(config, run_suffix="td3-bc-eval-source")
    metrics = evaluate_checkpoint(train_result.checkpoint_path, num_episodes=1)

    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_resume_training_advances_global_step_for_td3_bc(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="td3_bc",
        env_id="Pendulum-v1",
        seed=71,
        total_timesteps=96,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 192,
            "dataset_seed": 43,
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

    train_result = train_td3_bc(config, run_suffix="td3-bc-resume-source")
    resumed = resume_training(
        train_result.checkpoint_path,
        total_timesteps=160,
        run_suffix="td3-bc-resume-target",
    )

    assert resumed.checkpoint_path is not None
    assert resumed.checkpoint_path.exists()
    assert resumed.metrics["global_step"] >= 160
