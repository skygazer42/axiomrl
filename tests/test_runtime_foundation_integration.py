from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rl_training.envs import POINT_GOAL_ENV_ID
from rl_training.experiment.config import TrainConfig
from rl_training.runtime import (
    a2c_trainer,
    apex_dqn_trainer,
    agent57_trainer,
    awac_trainer,
    awr_trainer,
    appo_trainer,
    ars_trainer,
    bc_trainer,
    bcq_trainer,
    bear_trainer,
    cal_ql_trainer,
    cql_trainer,
    crr_trainer,
    crossq_trainer,
    curl_trainer,
    d4pg_trainer,
    ddpg_trainer,
    decision_transformer_trainer,
    discrete_sac_trainer,
    dreamer_trainer,
    drqn_trainer,
    drq_trainer,
    drqv2_trainer,
    edac_trainer,
    gail_trainer,
    her_trainer,
    impala_trainer,
    iql_trainer,
    marwil_trainer,
    mbpo_trainer,
    mopo_trainer,
    muzero_trainer,
    naf_trainer,
    openai_es_trainer,
    pets_trainer,
    ppg_trainer,
    r2d2_trainer,
    rebrac_trainer,
    recurrent_ppo_trainer,
    redq_trainer,
    rlpd_trainer,
    sac_trainer,
    efficientzero_trainer,
    td3_trainer,
    td3_bc_trainer,
    xql_trainer,
    tqc_trainer,
    trpo_trainer,
)


class ExpectedSharedSession(RuntimeError):
    pass


class LegacyRunSetupUsed(RuntimeError):
    pass


class ExpectedEvaluationSupport(RuntimeError):
    pass


class LegacyEvaluationPathUsed(RuntimeError):
    pass


def _raise_shared_session(*args, **kwargs):
    raise ExpectedSharedSession("shared session was used")


def _raise_legacy_run_setup(*args, **kwargs):
    raise LegacyRunSetupUsed("legacy run setup was used")


def _raise_expected_eval_support(*args, **kwargs):
    raise ExpectedEvaluationSupport("shared evaluation support was used")


def _raise_legacy_eval_path(*args, **kwargs):
    raise LegacyEvaluationPathUsed("legacy evaluation path was used")


def _make_a2c_config(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        algo="a2c",
        env_id="CartPole-v1",
        seed=13,
        total_timesteps=64,
        output_dir=tmp_path,
        num_envs=2,
        eval_episodes=1,
        algo_kwargs={
            "num_steps": 32,
            "hidden_sizes": (32, 32),
        },
    )


def _make_trpo_config(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        algo="trpo",
        env_id="CartPole-v1",
        seed=17,
        total_timesteps=64,
        output_dir=tmp_path,
        num_envs=2,
        eval_episodes=1,
        algo_kwargs={
            "num_steps": 32,
            "hidden_sizes": (32, 32),
            "value_updates": 3,
            "max_kl": 0.01,
            "cg_iterations": 5,
            "line_search_steps": 5,
        },
    )


def _make_discrete_sac_config(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        algo="discrete_sac",
        env_id="CartPole-v1",
        seed=19,
        total_timesteps=64,
        output_dir=tmp_path,
        num_envs=2,
        eval_episodes=1,
        algo_kwargs={
            "buffer_capacity": 256,
            "batch_size": 32,
            "learning_starts": 32,
            "train_frequency": 1,
            "hidden_sizes": (32, 32),
        },
    )


def _make_ddpg_config(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        algo="ddpg",
        env_id="Pendulum-v1",
        seed=23,
        total_timesteps=64,
        output_dir=tmp_path,
        num_envs=2,
        eval_episodes=1,
        algo_kwargs={
            "buffer_capacity": 256,
            "batch_size": 32,
            "learning_starts": 32,
            "train_frequency": 1,
            "hidden_sizes": (32, 32),
            "tau": 0.005,
        },
    )


def _make_td3_config(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        algo="td3",
        env_id="Pendulum-v1",
        seed=29,
        total_timesteps=64,
        output_dir=tmp_path,
        num_envs=2,
        eval_episodes=1,
        algo_kwargs={
            "buffer_capacity": 256,
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


def _make_redq_config(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        algo="redq",
        env_id="Pendulum-v1",
        seed=31,
        total_timesteps=64,
        output_dir=tmp_path,
        num_envs=2,
        eval_episodes=1,
        algo_kwargs={
            "buffer_capacity": 256,
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


def _make_sac_config(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        algo="sac",
        env_id="Pendulum-v1",
        seed=37,
        total_timesteps=64,
        output_dir=tmp_path,
        num_envs=2,
        eval_episodes=1,
        algo_kwargs={
            "buffer_capacity": 256,
            "batch_size": 32,
            "learning_starts": 32,
            "train_frequency": 1,
            "hidden_sizes": (32, 32),
            "alpha": 0.2,
            "tau": 0.005,
        },
    )


def _make_crossq_config(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        algo="crossq",
        env_id="Pendulum-v1",
        seed=41,
        total_timesteps=64,
        output_dir=tmp_path,
        num_envs=2,
        eval_episodes=1,
        algo_kwargs={
            "buffer_capacity": 256,
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


def _make_tqc_config(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        algo="tqc",
        env_id="Pendulum-v1",
        seed=43,
        total_timesteps=64,
        output_dir=tmp_path,
        num_envs=2,
        eval_episodes=1,
        algo_kwargs={
            "buffer_capacity": 256,
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


def _make_d4pg_config(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        algo="d4pg",
        env_id="Pendulum-v1",
        seed=47,
        total_timesteps=64,
        output_dir=tmp_path,
        num_envs=2,
        eval_episodes=1,
        algo_kwargs={
            "buffer_capacity": 256,
            "batch_size": 32,
            "learning_starts": 32,
            "train_frequency": 1,
            "hidden_sizes": (32, 32),
            "tau": 0.005,
            "exploration_noise": 0.1,
            "v_min": -50.0,
            "v_max": 10.0,
            "num_atoms": 21,
        },
    )


def _make_naf_config(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        algo="naf",
        env_id="Pendulum-v1",
        seed=53,
        total_timesteps=64,
        output_dir=tmp_path,
        num_envs=2,
        eval_episodes=1,
        algo_kwargs={
            "buffer_capacity": 256,
            "batch_size": 32,
            "learning_starts": 32,
            "train_frequency": 1,
            "hidden_sizes": (32, 32),
            "tau": 0.005,
            "exploration_noise": 0.1,
        },
    )


def _make_drq_config(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        algo="drq",
        env_id="Pendulum-v1",
        seed=59,
        total_timesteps=64,
        output_dir=tmp_path,
        num_envs=2,
        eval_episodes=1,
        algo_kwargs={
            "buffer_capacity": 256,
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
    )


def _make_apex_dqn_config(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        algo="apex_dqn",
        env_id="CartPole-v1",
        seed=60,
        total_timesteps=64,
        output_dir=tmp_path,
        num_envs=2,
        eval_episodes=1,
        algo_kwargs={
            "buffer_capacity": 256,
            "batch_size": 32,
            "learning_starts": 32,
            "train_frequency": 1,
            "target_update_interval": 16,
            "learning_rate": 1e-3,
            "gamma": 0.99,
            "n_step": 2,
            "prioritized_alpha": 0.6,
            "prioritized_beta_start": 0.4,
            "prioritized_beta_end": 1.0,
            "prioritized_beta_fraction": 1.0,
            "prioritized_eps": 1e-6,
            "updates_per_collect": 1,
            "actor_epsilon_base": 0.4,
            "actor_epsilon_alpha": 7.0,
            "hidden_sizes": (32, 32),
        },
    )


def _make_drqv2_config(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        algo="drqv2",
        env_id="Pendulum-v1",
        seed=60,
        total_timesteps=64,
        output_dir=tmp_path,
        num_envs=2,
        eval_episodes=1,
        algo_kwargs={
            "buffer_capacity": 256,
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
            "wrappers": {
                "pixels": {
                    "resize_shape": [84, 84],
                    "frame_stack": 3,
                    "channel_first": True,
                }
            }
        },
    )


def _make_curl_config(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        algo="curl",
        env_id="Pendulum-v1",
        seed=62,
        total_timesteps=64,
        output_dir=tmp_path,
        num_envs=2,
        eval_episodes=1,
        algo_kwargs={
            "buffer_capacity": 256,
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
            "wrappers": {
                "pixels": {
                    "resize_shape": [84, 84],
                    "frame_stack": 3,
                    "channel_first": True,
                }
            }
        },
    )


def _make_impala_config(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        algo="impala",
        env_id="CartPole-v1",
        seed=61,
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


def _make_dreamer_config(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        algo="dreamer",
        env_id="CartPole-v1",
        seed=65,
        total_timesteps=64,
        output_dir=tmp_path,
        num_envs=2,
        eval_episodes=1,
        algo_kwargs={
            "buffer_capacity": 256,
            "batch_size": 16,
            "learning_starts": 16,
            "train_frequency": 1,
            "world_model_updates": 1,
            "actor_critic_updates": 1,
            "imagination_batch_size": 16,
            "imagination_horizon": 3,
            "features_dim": 32,
            "action_embed_dim": 16,
            "actor_hidden_sizes": (32,),
            "critic_hidden_sizes": (32,),
            "reward_hidden_sizes": (32,),
            "world_model_learning_rate": 1e-3,
            "actor_learning_rate": 3e-4,
            "critic_learning_rate": 3e-4,
            "gamma": 0.99,
            "entropy_coef": 1e-3,
        },
        env_kwargs={
            "wrappers": {
                "pixels": {
                    "resize_shape": [84, 84],
                    "frame_stack": 3,
                    "channel_first": True,
                }
            }
        },
    )


def _make_iql_config(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        algo="iql",
        env_id="Pendulum-v1",
        seed=63,
        total_timesteps=64,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 128,
            "dataset_seed": 17,
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


def _make_gail_config(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        algo="gail",
        env_id="CartPole-v1",
        seed=64,
        total_timesteps=128,
        output_dir=tmp_path,
        num_envs=2,
        eval_episodes=1,
        algo_kwargs={
            "num_steps": 32,
            "update_epochs": 1,
            "minibatch_size": 32,
            "hidden_sizes": (32, 32),
            "learning_rate": 3e-4,
            "clip_coef": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "discriminator_learning_rate": 3e-4,
            "discriminator_updates": 1,
            "discriminator_batch_size": 32,
            "expert_dataset_kind": "random",
            "expert_dataset_size": 64,
            "expert_dataset_seed": 17,
        },
    )


def _make_mbpo_config(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        algo="mbpo",
        env_id="Pendulum-v1",
        seed=66,
        total_timesteps=64,
        output_dir=tmp_path,
        num_envs=2,
        eval_episodes=1,
        algo_kwargs={
            "buffer_capacity": 256,
            "synthetic_buffer_capacity": 256,
            "batch_size": 16,
            "learning_starts": 16,
            "train_frequency": 1,
            "hidden_sizes": (32, 32),
            "model_hidden_sizes": (32, 32),
            "num_ensembles": 3,
            "model_batch_size": 16,
            "model_updates": 1,
            "rollout_batch_size": 16,
            "rollout_horizon": 1,
            "rollout_refresh_interval": 16,
            "synthetic_batch_ratio": 0.5,
            "policy_learning_rate": 1e-3,
            "model_learning_rate": 1e-3,
            "gamma": 0.99,
            "alpha": 0.2,
            "tau": 0.005,
        },
    )


def _make_mopo_config(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        algo="mopo",
        env_id="Pendulum-v1",
        seed=67,
        total_timesteps=16,
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


def _make_muzero_config(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        algo="muzero",
        env_id="CartPole-v1",
        seed=68,
        total_timesteps=8,
        output_dir=tmp_path,
        num_envs=1,
        eval_episodes=1,
        algo_kwargs={
            "buffer_capacity": 32,
            "batch_size": 4,
            "learning_starts": 4,
            "train_frequency": 1,
            "unroll_steps": 2,
            "learning_rate": 1e-3,
            "gamma": 0.99,
            "num_simulations": 5,
            "latent_dim": 32,
            "action_embed_dim": 16,
            "dynamics_hidden_sizes": (32,),
            "prediction_hidden_sizes": (32,),
        },
        env_kwargs={
            "render_mode": "rgb_array",
            "wrappers": {
                "pixels": {
                    "resize_shape": [84, 84],
                    "frame_stack": 3,
                    "channel_first": True,
                }
            }
        },
    )


def _make_efficientzero_config(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        algo="efficientzero",
        env_id="CartPole-v1",
        seed=69,
        total_timesteps=8,
        output_dir=tmp_path,
        num_envs=1,
        eval_episodes=1,
        algo_kwargs={
            "buffer_capacity": 32,
            "batch_size": 4,
            "learning_starts": 4,
            "train_frequency": 1,
            "unroll_steps": 2,
            "learning_rate": 1e-3,
            "gamma": 0.99,
            "num_simulations": 5,
            "latent_dim": 32,
            "action_embed_dim": 16,
            "dynamics_hidden_sizes": (32,),
            "prediction_hidden_sizes": (32,),
            "consistency_loss_weight": 0.5,
        },
        env_kwargs={
            "render_mode": "rgb_array",
            "wrappers": {
                "pixels": {
                    "resize_shape": [84, 84],
                    "frame_stack": 3,
                    "channel_first": True,
                }
            }
        },
    )


def _make_decision_transformer_config(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        algo="decision_transformer",
        env_id="Pendulum-v1",
        seed=70,
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


def _make_her_config(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        algo="her",
        env_id=POINT_GOAL_ENV_ID,
        seed=71,
        total_timesteps=32,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "buffer_capacity": 128,
            "batch_size": 16,
            "learning_starts": 8,
            "train_frequency": 1,
            "hidden_sizes": (32, 32),
            "learning_rate": 3e-4,
            "her_ratio": 0.8,
            "exploration_noise": 0.1,
            "eval_interval": 8,
        },
    )


def _make_appo_config(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        algo="appo",
        env_id="CartPole-v1",
        seed=67,
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


def _make_ppg_config(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        algo="ppg",
        env_id="CartPole-v1",
        seed=71,
        total_timesteps=64,
        output_dir=tmp_path,
        num_envs=2,
        eval_episodes=1,
        algo_kwargs={
            "num_steps": 32,
            "update_epochs": 1,
            "minibatch_size": 32,
            "hidden_sizes": (16, 16),
            "aux_frequency": 1,
            "aux_epochs": 1,
            "aux_minibatch_size": 32,
            "aux_buffer_rollouts": 2,
        },
    )


def _make_ars_config(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        algo="ars",
        env_id="Pendulum-v1",
        seed=89,
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


def _make_openai_es_config(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        algo="openai_es",
        env_id="Pendulum-v1",
        seed=97,
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


def _make_pets_config(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        algo="pets",
        env_id="Pendulum-v1",
        seed=101,
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


def _make_awr_config(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        algo="awr",
        env_id="Pendulum-v1",
        seed=103,
        total_timesteps=1,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 32,
            "dataset_seed": 33,
            "batch_size": 16,
            "hidden_sizes": (32, 32),
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "returns_to_go_gamma": 0.99,
            "beta": 1.0,
            "max_weight": 20.0,
            "normalize_advantages": True,
            "eval_interval": 1,
        },
    )


def _make_awac_config(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        algo="awac",
        env_id="Pendulum-v1",
        seed=107,
        total_timesteps=1,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 32,
            "dataset_seed": 19,
            "batch_size": 16,
            "hidden_sizes": (32, 32),
            "learning_rate": 3e-4,
            "awac_lambda": 1.0,
            "eval_interval": 1,
        },
    )


def _make_cal_ql_config(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        algo="cal_ql",
        env_id="Pendulum-v1",
        seed=109,
        total_timesteps=1,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 32,
            "dataset_seed": 27,
            "batch_size": 16,
            "hidden_sizes": (32, 32),
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "alpha": 0.2,
            "tau": 0.005,
            "cql_alpha": 5.0,
            "num_cql_samples": 10,
            "eval_interval": 1,
        },
    )


def _make_bc_config(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        algo="bc",
        env_id="Pendulum-v1",
        seed=113,
        total_timesteps=1,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 32,
            "dataset_seed": 11,
            "batch_size": 16,
            "hidden_sizes": (32, 32),
            "learning_rate": 3e-4,
            "eval_interval": 1,
        },
    )


def _make_bcq_config(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        algo="bcq",
        env_id="Pendulum-v1",
        seed=127,
        total_timesteps=1,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 32,
            "dataset_seed": 41,
            "batch_size": 16,
            "hidden_sizes": (32, 32),
            "latent_dim": 2,
            "num_action_samples": 10,
            "perturbation_scale": 0.05,
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "tau": 0.005,
            "vae_kl_weight": 0.5,
            "eval_interval": 1,
        },
    )


def _make_bear_config(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        algo="bear",
        env_id="Pendulum-v1",
        seed=131,
        total_timesteps=1,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 32,
            "dataset_seed": 43,
            "batch_size": 16,
            "hidden_sizes": (32, 32),
            "latent_dim": 2,
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "tau": 0.005,
            "behavior_kl_weight": 0.5,
            "mmd_sigma": 20.0,
            "mmd_alpha": 10.0,
            "num_mmd_action_samples": 10,
            "eval_interval": 1,
        },
    )


def _make_cql_config(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        algo="cql",
        env_id="Pendulum-v1",
        seed=137,
        total_timesteps=1,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 32,
            "dataset_seed": 17,
            "batch_size": 16,
            "hidden_sizes": (32, 32),
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "alpha": 0.2,
            "tau": 0.005,
            "cql_alpha": 5.0,
            "num_cql_samples": 10,
            "eval_interval": 1,
        },
    )


def _make_edac_config(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        algo="edac",
        env_id="Pendulum-v1",
        seed=139,
        total_timesteps=1,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 32,
            "dataset_seed": 29,
            "batch_size": 16,
            "hidden_sizes": (32, 32),
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "alpha": 0.2,
            "tau": 0.005,
            "num_critics": 5,
            "eta": 1.0,
            "eval_interval": 1,
        },
    )


def _make_rebrac_config(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        algo="rebrac",
        env_id="Pendulum-v1",
        seed=149,
        total_timesteps=1,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 32,
            "dataset_seed": 31,
            "batch_size": 16,
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
            "eval_interval": 1,
        },
    )


def _make_xql_config(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        algo="xql",
        env_id="Pendulum-v1",
        seed=151,
        total_timesteps=1,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 32,
            "dataset_seed": 37,
            "batch_size": 16,
            "hidden_sizes": (32, 32),
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "tau": 0.005,
            "beta": 3.0,
            "loss_temperature": 1.0,
            "max_advantage_weight": 20.0,
            "expectile": 0.7,
            "eval_interval": 1,
        },
    )


def _make_crr_config(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        algo="crr",
        env_id="Pendulum-v1",
        seed=157,
        total_timesteps=1,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 32,
            "dataset_seed": 21,
            "batch_size": 16,
            "hidden_sizes": (32, 32),
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "tau": 0.005,
            "beta": 1.0,
            "n_action_samples": 4,
            "max_weight": 20.0,
            "advantage_type": "mean",
            "weight_type": "exp",
            "eval_interval": 1,
        },
    )


def _make_marwil_config(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        algo="marwil",
        env_id="Pendulum-v1",
        seed=163,
        total_timesteps=1,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 32,
            "dataset_seed": 17,
            "batch_size": 16,
            "hidden_sizes": (32, 32),
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "beta": 1.0,
            "vf_coeff": 1.0,
            "moving_average_sqd_adv_norm_start": 100.0,
            "moving_average_sqd_adv_norm_update_rate": 0.05,
            "eval_interval": 1,
        },
    )


def _make_td3_bc_config(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        algo="td3_bc",
        env_id="Pendulum-v1",
        seed=167,
        total_timesteps=1,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 32,
            "dataset_seed": 19,
            "batch_size": 16,
            "hidden_sizes": (32, 32),
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "tau": 0.005,
            "policy_noise": 0.2,
            "noise_clip": 0.5,
            "policy_delay": 2,
            "bc_alpha": 2.5,
            "eval_interval": 1,
        },
    )


def _make_rlpd_config(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        algo="rlpd",
        env_id="Pendulum-v1",
        seed=173,
        total_timesteps=1,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 32,
            "dataset_seed": 41,
            "buffer_capacity": 64,
            "batch_size": 8,
            "learning_starts": 4,
            "train_frequency": 1,
            "gradient_updates_per_step": 1,
            "offline_pretrain_updates": 1,
            "offline_batch_ratio": 0.5,
            "hidden_sizes": (32, 32),
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "alpha": 0.2,
            "tau": 0.005,
            "eval_interval": 1,
        },
    )


def _make_agent57_config(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        algo="agent57",
        env_id="CartPole-v1",
        seed=179,
        total_timesteps=1,
        output_dir=tmp_path,
        device="cpu",
        num_envs=2,
        eval_episodes=1,
        algo_kwargs={
            "buffer_capacity": 32,
            "batch_size": 2,
            "learning_starts": 2,
            "train_frequency": 1,
            "target_update_interval": 4,
            "hidden_sizes": (16,),
            "head_hidden_sizes": (16,),
            "features_dim": 32,
            "recurrent_hidden_size": 32,
            "recurrent_num_layers": 1,
            "sequence_length": 4,
            "epsilon_start": 1.0,
            "epsilon_end": 0.05,
            "exploration_fraction": 0.2,
            "prioritized_alpha": 0.6,
            "prioritized_beta_start": 0.4,
            "prioritized_beta_end": 1.0,
            "prioritized_beta_fraction": 1.0,
            "priority_eta": 0.9,
            "n_step": 2,
            "intrinsic_reward_coef": 0.5,
            "rnd_learning_rate": 1e-3,
            "rnd_hidden_sizes": (32,),
            "rnd_embedding_dim": 32,
        },
    )


def _make_recurrent_ppo_config(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        algo="recurrent_ppo",
        env_id="CartPole-v1",
        seed=73,
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
            "encoder_hidden_sizes": (16,),
            "head_hidden_sizes": (16,),
            "features_dim": 32,
            "recurrent_hidden_size": 32,
            "recurrent_num_layers": 1,
        },
    )


def _make_drqn_config(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        algo="drqn",
        env_id="CartPole-v1",
        seed=79,
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
        },
    )


def _make_r2d2_config(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        algo="r2d2",
        env_id="CartPole-v1",
        seed=83,
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


def test_train_a2c_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(a2c_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(a2c_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        a2c_trainer.train_a2c(_make_a2c_config(tmp_path), run_suffix="session")


def test_train_trpo_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(trpo_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(trpo_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        trpo_trainer.train_trpo(_make_trpo_config(tmp_path), run_suffix="session")


def test_train_discrete_sac_uses_shared_training_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(discrete_sac_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(discrete_sac_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        discrete_sac_trainer.train_discrete_sac(_make_discrete_sac_config(tmp_path), run_suffix="session")


def test_train_ddpg_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ddpg_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(ddpg_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        ddpg_trainer.train_ddpg(_make_ddpg_config(tmp_path), run_suffix="session")


def test_train_td3_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(td3_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(td3_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        td3_trainer.train_td3(_make_td3_config(tmp_path), run_suffix="session")


def test_train_redq_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(redq_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(redq_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        redq_trainer.train_redq(_make_redq_config(tmp_path), run_suffix="session")


def test_train_crossq_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(crossq_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(crossq_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        crossq_trainer.train_crossq(_make_crossq_config(tmp_path), run_suffix="session")


def test_train_tqc_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tqc_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(tqc_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        tqc_trainer.train_tqc(_make_tqc_config(tmp_path), run_suffix="session")


def test_train_d4pg_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(d4pg_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(d4pg_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        d4pg_trainer.train_d4pg(_make_d4pg_config(tmp_path), run_suffix="session")


def test_train_naf_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(naf_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(naf_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        naf_trainer.train_naf(_make_naf_config(tmp_path), run_suffix="session")


def test_train_drq_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(drq_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(drq_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        drq_trainer.train_drq(_make_drq_config(tmp_path), run_suffix="session")


def test_train_drqv2_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(drqv2_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(drqv2_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        drqv2_trainer.train_drqv2(_make_drqv2_config(tmp_path), run_suffix="session")


def test_train_curl_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(curl_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(curl_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        curl_trainer.train_curl(_make_curl_config(tmp_path), run_suffix="session")


def test_train_impala_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(impala_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(impala_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        impala_trainer.train_impala(_make_impala_config(tmp_path), run_suffix="session")


def test_train_dreamer_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dreamer_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(dreamer_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        dreamer_trainer.train_dreamer(_make_dreamer_config(tmp_path), run_suffix="session")


def test_train_apex_dqn_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(apex_dqn_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(apex_dqn_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        apex_dqn_trainer.train_apex_dqn(_make_apex_dqn_config(tmp_path), run_suffix="session")


def test_train_iql_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(iql_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(iql_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        iql_trainer.train_iql(_make_iql_config(tmp_path), run_suffix="session")


def test_train_gail_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gail_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(gail_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        gail_trainer.train_gail(_make_gail_config(tmp_path), run_suffix="session")


def test_train_mbpo_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mbpo_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(mbpo_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        mbpo_trainer.train_mbpo(_make_mbpo_config(tmp_path), run_suffix="session")


def test_train_mopo_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mopo_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(mopo_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        mopo_trainer.train_mopo(_make_mopo_config(tmp_path), run_suffix="session")


def test_train_muzero_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(muzero_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(muzero_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        muzero_trainer.train_muzero(_make_muzero_config(tmp_path), run_suffix="session")


def test_train_efficientzero_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(efficientzero_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(efficientzero_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        efficientzero_trainer.train_efficientzero(_make_efficientzero_config(tmp_path), run_suffix="session")


def test_train_decision_transformer_uses_shared_training_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(decision_transformer_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(decision_transformer_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        decision_transformer_trainer.train_decision_transformer(
            _make_decision_transformer_config(tmp_path),
            run_suffix="session",
        )


def test_train_her_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(her_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(her_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        her_trainer.train_her(_make_her_config(tmp_path), run_suffix="session")


def test_train_appo_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(appo_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(appo_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        appo_trainer.train_appo(_make_appo_config(tmp_path), run_suffix="session")


def test_train_ppg_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ppg_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(ppg_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        ppg_trainer.train_ppg(_make_ppg_config(tmp_path), run_suffix="session")


def test_train_ars_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ars_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(ars_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        ars_trainer.train_ars(_make_ars_config(tmp_path), run_suffix="session")


def test_train_openai_es_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(openai_es_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(openai_es_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        openai_es_trainer.train_openai_es(_make_openai_es_config(tmp_path), run_suffix="session")


def test_train_pets_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pets_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(pets_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        pets_trainer.train_pets(_make_pets_config(tmp_path), run_suffix="session")


def test_train_awr_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(awr_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(awr_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        awr_trainer.train_awr(_make_awr_config(tmp_path), run_suffix="session")


def test_train_awac_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(awac_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(awac_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        awac_trainer.train_awac(_make_awac_config(tmp_path), run_suffix="session")


def test_train_cal_ql_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cal_ql_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(cal_ql_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        cal_ql_trainer.train_cal_ql(_make_cal_ql_config(tmp_path), run_suffix="session")


def test_train_bc_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bc_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(bc_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        bc_trainer.train_bc(_make_bc_config(tmp_path), run_suffix="session")


def test_train_bcq_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bcq_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(bcq_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        bcq_trainer.train_bcq(_make_bcq_config(tmp_path), run_suffix="session")


def test_train_bear_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bear_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(bear_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        bear_trainer.train_bear(_make_bear_config(tmp_path), run_suffix="session")


def test_train_cql_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cql_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(cql_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        cql_trainer.train_cql(_make_cql_config(tmp_path), run_suffix="session")


def test_train_edac_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(edac_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(edac_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        edac_trainer.train_edac(_make_edac_config(tmp_path), run_suffix="session")


def test_train_rebrac_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rebrac_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(rebrac_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        rebrac_trainer.train_rebrac(_make_rebrac_config(tmp_path), run_suffix="session")


def test_train_xql_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(xql_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(xql_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        xql_trainer.train_xql(_make_xql_config(tmp_path), run_suffix="session")


def test_train_crr_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(crr_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(crr_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        crr_trainer.train_crr(_make_crr_config(tmp_path), run_suffix="session")


def test_train_marwil_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(marwil_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(marwil_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        marwil_trainer.train_marwil(_make_marwil_config(tmp_path), run_suffix="session")


def test_train_td3_bc_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(td3_bc_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(td3_bc_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        td3_bc_trainer.train_td3_bc(_make_td3_bc_config(tmp_path), run_suffix="session")


def test_train_rlpd_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rlpd_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(rlpd_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        rlpd_trainer.train_rlpd(_make_rlpd_config(tmp_path), run_suffix="session")


def test_train_agent57_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent57_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(agent57_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        agent57_trainer.train_agent57(_make_agent57_config(tmp_path), run_suffix="session")


def test_train_recurrent_ppo_uses_shared_training_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(recurrent_ppo_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(recurrent_ppo_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        recurrent_ppo_trainer.train_recurrent_ppo(_make_recurrent_ppo_config(tmp_path), run_suffix="session")


def test_train_drqn_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(drqn_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(drqn_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        drqn_trainer.train_drqn(_make_drqn_config(tmp_path), run_suffix="session")


def test_train_r2d2_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(r2d2_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(r2d2_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        r2d2_trainer.train_r2d2(_make_r2d2_config(tmp_path), run_suffix="session")


def test_a2c_evaluation_uses_shared_evaluation_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(a2c_trainer, "evaluate_discrete_episodes", _raise_expected_eval_support, raising=False)
    monkeypatch.setattr(a2c_trainer, "build_env", _raise_legacy_eval_path, raising=False)

    policy = SimpleNamespace(
        act=lambda obs_tensor, deterministic=True: SimpleNamespace(actions=torch.tensor([0])),
    )

    with pytest.raises(ExpectedEvaluationSupport):
        a2c_trainer._evaluate_policy(
            policy,
            _make_a2c_config(tmp_path),
            device=torch.device("cpu"),
            num_episodes=1,
        )


def test_discrete_sac_evaluation_uses_shared_evaluation_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(discrete_sac_trainer, "evaluate_discrete_episodes", _raise_expected_eval_support, raising=False)
    monkeypatch.setattr(discrete_sac_trainer, "build_env", _raise_legacy_eval_path, raising=False)

    model = SimpleNamespace(
        sample_actions=lambda obs_tensor, deterministic=True: SimpleNamespace(actions=torch.tensor([0])),
    )

    with pytest.raises(ExpectedEvaluationSupport):
        discrete_sac_trainer._evaluate_discrete_sac_policy(
            model,
            _make_discrete_sac_config(tmp_path),
            device=torch.device("cpu"),
            num_episodes=1,
        )


def test_ddpg_evaluation_uses_shared_evaluation_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ddpg_trainer, "evaluate_continuous_episodes", _raise_expected_eval_support, raising=False)
    monkeypatch.setattr(ddpg_trainer, "build_env", _raise_legacy_eval_path, raising=False)

    model = SimpleNamespace(
        actor=lambda obs_tensor: torch.tensor([[0.0]]),
    )

    with pytest.raises(ExpectedEvaluationSupport):
        ddpg_trainer._evaluate_ddpg_policy(
            model,
            _make_ddpg_config(tmp_path),
            device=torch.device("cpu"),
            num_episodes=1,
        )


def test_td3_evaluation_uses_shared_evaluation_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(td3_trainer, "evaluate_continuous_episodes", _raise_expected_eval_support, raising=False)
    monkeypatch.setattr(td3_trainer, "build_env", _raise_legacy_eval_path, raising=False)

    model = SimpleNamespace(
        actor=lambda obs_tensor: torch.tensor([[0.0]]),
    )

    with pytest.raises(ExpectedEvaluationSupport):
        td3_trainer._evaluate_td3_policy(
            model,
            _make_td3_config(tmp_path),
            device=torch.device("cpu"),
            num_episodes=1,
        )


def test_redq_evaluation_uses_shared_evaluation_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(redq_trainer, "evaluate_continuous_episodes", _raise_expected_eval_support, raising=False)
    monkeypatch.setattr(redq_trainer, "build_env", _raise_legacy_eval_path, raising=False)

    model = SimpleNamespace(
        sample_actions=lambda obs_tensor, deterministic=True: SimpleNamespace(actions=torch.tensor([[0.0]])),
    )

    with pytest.raises(ExpectedEvaluationSupport):
        redq_trainer._evaluate_redq_policy(
            model,
            _make_redq_config(tmp_path),
            device=torch.device("cpu"),
            num_episodes=1,
        )


def test_sac_evaluation_uses_shared_evaluation_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sac_trainer, "evaluate_continuous_episodes", _raise_expected_eval_support, raising=False)
    monkeypatch.setattr(sac_trainer, "build_env", _raise_legacy_eval_path, raising=False)

    model = SimpleNamespace(
        sample_actions=lambda obs_tensor, deterministic=True: SimpleNamespace(actions=torch.tensor([[0.0]])),
    )

    with pytest.raises(ExpectedEvaluationSupport):
        sac_trainer._evaluate_sac_policy(
            model,
            _make_sac_config(tmp_path),
            device=torch.device("cpu"),
            num_episodes=1,
        )


def test_crossq_evaluation_uses_shared_evaluation_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(crossq_trainer, "evaluate_continuous_episodes", _raise_expected_eval_support, raising=False)
    monkeypatch.setattr(crossq_trainer, "build_env", _raise_legacy_eval_path, raising=False)

    model = SimpleNamespace(
        sample_actions=lambda obs_tensor, deterministic=True: SimpleNamespace(actions=torch.tensor([[0.0]])),
    )

    with pytest.raises(ExpectedEvaluationSupport):
        crossq_trainer._evaluate_crossq_policy(
            model,
            _make_crossq_config(tmp_path),
            device=torch.device("cpu"),
            num_episodes=1,
        )


def test_tqc_evaluation_uses_shared_evaluation_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tqc_trainer, "evaluate_continuous_episodes", _raise_expected_eval_support, raising=False)
    monkeypatch.setattr(tqc_trainer, "build_env", _raise_legacy_eval_path, raising=False)

    model = SimpleNamespace(
        sample_actions=lambda obs_tensor, deterministic=True: SimpleNamespace(actions=torch.tensor([[0.0]])),
    )

    with pytest.raises(ExpectedEvaluationSupport):
        tqc_trainer._evaluate_tqc_policy(
            model,
            _make_tqc_config(tmp_path),
            device=torch.device("cpu"),
            num_episodes=1,
        )


def test_d4pg_evaluation_uses_shared_evaluation_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(d4pg_trainer, "evaluate_continuous_episodes", _raise_expected_eval_support, raising=False)
    monkeypatch.setattr(d4pg_trainer, "build_env", _raise_legacy_eval_path, raising=False)

    model = SimpleNamespace(
        actor=lambda obs_tensor: torch.tensor([[0.0]]),
    )

    with pytest.raises(ExpectedEvaluationSupport):
        d4pg_trainer._evaluate_d4pg_policy(
            model,
            _make_d4pg_config(tmp_path),
            device=torch.device("cpu"),
            num_episodes=1,
        )


def test_naf_evaluation_uses_shared_evaluation_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(naf_trainer, "evaluate_continuous_episodes", _raise_expected_eval_support, raising=False)
    monkeypatch.setattr(naf_trainer, "build_env", _raise_legacy_eval_path, raising=False)

    model = SimpleNamespace(
        actor=lambda obs_tensor: torch.tensor([[0.0]]),
    )

    with pytest.raises(ExpectedEvaluationSupport):
        naf_trainer._evaluate_naf_policy(
            model,
            _make_naf_config(tmp_path),
            device=torch.device("cpu"),
            num_episodes=1,
        )


def test_drq_evaluation_uses_shared_evaluation_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(drq_trainer, "evaluate_continuous_episodes", _raise_expected_eval_support, raising=False)
    monkeypatch.setattr(drq_trainer, "build_env", _raise_legacy_eval_path, raising=False)

    model = SimpleNamespace(
        actor=lambda obs_tensor: torch.tensor([[0.0]]),
    )

    with pytest.raises(ExpectedEvaluationSupport):
        drq_trainer._evaluate_drq_policy(
            model,
            _make_drq_config(tmp_path),
            device=torch.device("cpu"),
            num_episodes=1,
        )


def test_drqv2_evaluation_uses_shared_evaluation_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(drqv2_trainer, "evaluate_continuous_episodes", _raise_expected_eval_support, raising=False)
    monkeypatch.setattr(drqv2_trainer, "build_env", _raise_legacy_eval_path, raising=False)

    model = SimpleNamespace(
        actor=lambda obs_tensor: torch.tensor([[0.0]]),
    )

    with pytest.raises(ExpectedEvaluationSupport):
        drqv2_trainer._evaluate_drqv2_policy(
            model,
            _make_drqv2_config(tmp_path),
            device=torch.device("cpu"),
            num_episodes=1,
        )


def test_curl_evaluation_uses_shared_evaluation_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(curl_trainer, "evaluate_continuous_episodes", _raise_expected_eval_support, raising=False)
    monkeypatch.setattr(curl_trainer, "build_env", _raise_legacy_eval_path, raising=False)

    model = SimpleNamespace(
        actor=lambda obs_tensor: torch.tensor([[0.0]]),
    )

    with pytest.raises(ExpectedEvaluationSupport):
        curl_trainer._evaluate_curl_policy(
            model,
            _make_curl_config(tmp_path),
            device=torch.device("cpu"),
            num_episodes=1,
        )


def test_impala_evaluation_uses_shared_evaluation_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(impala_trainer, "evaluate_discrete_episodes", _raise_expected_eval_support, raising=False)
    monkeypatch.setattr(impala_trainer, "build_env", _raise_legacy_eval_path, raising=False)

    policy = SimpleNamespace(
        act=lambda obs_tensor, deterministic=True: SimpleNamespace(actions=torch.tensor([0])),
    )

    with pytest.raises(ExpectedEvaluationSupport):
        impala_trainer._evaluate_impala_policy(
            policy,
            _make_impala_config(tmp_path),
            device=torch.device("cpu"),
            num_episodes=1,
        )


def test_dreamer_evaluation_uses_shared_evaluation_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dreamer_trainer, "evaluate_discrete_episodes", _raise_expected_eval_support, raising=False)
    monkeypatch.setattr(dreamer_trainer, "build_env", _raise_legacy_eval_path, raising=False)

    model = SimpleNamespace(
        act=lambda obs_tensor, deterministic=True: SimpleNamespace(actions=torch.tensor([[0]])),
    )

    with pytest.raises(ExpectedEvaluationSupport):
        dreamer_trainer._evaluate_policy(
            model,
            _make_dreamer_config(tmp_path),
            device=torch.device("cpu"),
            num_episodes=1,
        )


def test_apex_dqn_evaluation_delegates_to_shared_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(apex_dqn_trainer, "_evaluate_q_policy", _raise_expected_eval_support, raising=False)

    with pytest.raises(ExpectedEvaluationSupport):
        apex_dqn_trainer.train_apex_dqn(_make_apex_dqn_config(tmp_path), run_suffix="eval-helper")


def test_iql_evaluation_uses_shared_evaluation_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(iql_trainer, "evaluate_continuous_episodes", _raise_expected_eval_support, raising=False)
    monkeypatch.setattr(iql_trainer, "build_env", _raise_legacy_eval_path, raising=False)

    model = SimpleNamespace(
        sample_actions=lambda obs_tensor, deterministic=True: SimpleNamespace(actions=torch.tensor([[0.0]])),
    )

    with pytest.raises(ExpectedEvaluationSupport):
        iql_trainer._evaluate_iql_policy(
            model,
            _make_iql_config(tmp_path),
            device=torch.device("cpu"),
            num_episodes=1,
        )


def test_gail_evaluation_uses_shared_evaluation_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gail_trainer, "evaluate_discrete_episodes", _raise_expected_eval_support, raising=False)
    monkeypatch.setattr(gail_trainer, "build_env", _raise_legacy_eval_path, raising=False)

    policy = SimpleNamespace(
        act=lambda obs_tensor, deterministic=True: SimpleNamespace(actions=torch.tensor([[0]])),
    )

    with pytest.raises(ExpectedEvaluationSupport):
        gail_trainer._evaluate_policy(
            policy,
            _make_gail_config(tmp_path),
            device=torch.device("cpu"),
            num_episodes=1,
        )


def test_mbpo_evaluation_delegates_to_shared_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mbpo_trainer, "_evaluate_sac_policy", _raise_expected_eval_support, raising=False)

    with pytest.raises(ExpectedEvaluationSupport):
        mbpo_trainer.train_mbpo(_make_mbpo_config(tmp_path), run_suffix="eval-helper")


def test_mopo_evaluation_delegates_to_shared_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mopo_trainer, "_evaluate_sac_policy", _raise_expected_eval_support, raising=False)

    with pytest.raises(ExpectedEvaluationSupport):
        mopo_trainer.train_mopo(_make_mopo_config(tmp_path), run_suffix="eval-helper")


def test_muzero_evaluation_uses_shared_evaluation_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(muzero_trainer, "evaluate_discrete_episodes", _raise_expected_eval_support, raising=False)
    monkeypatch.setattr(muzero_trainer, "build_env", _raise_legacy_eval_path, raising=False)

    algorithm = SimpleNamespace(
        set_eval_mode=lambda: None,
        act=lambda obs, deterministic=True: SimpleNamespace(actions=torch.tensor([0])),
    )

    with pytest.raises(ExpectedEvaluationSupport):
        muzero_trainer._evaluate_muzero_policy(
            algorithm,
            _make_muzero_config(tmp_path),
            device=torch.device("cpu"),
            num_episodes=1,
        )


def test_decision_transformer_evaluation_uses_shared_evaluation_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        decision_transformer_trainer,
        "evaluate_continuous_episodes",
        _raise_expected_eval_support,
        raising=False,
    )
    monkeypatch.setattr(decision_transformer_trainer, "build_env", _raise_legacy_eval_path, raising=False)

    model = SimpleNamespace(
        predict_last_action=lambda **kwargs: torch.tensor([[0.0]]),
    )

    with pytest.raises(ExpectedEvaluationSupport):
        decision_transformer_trainer._evaluate_decision_transformer_policy(
            model,
            _make_decision_transformer_config(tmp_path),
            device=torch.device("cpu"),
            num_episodes=1,
            context_length=4,
            target_return=0.0,
            max_timestep=64,
            gamma=0.99,
        )


def test_her_evaluation_uses_shared_evaluation_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(her_trainer, "evaluate_continuous_episodes", _raise_expected_eval_support, raising=False)
    monkeypatch.setattr(her_trainer, "build_env", _raise_legacy_eval_path, raising=False)

    model = SimpleNamespace(
        actor=lambda obs_tensor: torch.tensor([[0.0]]),
    )

    with pytest.raises(ExpectedEvaluationSupport):
        her_trainer._evaluate_her_policy(
            model,
            _make_her_config(tmp_path),
            device=torch.device("cpu"),
            num_episodes=1,
        )


def test_efficientzero_evaluation_delegates_to_shared_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(efficientzero_trainer, "_maybe_run_muzero_evaluation", _raise_expected_eval_support, raising=False)

    with pytest.raises(ExpectedEvaluationSupport):
        efficientzero_trainer.train_efficientzero(_make_efficientzero_config(tmp_path), run_suffix="eval-helper")


def test_appo_evaluation_uses_shared_evaluation_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(appo_trainer, "evaluate_discrete_episodes", _raise_expected_eval_support, raising=False)
    monkeypatch.setattr(appo_trainer, "build_env", _raise_legacy_eval_path, raising=False)

    policy = SimpleNamespace(
        act=lambda obs_tensor, deterministic=True: SimpleNamespace(actions=torch.tensor([0])),
    )

    with pytest.raises(ExpectedEvaluationSupport):
        appo_trainer._evaluate_appo_policy(
            policy,
            _make_appo_config(tmp_path),
            device=torch.device("cpu"),
            num_episodes=1,
        )


def test_ppg_evaluation_uses_shared_evaluation_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ppg_trainer, "evaluate_discrete_episodes", _raise_expected_eval_support, raising=False)
    monkeypatch.setattr(ppg_trainer, "build_env", _raise_legacy_eval_path, raising=False)

    model = SimpleNamespace(
        act=lambda obs_tensor, deterministic=True: SimpleNamespace(actions=torch.tensor([0])),
    )

    with pytest.raises(ExpectedEvaluationSupport):
        ppg_trainer._evaluate_ppg_policy(
            model,
            _make_ppg_config(tmp_path),
            device=torch.device("cpu"),
            num_episodes=1,
        )


def test_ars_evaluation_uses_shared_evaluation_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ars_trainer, "evaluate_continuous_episodes", _raise_expected_eval_support, raising=False)
    monkeypatch.setattr(ars_trainer, "build_env", _raise_legacy_eval_path, raising=False)

    model = SimpleNamespace(
        actor=lambda obs_tensor: torch.tensor([[0.0]], dtype=torch.float32),
    )

    with pytest.raises(ExpectedEvaluationSupport):
        ars_trainer._evaluate_ars_policy(
            model,
            _make_ars_config(tmp_path),
            device=torch.device("cpu"),
            num_episodes=1,
        )


def test_openai_es_evaluation_uses_shared_evaluation_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(openai_es_trainer, "evaluate_continuous_episodes", _raise_expected_eval_support, raising=False)
    monkeypatch.setattr(openai_es_trainer, "build_env", _raise_legacy_eval_path, raising=False)

    model = SimpleNamespace(
        actor=lambda obs_tensor: torch.tensor([[0.0]], dtype=torch.float32),
    )

    with pytest.raises(ExpectedEvaluationSupport):
        openai_es_trainer._evaluate_openai_es_policy(
            model,
            _make_openai_es_config(tmp_path),
            device=torch.device("cpu"),
            num_episodes=1,
        )


def test_pets_evaluation_uses_shared_evaluation_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pets_trainer, "evaluate_continuous_episodes", _raise_expected_eval_support, raising=False)
    monkeypatch.setattr(pets_trainer, "build_env", _raise_legacy_eval_path, raising=False)

    algorithm = SimpleNamespace(
        set_eval_mode=lambda: None,
        plan_action=lambda *args, **kwargs: np.zeros((1,), dtype=np.float32),
    )

    with pytest.raises(ExpectedEvaluationSupport):
        pets_trainer._evaluate_pets_policy(
            algorithm,
            _make_pets_config(tmp_path),
            device=torch.device("cpu"),
            num_episodes=1,
        )


def test_awr_evaluation_delegates_to_shared_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(awr_trainer, "_evaluate_iql_policy", _raise_expected_eval_support, raising=False)

    with pytest.raises(ExpectedEvaluationSupport):
        awr_trainer.train_awr(_make_awr_config(tmp_path), run_suffix="eval-helper")


def test_awac_evaluation_delegates_to_shared_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(awac_trainer, "_evaluate_sac_policy", _raise_expected_eval_support, raising=False)

    with pytest.raises(ExpectedEvaluationSupport):
        awac_trainer.train_awac(_make_awac_config(tmp_path), run_suffix="eval-helper")


def test_cal_ql_evaluation_delegates_to_shared_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cal_ql_trainer, "_evaluate_sac_policy", _raise_expected_eval_support, raising=False)

    with pytest.raises(ExpectedEvaluationSupport):
        cal_ql_trainer.train_cal_ql(_make_cal_ql_config(tmp_path), run_suffix="eval-helper")


def test_bc_evaluation_uses_shared_evaluation_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bc_trainer, "evaluate_continuous_episodes", _raise_expected_eval_support, raising=False)
    monkeypatch.setattr(bc_trainer, "build_env", _raise_legacy_eval_path, raising=False)

    model = SimpleNamespace(
        actor=lambda obs_tensor: torch.tensor([[0.0]], dtype=torch.float32),
    )

    with pytest.raises(ExpectedEvaluationSupport):
        bc_trainer._evaluate_bc_policy(
            model,
            _make_bc_config(tmp_path),
            device=torch.device("cpu"),
            num_episodes=1,
        )


def test_bcq_evaluation_uses_shared_evaluation_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bcq_trainer, "evaluate_continuous_episodes", _raise_expected_eval_support, raising=False)
    monkeypatch.setattr(bcq_trainer, "build_env", _raise_legacy_eval_path, raising=False)

    model = SimpleNamespace(
        select_actions=lambda obs_tensor, num_action_samples, deterministic=True: torch.tensor([[0.0]], dtype=torch.float32),
    )

    with pytest.raises(ExpectedEvaluationSupport):
        bcq_trainer._evaluate_bcq_policy(
            model,
            _make_bcq_config(tmp_path),
            device=torch.device("cpu"),
            num_episodes=1,
            num_action_samples=10,
        )


def test_bear_evaluation_delegates_to_shared_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bear_trainer, "_evaluate_sac_policy", _raise_expected_eval_support, raising=False)

    with pytest.raises(ExpectedEvaluationSupport):
        bear_trainer.train_bear(_make_bear_config(tmp_path), run_suffix="eval-helper")


def test_cql_evaluation_delegates_to_shared_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cql_trainer, "_evaluate_sac_policy", _raise_expected_eval_support, raising=False)

    with pytest.raises(ExpectedEvaluationSupport):
        cql_trainer.train_cql(_make_cql_config(tmp_path), run_suffix="eval-helper")


def test_edac_evaluation_delegates_to_shared_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(edac_trainer, "_evaluate_redq_policy", _raise_expected_eval_support, raising=False)

    with pytest.raises(ExpectedEvaluationSupport):
        edac_trainer.train_edac(_make_edac_config(tmp_path), run_suffix="eval-helper")


def test_rebrac_evaluation_delegates_to_shared_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rebrac_trainer, "_evaluate_td3_policy", _raise_expected_eval_support, raising=False)

    with pytest.raises(ExpectedEvaluationSupport):
        rebrac_trainer.train_rebrac(_make_rebrac_config(tmp_path), run_suffix="eval-helper")


def test_xql_evaluation_delegates_to_shared_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(xql_trainer, "_evaluate_iql_policy", _raise_expected_eval_support, raising=False)

    with pytest.raises(ExpectedEvaluationSupport):
        xql_trainer.train_xql(_make_xql_config(tmp_path), run_suffix="eval-helper")


def test_crr_evaluation_delegates_to_shared_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(crr_trainer, "_evaluate_sac_policy", _raise_expected_eval_support, raising=False)

    with pytest.raises(ExpectedEvaluationSupport):
        crr_trainer.train_crr(_make_crr_config(tmp_path), run_suffix="eval-helper")


def test_marwil_evaluation_delegates_to_shared_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(marwil_trainer, "_evaluate_iql_policy", _raise_expected_eval_support, raising=False)

    with pytest.raises(ExpectedEvaluationSupport):
        marwil_trainer.train_marwil(_make_marwil_config(tmp_path), run_suffix="eval-helper")


def test_td3_bc_evaluation_delegates_to_shared_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(td3_bc_trainer, "_evaluate_td3_policy", _raise_expected_eval_support, raising=False)

    with pytest.raises(ExpectedEvaluationSupport):
        td3_bc_trainer.train_td3_bc(_make_td3_bc_config(tmp_path), run_suffix="eval-helper")


def test_rlpd_evaluation_delegates_to_shared_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rlpd_trainer, "_evaluate_sac_policy", _raise_expected_eval_support, raising=False)

    with pytest.raises(ExpectedEvaluationSupport):
        rlpd_trainer.train_rlpd(_make_rlpd_config(tmp_path), run_suffix="eval-helper")


def test_agent57_evaluation_delegates_to_shared_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agent57_trainer, "_maybe_run_r2d2_evaluation", _raise_expected_eval_support, raising=False)

    with pytest.raises(ExpectedEvaluationSupport):
        agent57_trainer.train_agent57(_make_agent57_config(tmp_path), run_suffix="eval-helper")


def test_recurrent_ppo_evaluation_uses_shared_evaluation_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(recurrent_ppo_trainer, "evaluate_discrete_episodes", _raise_expected_eval_support, raising=False)
    monkeypatch.setattr(recurrent_ppo_trainer, "build_env", _raise_legacy_eval_path, raising=False)

    policy = SimpleNamespace(
        initial_state=lambda batch_size, device=None: ("state", "cell"),
        act=lambda obs_tensor, state=None, deterministic=True: SimpleNamespace(
            actions=torch.tensor([0]),
            state=state,
        ),
    )

    with pytest.raises(ExpectedEvaluationSupport):
        recurrent_ppo_trainer._evaluate_recurrent_policy(
            policy,
            _make_recurrent_ppo_config(tmp_path),
            device=torch.device("cpu"),
            num_episodes=1,
        )


def test_drqn_evaluation_uses_shared_evaluation_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(drqn_trainer, "evaluate_discrete_episodes", _raise_expected_eval_support, raising=False)
    monkeypatch.setattr(drqn_trainer, "build_env", _raise_legacy_eval_path, raising=False)

    q_network = SimpleNamespace(
        initial_state=lambda batch_size, device=None: ("state", "cell"),
        act=lambda obs_tensor, state=None, epsilon=0.0, deterministic=True, episode_starts=None: SimpleNamespace(
            actions=torch.tensor([0]),
            state=state,
        ),
    )

    with pytest.raises(ExpectedEvaluationSupport):
        drqn_trainer._evaluate_drqn_policy(
            q_network,
            _make_drqn_config(tmp_path),
            device=torch.device("cpu"),
            num_episodes=1,
        )


def test_r2d2_evaluation_uses_shared_evaluation_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(r2d2_trainer, "evaluate_discrete_episodes", _raise_expected_eval_support, raising=False)
    monkeypatch.setattr(r2d2_trainer, "build_env", _raise_legacy_eval_path, raising=False)

    q_network = SimpleNamespace(
        initial_state=lambda batch_size, device=None: ("state", "cell"),
        act=lambda obs_tensor, state=None, epsilon=0.0, deterministic=True, episode_starts=None: SimpleNamespace(
            actions=torch.tensor([0]),
            state=state,
        ),
    )

    with pytest.raises(ExpectedEvaluationSupport):
        r2d2_trainer._evaluate_r2d2_policy(
            q_network,
            _make_r2d2_config(tmp_path),
            device=torch.device("cpu"),
            num_episodes=1,
        )
