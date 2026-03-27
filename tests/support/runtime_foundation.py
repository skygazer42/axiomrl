from __future__ import annotations

from pathlib import Path

from rl_training.envs import POINT_GOAL_ENV_ID
from rl_training.experiment.config import TrainConfig


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
            },
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
            },
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
