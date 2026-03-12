from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np
import torch

from rl_training.algorithms.a2c import A2C as A2CAlgorithm
from rl_training.algorithms.awr import AWR as AWRAlgorithm
from rl_training.algorithms.awac import AWAC as AWACAlgorithm
from rl_training.algorithms.marwil import MARWIL as MARWILAlgorithm
from rl_training.algorithms.bc import BC as BCAlgorithm
from rl_training.algorithms.bcq import BCQ as BCQAlgorithm
from rl_training.algorithms.bear import BEAR as BEARAlgorithm
from rl_training.algorithms.cal_ql import CalQL as CalQLAlgorithm
from rl_training.algorithms.c51_dqn import C51DQN as C51DQNAlgorithm
from rl_training.algorithms.crossq import CrossQ as CrossQAlgorithm
from rl_training.algorithms.crr import CRR as CRRAlgorithm
from rl_training.algorithms.cql import CQL as CQLAlgorithm
from rl_training.algorithms.ddpg import DDPG as DDPGAlgorithm
from rl_training.algorithms.edac import EDAC as EDACAlgorithm
from rl_training.algorithms.drqv2 import DrQv2 as DrQv2Algorithm
from rl_training.algorithms.discrete_sac import DiscreteSAC as DiscreteSACAlgorithm
from rl_training.algorithms.dqn import DQN as DQNAlgorithm
from rl_training.algorithms.dqn import DoubleDQN as DoubleDQNAlgorithm
from rl_training.algorithms.dqn import DuelingDQN as DuelingDQNAlgorithm
from rl_training.algorithms.her import HER as HERAlgorithm
from rl_training.algorithms.iql import IQL as IQLAlgorithm
from rl_training.algorithms.iqn import IQN as IQNAlgorithm
from rl_training.algorithms.xql import XQL as XQLAlgorithm
from rl_training.algorithms.rlpd import RLPD as RLPDAlgorithm
from rl_training.algorithms.dqn import NoisyDQN as NoisyDQNAlgorithm
from rl_training.algorithms.dqn import PrioritizedDQN as PrioritizedDQNAlgorithm
from rl_training.algorithms.dqn import RainbowDQN as RainbowDQNAlgorithm
from rl_training.algorithms.ppo import PPO as PPOAlgorithm
from rl_training.algorithms.qr_dqn import QRDQN as QRDQNAlgorithm
from rl_training.algorithms.redq import REDQ as REDQAlgorithm
from rl_training.algorithms.rebrac import ReBRAC as ReBRACAlgorithm
from rl_training.algorithms.sac import SAC as SACAlgorithm
from rl_training.algorithms.trpo import TRPO as TRPOAlgorithm
from rl_training.algorithms.tqc import TQC as TQCAlgorithm
from rl_training.algorithms.td3 import TD3 as TD3Algorithm
from rl_training.algorithms.td3_bc import TD3BC as TD3BCAlgorithm
from rl_training.contrib.recurrent_ppo import RecurrentPPOAlgorithm
from rl_training.experiment.checkpointing import CheckpointState
from rl_training.experiment.config import TrainConfig
from rl_training.envs.factory import build_env
from rl_training.envs.goals import flatten_goal_observation
from rl_training.models.cnn import CNNActorCritic, CNNDrQv2Model, CNNQNetwork
from rl_training.models.mlp_actor_critic import MLPActorCritic
from rl_training.models.mlp_bc import MLPBCModel
from rl_training.models.mlp_bcq import MLPBCQModel
from rl_training.models.mlp_bear import MLPBEARModel
from rl_training.models.mlp_c51_q_network import MLPC51QNetwork
from rl_training.models.mlp_crossq import MLPCrossQModel
from rl_training.models.mlp_ddpg import MLPDDPGModel
from rl_training.models.mlp_discrete_sac import MLPDiscreteSACModel
from rl_training.models.mlp_dueling_noisy_q_network import MLPDuelingNoisyQNetwork
from rl_training.models.mlp_dueling_q_network import MLPDuelingQNetwork
from rl_training.models.mlp_iql import MLPIQLModel
from rl_training.models.mlp_iqn_network import MLPIQNetwork
from rl_training.models.mlp_noisy_q_network import MLPNoisyQNetwork
from rl_training.models.mlp_q_network import MLPQNetwork
from rl_training.models.mlp_qr_q_network import MLPQRQNetwork
from rl_training.models.mlp_redq import MLPREDQModel
from rl_training.models.mlp_sac import MLPSACModel
from rl_training.models.mlp_tqc import MLPTQCModel
from rl_training.models.mlp_td3 import MLPTD3Model
from rl_training.models.recurrent import LSTMActorCritic
from rl_training.runtime.a2c_trainer import _evaluate_policy as _evaluate_a2c_policy
from rl_training.runtime.a2c_trainer import train_a2c
from rl_training.runtime.awr_trainer import train_awr
from rl_training.runtime.awac_trainer import train_awac
from rl_training.runtime.marwil_trainer import train_marwil
from rl_training.runtime.bc_trainer import _evaluate_bc_policy, train_bc
from rl_training.runtime.bcq_trainer import _evaluate_bcq_policy, train_bcq
from rl_training.runtime.bear_trainer import train_bear
from rl_training.runtime.cal_ql_trainer import train_cal_ql
from rl_training.runtime.cql_trainer import train_cql
from rl_training.runtime.crossq_trainer import _evaluate_crossq_policy, train_crossq
from rl_training.runtime.crr_trainer import train_crr
from rl_training.runtime.ddpg_trainer import _evaluate_ddpg_policy, train_ddpg
from rl_training.runtime.edac_trainer import train_edac
from rl_training.runtime.drqv2_trainer import _evaluate_drqv2_policy, train_drqv2
from rl_training.runtime.discrete_sac_trainer import _evaluate_discrete_sac_policy, train_discrete_sac
from rl_training.runtime.dqn_trainer import _evaluate_q_policy, train_dqn
from rl_training.runtime.her_trainer import _evaluate_her_policy, _infer_her_spaces, train_her
from rl_training.runtime.iql_trainer import _evaluate_iql_policy, train_iql
from rl_training.runtime.xql_trainer import train_xql
from rl_training.runtime.ppo_trainer import _evaluate_policy, train_ppo
from rl_training.runtime.recurrent_ppo_trainer import _evaluate_recurrent_policy, train_recurrent_ppo
from rl_training.runtime.redq_trainer import _evaluate_redq_policy, train_redq
from rl_training.runtime.rlpd_trainer import train_rlpd
from rl_training.runtime.rebrac_trainer import train_rebrac
from rl_training.runtime.sac_trainer import _evaluate_sac_policy, train_sac
from rl_training.runtime.tqc_trainer import _evaluate_tqc_policy, train_tqc
from rl_training.runtime.td3_trainer import _evaluate_td3_policy, train_td3
from rl_training.runtime.td3_bc_trainer import train_td3_bc
from rl_training.runtime.trpo_trainer import train_trpo
from rl_training.runtime.trainer import TrainResult
from rl_training.runtime.types import MetricDict


TrainFn = Callable[..., TrainResult]
EvaluateFn = Callable[[TrainConfig, CheckpointState, torch.device, int], MetricDict]
PredictFn = Callable[[TrainConfig, CheckpointState, torch.device, object, bool], int | np.ndarray]


@dataclass(frozen=True, slots=True)
class AlgorithmSpec:
    name: str
    train_fn: TrainFn
    evaluate_fn: EvaluateFn
    predict_fn: PredictFn


def _infer_discrete_env_spaces(config: TrainConfig) -> tuple[tuple[int, ...], int]:
    env = build_env(config, 0)
    try:
        obs_space = env.observation_space
        action_space = env.action_space
        if not isinstance(obs_space, gym.spaces.Box):
            raise TypeError(f"unsupported observation space: {type(obs_space)!r}")
        if not isinstance(action_space, gym.spaces.Discrete):
            raise TypeError(f"unsupported action space: {type(action_space)!r}")
        if obs_space.shape is None or len(obs_space.shape) not in (1, 3):
            raise ValueError(f"expected flat 1D or channel-first image observations, got shape={obs_space.shape!r}")
        return tuple(int(dim) for dim in obs_space.shape), int(action_space.n)
    finally:
        env.close()


def _infer_continuous_env_spaces(config: TrainConfig) -> tuple[int, int]:
    env = build_env(config, 0)
    try:
        obs_space = env.observation_space
        action_space = env.action_space
        if not isinstance(obs_space, gym.spaces.Box):
            raise TypeError(f"unsupported observation space: {type(obs_space)!r}")
        if not isinstance(action_space, gym.spaces.Box):
            raise TypeError(f"unsupported action space: {type(action_space)!r}")
        if obs_space.shape is None or len(obs_space.shape) != 1:
            raise ValueError(f"expected flat 1D observations, got shape={obs_space.shape!r}")
        if action_space.shape is None or len(action_space.shape) != 1:
            raise ValueError(f"expected flat 1D actions, got shape={action_space.shape!r}")
        return int(obs_space.shape[0]), int(action_space.shape[0])
    finally:
        env.close()


def _infer_image_continuous_env_spaces(config: TrainConfig) -> tuple[tuple[int, ...], int]:
    env = build_env(config, 0)
    try:
        obs_space = env.observation_space
        action_space = env.action_space
        if not isinstance(obs_space, gym.spaces.Box):
            raise TypeError(f"unsupported observation space: {type(obs_space)!r}")
        if not isinstance(action_space, gym.spaces.Box):
            raise TypeError(f"unsupported action space: {type(action_space)!r}")
        if obs_space.shape is None or len(obs_space.shape) != 3:
            raise ValueError(f"expected channel-first image observations, got shape={obs_space.shape!r}")
        if action_space.shape is None or len(action_space.shape) != 1:
            raise ValueError(f"expected flat 1D actions, got shape={action_space.shape!r}")
        return tuple(int(dim) for dim in obs_space.shape), int(action_space.shape[0])
    finally:
        env.close()


def _continuous_action_bounds(config: TrainConfig, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    env = build_env(config, 0, evaluation=True)
    try:
        action_space = env.action_space
        if not isinstance(action_space, gym.spaces.Box):
            raise TypeError(f"unsupported action space: {type(action_space)!r}")
        low = torch.as_tensor(action_space.low, dtype=torch.float32, device=device)
        high = torch.as_tensor(action_space.high, dtype=torch.float32, device=device)
        return low, high
    finally:
        env.close()


def _scale_continuous_actions(
    normalized_actions: torch.Tensor,
    *,
    low: torch.Tensor,
    high: torch.Tensor,
) -> torch.Tensor:
    scaled = low + 0.5 * (normalized_actions + 1.0) * (high - low)
    return torch.max(torch.min(scaled, high), low)


def _prepare_observation(obs: object, *, device: torch.device) -> torch.Tensor:
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
    if obs_tensor.ndim in (1, 3):
        obs_tensor = obs_tensor.unsqueeze(0)
    return obs_tensor


def _format_action_output(actions: torch.Tensor, *, discrete: bool) -> int | np.ndarray:
    action_tensor = actions.detach().cpu()
    if action_tensor.ndim > 1 and action_tensor.shape[0] == 1:
        action_tensor = action_tensor.squeeze(0)
    if discrete:
        if action_tensor.ndim == 0:
            return int(action_tensor.item())
        if action_tensor.numel() == 1:
            return int(action_tensor.reshape(-1)[0].item())
        return action_tensor.numpy()
    return action_tensor.numpy()


def _load_a2c_algorithm(config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device) -> A2CAlgorithm:
    obs_shape, action_dim = _infer_discrete_env_spaces(config)
    if len(obs_shape) != 1:
        raise ValueError("a2c checkpoint loading currently supports flat observations only")
    obs_dim = obs_shape[0]
    hidden_sizes = tuple(config.algo_kwargs.get("hidden_sizes", (64, 64)))
    algorithm = A2CAlgorithm(
        policy=MLPActorCritic(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=hidden_sizes).to(device),
        learning_rate=float(config.algo_kwargs.get("learning_rate", 3e-4)),
        ent_coef=float(config.algo_kwargs.get("ent_coef", 0.01)),
        vf_coef=float(config.algo_kwargs.get("vf_coef", 0.5)),
        max_grad_norm=float(config.algo_kwargs.get("max_grad_norm", 0.5)),
    )
    algorithm.load_state_dict(checkpoint_state.algorithm_state)
    return algorithm


def _load_bc_algorithm(config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device) -> BCAlgorithm:
    obs_dim, action_dim = _infer_continuous_env_spaces(config)
    hidden_sizes = tuple(config.algo_kwargs.get("hidden_sizes", (256, 256)))
    algorithm = BCAlgorithm(
        model=MLPBCModel(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=hidden_sizes).to(device),
        learning_rate=float(config.algo_kwargs.get("learning_rate", 3e-4)),
    )
    algorithm.load_state_dict(checkpoint_state.algorithm_state)
    return algorithm


def _load_bcq_algorithm(config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device) -> BCQAlgorithm:
    obs_dim, action_dim = _infer_continuous_env_spaces(config)
    hidden_sizes = tuple(config.algo_kwargs.get("hidden_sizes", (256, 256)))
    latent_dim = int(config.algo_kwargs.get("latent_dim", action_dim * 2))
    num_action_samples = int(config.algo_kwargs.get("num_action_samples", 10))
    algorithm = BCQAlgorithm(
        model=MLPBCQModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            latent_dim=latent_dim,
            hidden_sizes=hidden_sizes,
            perturbation_scale=float(config.algo_kwargs.get("perturbation_scale", 0.05)),
            num_action_samples=num_action_samples,
        ).to(device),
        learning_rate=float(config.algo_kwargs.get("learning_rate", 3e-4)),
        gamma=float(config.algo_kwargs.get("gamma", 0.99)),
        tau=float(config.algo_kwargs.get("tau", 0.005)),
        num_action_samples=num_action_samples,
        vae_kl_weight=float(config.algo_kwargs.get("vae_kl_weight", 0.5)),
    )
    algorithm.load_state_dict(checkpoint_state.algorithm_state)
    return algorithm


def _load_bear_algorithm(config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device) -> BEARAlgorithm:
    obs_dim, action_dim = _infer_continuous_env_spaces(config)
    hidden_sizes = tuple(config.algo_kwargs.get("hidden_sizes", (256, 256)))
    algorithm = BEARAlgorithm(
        model=MLPBEARModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            latent_dim=int(config.algo_kwargs.get("latent_dim", action_dim * 2)),
            hidden_sizes=hidden_sizes,
        ).to(device),
        learning_rate=float(config.algo_kwargs.get("learning_rate", 3e-4)),
        gamma=float(config.algo_kwargs.get("gamma", 0.99)),
        tau=float(config.algo_kwargs.get("tau", 0.005)),
        behavior_kl_weight=float(config.algo_kwargs.get("behavior_kl_weight", 0.5)),
        mmd_sigma=float(config.algo_kwargs.get("mmd_sigma", 20.0)),
        mmd_alpha=float(config.algo_kwargs.get("mmd_alpha", 10.0)),
        num_mmd_action_samples=int(config.algo_kwargs.get("num_mmd_action_samples", 10)),
    )
    algorithm.load_state_dict(checkpoint_state.algorithm_state)
    return algorithm


def _load_awac_algorithm(config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device) -> AWACAlgorithm:
    obs_dim, action_dim = _infer_continuous_env_spaces(config)
    hidden_sizes = tuple(config.algo_kwargs.get("hidden_sizes", (256, 256)))
    algorithm = AWACAlgorithm(
        model=MLPSACModel(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=hidden_sizes).to(device),
        learning_rate=float(config.algo_kwargs.get("learning_rate", 3e-4)),
        gamma=float(config.algo_kwargs.get("gamma", 0.99)),
        alpha=float(config.algo_kwargs.get("alpha", 0.2)),
        tau=float(config.algo_kwargs.get("tau", 0.005)),
        awac_lambda=float(config.algo_kwargs.get("awac_lambda", 1.0)),
        max_advantage_weight=float(config.algo_kwargs.get("max_advantage_weight", 20.0)),
    )
    algorithm.load_state_dict(checkpoint_state.algorithm_state)
    return algorithm


def _load_crr_algorithm(config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device) -> CRRAlgorithm:
    obs_dim, action_dim = _infer_continuous_env_spaces(config)
    hidden_sizes = tuple(config.algo_kwargs.get("hidden_sizes", (256, 256)))
    algorithm = CRRAlgorithm(
        model=MLPSACModel(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=hidden_sizes).to(device),
        learning_rate=float(config.algo_kwargs.get("learning_rate", 3e-4)),
        gamma=float(config.algo_kwargs.get("gamma", 0.99)),
        tau=float(config.algo_kwargs.get("tau", 0.005)),
        beta=float(config.algo_kwargs.get("beta", 1.0)),
        n_action_samples=int(config.algo_kwargs.get("n_action_samples", 4)),
        max_weight=float(config.algo_kwargs.get("max_weight", 20.0)),
        advantage_type=str(config.algo_kwargs.get("advantage_type", "mean")),
        weight_type=str(config.algo_kwargs.get("weight_type", "exp")),
    )
    algorithm.load_state_dict(checkpoint_state.algorithm_state)
    return algorithm


def _load_cal_ql_algorithm(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    *,
    device: torch.device,
) -> CalQLAlgorithm:
    obs_dim, action_dim = _infer_continuous_env_spaces(config)
    hidden_sizes = tuple(config.algo_kwargs.get("hidden_sizes", (256, 256)))
    algorithm = CalQLAlgorithm(
        model=MLPSACModel(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=hidden_sizes).to(device),
        learning_rate=float(config.algo_kwargs.get("learning_rate", 3e-4)),
        gamma=float(config.algo_kwargs.get("gamma", 0.99)),
        alpha=float(config.algo_kwargs.get("alpha", 0.2)),
        tau=float(config.algo_kwargs.get("tau", 0.005)),
        cql_alpha=float(config.algo_kwargs.get("cql_alpha", 5.0)),
        num_cql_samples=int(config.algo_kwargs.get("num_cql_samples", 10)),
    )
    algorithm.load_state_dict(checkpoint_state.algorithm_state)
    return algorithm


def _load_xql_algorithm(config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device) -> XQLAlgorithm:
    obs_dim, action_dim = _infer_continuous_env_spaces(config)
    hidden_sizes = tuple(config.algo_kwargs.get("hidden_sizes", (256, 256)))
    algorithm = XQLAlgorithm(
        model=MLPIQLModel(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=hidden_sizes).to(device),
        learning_rate=float(config.algo_kwargs.get("learning_rate", 3e-4)),
        gamma=float(config.algo_kwargs.get("gamma", 0.99)),
        tau=float(config.algo_kwargs.get("tau", 0.005)),
        beta=float(config.algo_kwargs.get("beta", 3.0)),
        loss_temperature=float(config.algo_kwargs.get("loss_temperature", 1.0)),
        max_advantage_weight=float(config.algo_kwargs.get("max_advantage_weight", 100.0)),
        vanilla_value_loss=bool(config.algo_kwargs.get("vanilla_value_loss", False)),
        expectile=float(config.algo_kwargs.get("expectile", 0.7)),
        max_value_diff_exp=(
            None
            if config.algo_kwargs.get("max_value_diff_exp", 5.0) is None
            else float(config.algo_kwargs.get("max_value_diff_exp", 5.0))
        ),
    )
    algorithm.load_state_dict(checkpoint_state.algorithm_state)
    return algorithm


def _load_rebrac_algorithm(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    *,
    device: torch.device,
) -> ReBRACAlgorithm:
    obs_dim, action_dim = _infer_continuous_env_spaces(config)
    hidden_sizes = tuple(config.algo_kwargs.get("hidden_sizes", (256, 256)))
    algorithm = ReBRACAlgorithm(
        model=MLPTD3Model(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=hidden_sizes).to(device),
        learning_rate=float(config.algo_kwargs.get("learning_rate", 3e-4)),
        gamma=float(config.algo_kwargs.get("gamma", 0.99)),
        tau=float(config.algo_kwargs.get("tau", 0.005)),
        policy_noise=float(config.algo_kwargs.get("policy_noise", 0.2)),
        noise_clip=float(config.algo_kwargs.get("noise_clip", 0.5)),
        policy_delay=int(config.algo_kwargs.get("policy_delay", 2)),
        actor_bc_weight=float(config.algo_kwargs.get("actor_bc_weight", 1.0)),
        critic_bc_weight=float(config.algo_kwargs.get("critic_bc_weight", 1.0)),
        actor_q_weight=float(config.algo_kwargs.get("actor_q_weight", 1.0)),
    )
    algorithm.load_state_dict(checkpoint_state.algorithm_state)
    return algorithm


def _load_her_algorithm(config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device) -> HERAlgorithm:
    goal_spec, action_dim = _infer_her_spaces(config)
    hidden_sizes = tuple(config.algo_kwargs.get("hidden_sizes", (256, 256)))
    algorithm = HERAlgorithm(
        model=MLPDDPGModel(obs_dim=goal_spec.flat_observation_dim, action_dim=action_dim, hidden_sizes=hidden_sizes).to(device),
        learning_rate=float(config.algo_kwargs.get("learning_rate", 3e-4)),
        gamma=float(config.algo_kwargs.get("gamma", 0.99)),
        tau=float(config.algo_kwargs.get("tau", 0.005)),
    )
    algorithm.load_state_dict(checkpoint_state.algorithm_state)
    return algorithm


def _load_ppo_algorithm(config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device) -> PPOAlgorithm:
    obs_shape, action_dim = _infer_discrete_env_spaces(config)
    if len(obs_shape) == 1:
        hidden_sizes = tuple(config.algo_kwargs.get("hidden_sizes", (64, 64)))
        policy = MLPActorCritic(obs_dim=obs_shape[0], action_dim=action_dim, hidden_sizes=hidden_sizes).to(device)
    else:
        policy = CNNActorCritic(
            obs_shape=obs_shape,
            action_dim=action_dim,
            hidden_sizes=tuple(config.algo_kwargs.get("head_hidden_sizes", config.algo_kwargs.get("hidden_sizes", (512,)))),
            features_dim=int(config.algo_kwargs.get("features_dim", 512)),
        ).to(device)
    algorithm = PPOAlgorithm(
        policy=policy,
        learning_rate=float(config.algo_kwargs.get("learning_rate", 3e-4)),
        clip_coef=float(config.algo_kwargs.get("clip_coef", 0.2)),
        ent_coef=float(config.algo_kwargs.get("ent_coef", 0.01)),
        vf_coef=float(config.algo_kwargs.get("vf_coef", 0.5)),
        max_grad_norm=float(config.algo_kwargs.get("max_grad_norm", 0.5)),
    )
    algorithm.load_state_dict(checkpoint_state.algorithm_state)
    return algorithm


def _load_trpo_algorithm(config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device) -> TRPOAlgorithm:
    obs_shape, action_dim = _infer_discrete_env_spaces(config)
    if len(obs_shape) != 1:
        raise ValueError("trpo checkpoint loading currently supports flat observations only")

    algorithm = TRPOAlgorithm(
        policy=MLPActorCritic(
            obs_dim=obs_shape[0],
            action_dim=action_dim,
            hidden_sizes=tuple(config.algo_kwargs.get("hidden_sizes", (64, 64))),
        ).to(device),
        value_learning_rate=float(
            config.algo_kwargs.get("value_learning_rate", config.algo_kwargs.get("learning_rate", 1e-3))
        ),
        max_kl=float(config.algo_kwargs.get("max_kl", 0.01)),
        cg_iterations=int(config.algo_kwargs.get("cg_iterations", 10)),
        cg_damping=float(config.algo_kwargs.get("cg_damping", 0.1)),
        line_search_steps=int(config.algo_kwargs.get("line_search_steps", 10)),
        line_search_shrink=float(config.algo_kwargs.get("line_search_shrink", 0.8)),
        value_updates=int(config.algo_kwargs.get("value_updates", 5)),
        ent_coef=float(config.algo_kwargs.get("ent_coef", 0.0)),
    )
    algorithm.load_state_dict(checkpoint_state.algorithm_state)
    return algorithm


def _load_recurrent_ppo_algorithm(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    *,
    device: torch.device,
) -> RecurrentPPOAlgorithm:
    obs_shape, action_dim = _infer_discrete_env_spaces(config)
    if len(obs_shape) == 1:
        encoder_hidden_sizes = tuple(config.algo_kwargs.get("encoder_hidden_sizes", (128,)))
        head_hidden_sizes = tuple(config.algo_kwargs.get("head_hidden_sizes", (128,)))
    else:
        encoder_hidden_sizes = ()
        head_hidden_sizes = tuple(config.algo_kwargs.get("head_hidden_sizes", (128,)))

    algorithm = RecurrentPPOAlgorithm(
        policy=LSTMActorCritic(
            obs_shape=obs_shape,
            action_dim=action_dim,
            features_dim=int(config.algo_kwargs.get("features_dim", 256)),
            encoder_hidden_sizes=encoder_hidden_sizes,
            head_hidden_sizes=head_hidden_sizes,
            hidden_size=int(config.algo_kwargs.get("recurrent_hidden_size", 256)),
            num_layers=int(config.algo_kwargs.get("recurrent_num_layers", 1)),
        ).to(device),
        learning_rate=float(config.algo_kwargs.get("learning_rate", 3e-4)),
        clip_coef=float(config.algo_kwargs.get("clip_coef", 0.2)),
        ent_coef=float(config.algo_kwargs.get("ent_coef", 0.01)),
        vf_coef=float(config.algo_kwargs.get("vf_coef", 0.5)),
        max_grad_norm=float(config.algo_kwargs.get("max_grad_norm", 0.5)),
    )
    algorithm.load_state_dict(checkpoint_state.algorithm_state)
    return algorithm


def _load_dqn_algorithm(config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device) -> DQNAlgorithm:
    obs_shape, action_dim = _infer_discrete_env_spaces(config)
    hidden_sizes = tuple(config.algo_kwargs.get("hidden_sizes", (64, 64)))

    if len(obs_shape) == 3:
        if config.algo != "dqn":
            raise ValueError(f"image observations are currently supported for algo='dqn' only, got {config.algo!r}")
        q_network = CNNQNetwork(
            obs_shape=obs_shape,
            action_dim=action_dim,
            hidden_sizes=tuple(config.algo_kwargs.get("head_hidden_sizes", config.algo_kwargs.get("hidden_sizes", (512,)))),
            features_dim=int(config.algo_kwargs.get("features_dim", 512)),
        ).to(device)
        algorithm_cls = DQNAlgorithm
    else:
        obs_dim = obs_shape[0]
        if config.algo == "rainbow_dqn":
            q_network = MLPDuelingNoisyQNetwork(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=hidden_sizes).to(device)
            algorithm_cls = RainbowDQNAlgorithm
        elif config.algo == "dueling_dqn":
            q_network = MLPDuelingQNetwork(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=hidden_sizes).to(device)
            algorithm_cls = DuelingDQNAlgorithm
        elif config.algo == "noisy_dqn":
            q_network = MLPNoisyQNetwork(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=hidden_sizes).to(device)
            algorithm_cls = NoisyDQNAlgorithm
        else:
            q_network = MLPQNetwork(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=hidden_sizes).to(device)
            if config.algo == "double_dqn":
                algorithm_cls = DoubleDQNAlgorithm
            elif config.algo == "prioritized_dqn":
                algorithm_cls = PrioritizedDQNAlgorithm
            else:
                algorithm_cls = DQNAlgorithm

    algorithm = algorithm_cls(
        q_network=q_network,
        learning_rate=float(config.algo_kwargs.get("learning_rate", 1e-3)),
        gamma=float(config.algo_kwargs.get("gamma", 0.99)),
        target_update_interval=int(config.algo_kwargs.get("target_update_interval", 250)),
    )
    algorithm.load_state_dict(checkpoint_state.algorithm_state)
    return algorithm


def _load_c51_dqn_algorithm(config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device) -> C51DQNAlgorithm:
    obs_shape, action_dim = _infer_discrete_env_spaces(config)
    if len(obs_shape) != 1:
        raise ValueError("c51_dqn checkpoint loading currently supports flat observations only")
    obs_dim = obs_shape[0]
    hidden_sizes = tuple(config.algo_kwargs.get("hidden_sizes", (64, 64)))
    v_min = float(config.algo_kwargs.get("v_min", 0.0))
    v_max = float(config.algo_kwargs.get("v_max", 200.0))
    num_atoms = int(config.algo_kwargs.get("num_atoms", 51))

    q_network = MLPC51QNetwork(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=hidden_sizes,
        v_min=v_min,
        v_max=v_max,
        num_atoms=num_atoms,
    ).to(device)
    algorithm = C51DQNAlgorithm(
        q_network=q_network,
        learning_rate=float(config.algo_kwargs.get("learning_rate", 1e-3)),
        gamma=float(config.algo_kwargs.get("gamma", 0.99)),
        target_update_interval=int(config.algo_kwargs.get("target_update_interval", 250)),
        v_min=v_min,
        v_max=v_max,
        num_atoms=num_atoms,
    )
    algorithm.load_state_dict(checkpoint_state.algorithm_state)
    return algorithm


def _load_qr_dqn_algorithm(config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device) -> QRDQNAlgorithm:
    obs_shape, action_dim = _infer_discrete_env_spaces(config)
    if len(obs_shape) != 1:
        raise ValueError("qr_dqn checkpoint loading currently supports flat observations only")
    obs_dim = obs_shape[0]
    hidden_sizes = tuple(config.algo_kwargs.get("hidden_sizes", (64, 64)))
    num_quantiles = int(config.algo_kwargs.get("num_quantiles", 51))
    kappa = float(config.algo_kwargs.get("kappa", 1.0))

    q_network = MLPQRQNetwork(
        obs_dim=obs_dim,
        action_dim=action_dim,
        num_quantiles=num_quantiles,
        hidden_sizes=hidden_sizes,
    ).to(device)
    algorithm = QRDQNAlgorithm(
        q_network=q_network,
        learning_rate=float(config.algo_kwargs.get("learning_rate", 1e-3)),
        gamma=float(config.algo_kwargs.get("gamma", 0.99)),
        target_update_interval=int(config.algo_kwargs.get("target_update_interval", 250)),
        num_quantiles=num_quantiles,
        kappa=kappa,
    )
    algorithm.load_state_dict(checkpoint_state.algorithm_state)
    return algorithm


def _load_iqn_algorithm(config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device) -> IQNAlgorithm:
    obs_shape, action_dim = _infer_discrete_env_spaces(config)
    if len(obs_shape) != 1:
        raise ValueError("iqn checkpoint loading currently supports flat observations only")
    obs_dim = obs_shape[0]
    hidden_sizes = tuple(config.algo_kwargs.get("hidden_sizes", (64, 64)))
    num_quantiles = int(config.algo_kwargs.get("num_quantiles", 32))
    embedding_dim = int(config.algo_kwargs.get("embedding_dim", 64))
    kappa = float(config.algo_kwargs.get("kappa", 1.0))

    q_network = MLPIQNetwork(
        obs_dim=obs_dim,
        action_dim=action_dim,
        num_quantiles=num_quantiles,
        hidden_sizes=hidden_sizes,
        embedding_dim=embedding_dim,
    ).to(device)
    algorithm = IQNAlgorithm(
        q_network=q_network,
        learning_rate=float(config.algo_kwargs.get("learning_rate", 1e-3)),
        gamma=float(config.algo_kwargs.get("gamma", 0.99)),
        target_update_interval=int(config.algo_kwargs.get("target_update_interval", 250)),
        num_quantiles=num_quantiles,
        kappa=kappa,
    )
    algorithm.load_state_dict(checkpoint_state.algorithm_state)
    return algorithm


def _load_iql_algorithm(config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device) -> IQLAlgorithm:
    obs_dim, action_dim = _infer_continuous_env_spaces(config)
    hidden_sizes = tuple(config.algo_kwargs.get("hidden_sizes", (256, 256)))
    algorithm = IQLAlgorithm(
        model=MLPIQLModel(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=hidden_sizes).to(device),
        learning_rate=float(config.algo_kwargs.get("learning_rate", 3e-4)),
        gamma=float(config.algo_kwargs.get("gamma", 0.99)),
        tau=float(config.algo_kwargs.get("tau", 0.005)),
        expectile=float(config.algo_kwargs.get("expectile", 0.7)),
        beta=float(config.algo_kwargs.get("beta", 3.0)),
        max_advantage_weight=float(config.algo_kwargs.get("max_advantage_weight", 100.0)),
    )
    algorithm.load_state_dict(checkpoint_state.algorithm_state)
    return algorithm


def _load_awr_algorithm(config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device) -> AWRAlgorithm:
    obs_dim, action_dim = _infer_continuous_env_spaces(config)
    hidden_sizes = tuple(config.algo_kwargs.get("hidden_sizes", (256, 256)))
    algorithm = AWRAlgorithm(
        model=MLPIQLModel(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=hidden_sizes).to(device),
        learning_rate=float(config.algo_kwargs.get("learning_rate", 3e-4)),
        beta=float(config.algo_kwargs.get("beta", 1.0)),
        max_weight=float(config.algo_kwargs.get("max_weight", 20.0)),
        normalize_advantages=bool(config.algo_kwargs.get("normalize_advantages", True)),
    )
    algorithm.load_state_dict(checkpoint_state.algorithm_state)
    return algorithm


def _load_marwil_algorithm(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    *,
    device: torch.device,
) -> MARWILAlgorithm:
    obs_dim, action_dim = _infer_continuous_env_spaces(config)
    hidden_sizes = tuple(config.algo_kwargs.get("hidden_sizes", (256, 256)))
    algorithm = MARWILAlgorithm(
        model=MLPIQLModel(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=hidden_sizes).to(device),
        learning_rate=float(config.algo_kwargs.get("learning_rate", 3e-4)),
        beta=float(config.algo_kwargs.get("beta", 1.0)),
        vf_coeff=float(config.algo_kwargs.get("vf_coeff", 1.0)),
        moving_average_sqd_adv_norm_start=float(
            config.algo_kwargs.get("moving_average_sqd_adv_norm_start", 100.0)
        ),
        moving_average_sqd_adv_norm_update_rate=float(
            config.algo_kwargs.get("moving_average_sqd_adv_norm_update_rate", 0.01)
        ),
    )
    algorithm.load_state_dict(checkpoint_state.algorithm_state)
    return algorithm


def _load_sac_algorithm(config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device) -> SACAlgorithm:
    obs_dim, action_dim = _infer_continuous_env_spaces(config)
    hidden_sizes = tuple(config.algo_kwargs.get("hidden_sizes", (256, 256)))
    algorithm = SACAlgorithm(
        model=MLPSACModel(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=hidden_sizes).to(device),
        learning_rate=float(config.algo_kwargs.get("learning_rate", 3e-4)),
        gamma=float(config.algo_kwargs.get("gamma", 0.99)),
        alpha=float(config.algo_kwargs.get("alpha", 0.2)),
        tau=float(config.algo_kwargs.get("tau", 0.005)),
    )
    algorithm.load_state_dict(checkpoint_state.algorithm_state)
    return algorithm


def _load_rlpd_algorithm(config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device) -> RLPDAlgorithm:
    obs_dim, action_dim = _infer_continuous_env_spaces(config)
    hidden_sizes = tuple(config.algo_kwargs.get("hidden_sizes", (256, 256)))
    algorithm = RLPDAlgorithm(
        model=MLPSACModel(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=hidden_sizes).to(device),
        learning_rate=float(config.algo_kwargs.get("learning_rate", 3e-4)),
        gamma=float(config.algo_kwargs.get("gamma", 0.99)),
        alpha=float(config.algo_kwargs.get("alpha", 0.2)),
        tau=float(config.algo_kwargs.get("tau", 0.005)),
    )
    algorithm.load_state_dict(checkpoint_state.algorithm_state)
    return algorithm


def _load_cql_algorithm(config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device) -> CQLAlgorithm:
    obs_dim, action_dim = _infer_continuous_env_spaces(config)
    hidden_sizes = tuple(config.algo_kwargs.get("hidden_sizes", (256, 256)))
    algorithm = CQLAlgorithm(
        model=MLPSACModel(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=hidden_sizes).to(device),
        learning_rate=float(config.algo_kwargs.get("learning_rate", 3e-4)),
        gamma=float(config.algo_kwargs.get("gamma", 0.99)),
        alpha=float(config.algo_kwargs.get("alpha", 0.2)),
        tau=float(config.algo_kwargs.get("tau", 0.005)),
        cql_alpha=float(config.algo_kwargs.get("cql_alpha", 5.0)),
        num_cql_samples=int(config.algo_kwargs.get("num_cql_samples", 10)),
    )
    algorithm.load_state_dict(checkpoint_state.algorithm_state)
    return algorithm


def _load_tqc_algorithm(config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device) -> TQCAlgorithm:
    obs_dim, action_dim = _infer_continuous_env_spaces(config)
    hidden_sizes = tuple(config.algo_kwargs.get("hidden_sizes", (256, 256)))
    num_critics = int(config.algo_kwargs.get("num_critics", 2))
    num_quantiles = int(config.algo_kwargs.get("num_quantiles", 25))
    top_quantiles_to_drop_per_net = int(config.algo_kwargs.get("top_quantiles_to_drop_per_net", 2))
    kappa = float(config.algo_kwargs.get("kappa", 1.0))

    algorithm = TQCAlgorithm(
        model=MLPTQCModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
            num_critics=num_critics,
            num_quantiles=num_quantiles,
        ).to(device),
        learning_rate=float(config.algo_kwargs.get("learning_rate", 3e-4)),
        gamma=float(config.algo_kwargs.get("gamma", 0.99)),
        alpha=float(config.algo_kwargs.get("alpha", 0.2)),
        tau=float(config.algo_kwargs.get("tau", 0.005)),
        top_quantiles_to_drop_per_net=top_quantiles_to_drop_per_net,
        num_quantiles=num_quantiles,
        kappa=kappa,
    )
    algorithm.load_state_dict(checkpoint_state.algorithm_state)
    return algorithm


def _load_redq_algorithm(config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device) -> REDQAlgorithm:
    obs_dim, action_dim = _infer_continuous_env_spaces(config)
    hidden_sizes = tuple(config.algo_kwargs.get("hidden_sizes", (256, 256)))
    num_critics = int(config.algo_kwargs.get("num_critics", 10))
    subset_size = int(config.algo_kwargs.get("subset_size", 2))

    algorithm = REDQAlgorithm(
        model=MLPREDQModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
            num_critics=num_critics,
        ).to(device),
        learning_rate=float(config.algo_kwargs.get("learning_rate", 3e-4)),
        gamma=float(config.algo_kwargs.get("gamma", 0.99)),
        alpha=float(config.algo_kwargs.get("alpha", 0.2)),
        tau=float(config.algo_kwargs.get("tau", 0.005)),
        num_critics=num_critics,
        subset_size=subset_size,
    )
    algorithm.load_state_dict(checkpoint_state.algorithm_state)
    return algorithm


def _load_edac_algorithm(config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device) -> EDACAlgorithm:
    obs_dim, action_dim = _infer_continuous_env_spaces(config)
    hidden_sizes = tuple(config.algo_kwargs.get("hidden_sizes", (256, 256)))
    num_critics = int(config.algo_kwargs.get("num_critics", 10))

    algorithm = EDACAlgorithm(
        model=MLPREDQModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
            num_critics=num_critics,
        ).to(device),
        learning_rate=float(config.algo_kwargs.get("learning_rate", 3e-4)),
        gamma=float(config.algo_kwargs.get("gamma", 0.99)),
        alpha=float(config.algo_kwargs.get("alpha", 0.2)),
        tau=float(config.algo_kwargs.get("tau", 0.005)),
        num_critics=num_critics,
        eta=float(config.algo_kwargs.get("eta", 1.0)),
    )
    algorithm.load_state_dict(checkpoint_state.algorithm_state)
    return algorithm


def _load_ddpg_algorithm(config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device) -> DDPGAlgorithm:
    obs_dim, action_dim = _infer_continuous_env_spaces(config)
    hidden_sizes = tuple(config.algo_kwargs.get("hidden_sizes", (256, 256)))
    algorithm = DDPGAlgorithm(
        model=MLPDDPGModel(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=hidden_sizes).to(device),
        learning_rate=float(config.algo_kwargs.get("learning_rate", 3e-4)),
        gamma=float(config.algo_kwargs.get("gamma", 0.99)),
        tau=float(config.algo_kwargs.get("tau", 0.005)),
    )
    algorithm.load_state_dict(checkpoint_state.algorithm_state)
    return algorithm


def _load_drqv2_algorithm(config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device) -> DrQv2Algorithm:
    obs_shape, action_dim = _infer_image_continuous_env_spaces(config)
    algorithm = DrQv2Algorithm(
        model=CNNDrQv2Model(
            obs_shape=obs_shape,
            action_dim=action_dim,
            features_dim=int(config.algo_kwargs.get("features_dim", 256)),
            actor_hidden_sizes=tuple(config.algo_kwargs.get("actor_hidden_sizes", (256, 256))),
            critic_hidden_sizes=tuple(config.algo_kwargs.get("critic_hidden_sizes", (256, 256))),
        ).to(device),
        learning_rate=float(config.algo_kwargs.get("learning_rate", 1e-4)),
        gamma=float(config.algo_kwargs.get("gamma", 0.99)),
        tau=float(config.algo_kwargs.get("tau", 0.01)),
        policy_delay=int(config.algo_kwargs.get("policy_delay", 2)),
        augmentation_pad=int(config.algo_kwargs.get("augmentation_pad", 4)),
    )
    algorithm.load_state_dict(checkpoint_state.algorithm_state)
    return algorithm


def _load_crossq_algorithm(config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device) -> CrossQAlgorithm:
    obs_dim, action_dim = _infer_continuous_env_spaces(config)
    algorithm = CrossQAlgorithm(
        model=MLPCrossQModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=tuple(config.algo_kwargs.get("hidden_sizes", (256, 256))),
            critic_hidden_sizes=tuple(config.algo_kwargs.get("critic_hidden_sizes", (512, 512))),
            bn_momentum=float(config.algo_kwargs.get("bn_momentum", 0.99)),
        ).to(device),
        learning_rate=float(config.algo_kwargs.get("learning_rate", 1e-3)),
        gamma=float(config.algo_kwargs.get("gamma", 0.99)),
        alpha=float(config.algo_kwargs.get("alpha", 0.1)),
        policy_delay=int(config.algo_kwargs.get("policy_delay", 3)),
        adam_beta1=float(config.algo_kwargs.get("adam_beta1", 0.5)),
    )
    algorithm.load_state_dict(checkpoint_state.algorithm_state)
    return algorithm


def _load_discrete_sac_algorithm(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    *,
    device: torch.device,
) -> DiscreteSACAlgorithm:
    obs_shape, action_dim = _infer_discrete_env_spaces(config)
    if len(obs_shape) != 1:
        raise ValueError("discrete_sac checkpoint loading currently supports flat observations only")

    algorithm = DiscreteSACAlgorithm(
        model=MLPDiscreteSACModel(
            obs_dim=obs_shape[0],
            action_dim=action_dim,
            hidden_sizes=tuple(config.algo_kwargs.get("hidden_sizes", (256, 256))),
        ).to(device),
        learning_rate=float(config.algo_kwargs.get("learning_rate", 3e-4)),
        gamma=float(config.algo_kwargs.get("gamma", 0.99)),
        alpha=float(config.algo_kwargs.get("alpha", 0.2)),
        tau=float(config.algo_kwargs.get("tau", 0.005)),
    )
    algorithm.load_state_dict(checkpoint_state.algorithm_state)
    return algorithm


def _load_td3_algorithm(config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device) -> TD3Algorithm:
    obs_dim, action_dim = _infer_continuous_env_spaces(config)
    hidden_sizes = tuple(config.algo_kwargs.get("hidden_sizes", (256, 256)))
    algorithm = TD3Algorithm(
        model=MLPTD3Model(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=hidden_sizes).to(device),
        learning_rate=float(config.algo_kwargs.get("learning_rate", 3e-4)),
        gamma=float(config.algo_kwargs.get("gamma", 0.99)),
        tau=float(config.algo_kwargs.get("tau", 0.005)),
        policy_noise=float(config.algo_kwargs.get("policy_noise", 0.2)),
        noise_clip=float(config.algo_kwargs.get("noise_clip", 0.5)),
        policy_delay=int(config.algo_kwargs.get("policy_delay", 2)),
    )
    algorithm.load_state_dict(checkpoint_state.algorithm_state)
    return algorithm


def _load_td3_bc_algorithm(
    config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device
) -> TD3BCAlgorithm:
    obs_dim, action_dim = _infer_continuous_env_spaces(config)
    hidden_sizes = tuple(config.algo_kwargs.get("hidden_sizes", (256, 256)))
    algorithm = TD3BCAlgorithm(
        model=MLPTD3Model(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=hidden_sizes).to(device),
        learning_rate=float(config.algo_kwargs.get("learning_rate", 3e-4)),
        gamma=float(config.algo_kwargs.get("gamma", 0.99)),
        tau=float(config.algo_kwargs.get("tau", 0.005)),
        policy_noise=float(config.algo_kwargs.get("policy_noise", 0.2)),
        noise_clip=float(config.algo_kwargs.get("noise_clip", 0.5)),
        policy_delay=int(config.algo_kwargs.get("policy_delay", 2)),
        bc_alpha=float(config.algo_kwargs.get("bc_alpha", 2.5)),
    )
    algorithm.load_state_dict(checkpoint_state.algorithm_state)
    return algorithm


def _evaluate_a2c(config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int) -> MetricDict:
    algorithm = _load_a2c_algorithm(config, checkpoint_state, device=device)
    return _evaluate_a2c_policy(algorithm.policy, config, device=device, num_episodes=num_episodes)


def _evaluate_bc(config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int) -> MetricDict:
    algorithm = _load_bc_algorithm(config, checkpoint_state, device=device)
    return _evaluate_bc_policy(algorithm.model, config, device=device, num_episodes=num_episodes)


def _evaluate_bcq(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    num_episodes: int,
) -> MetricDict:
    algorithm = _load_bcq_algorithm(config, checkpoint_state, device=device)
    return _evaluate_bcq_policy(
        algorithm.model,
        config,
        device=device,
        num_episodes=num_episodes,
        num_action_samples=int(config.algo_kwargs.get("num_action_samples", 10)),
    )


def _evaluate_bear(config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int) -> MetricDict:
    algorithm = _load_bear_algorithm(config, checkpoint_state, device=device)
    return _evaluate_sac_policy(algorithm.model, config, device=device, num_episodes=num_episodes)


def _evaluate_awac(config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int) -> MetricDict:
    algorithm = _load_awac_algorithm(config, checkpoint_state, device=device)
    return _evaluate_sac_policy(algorithm.model, config, device=device, num_episodes=num_episodes)


def _evaluate_crr(config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int) -> MetricDict:
    algorithm = _load_crr_algorithm(config, checkpoint_state, device=device)
    return _evaluate_sac_policy(algorithm.model, config, device=device, num_episodes=num_episodes)


def _evaluate_rebrac(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    num_episodes: int,
) -> MetricDict:
    algorithm = _load_rebrac_algorithm(config, checkpoint_state, device=device)
    return _evaluate_td3_policy(algorithm.model, config, device=device, num_episodes=num_episodes)


def _evaluate_her(config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int) -> MetricDict:
    algorithm = _load_her_algorithm(config, checkpoint_state, device=device)
    return _evaluate_her_policy(algorithm.model, config, device=device, num_episodes=num_episodes)


def _evaluate_ppo(config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int) -> MetricDict:
    algorithm = _load_ppo_algorithm(config, checkpoint_state, device=device)
    return _evaluate_policy(algorithm.policy, config, device=device, num_episodes=num_episodes)


def _evaluate_trpo(config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int) -> MetricDict:
    algorithm = _load_trpo_algorithm(config, checkpoint_state, device=device)
    return _evaluate_policy(algorithm.policy, config, device=device, num_episodes=num_episodes)


def _evaluate_recurrent_ppo(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    num_episodes: int,
) -> MetricDict:
    algorithm = _load_recurrent_ppo_algorithm(config, checkpoint_state, device=device)
    return _evaluate_recurrent_policy(algorithm.policy, config, device=device, num_episodes=num_episodes)


def _evaluate_dqn(config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int) -> MetricDict:
    algorithm = _load_dqn_algorithm(config, checkpoint_state, device=device)
    algorithm.set_eval_mode()
    return _evaluate_q_policy(algorithm.q_network, config, device=device, num_episodes=num_episodes)


def _evaluate_c51_dqn(
    config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int
) -> MetricDict:
    algorithm = _load_c51_dqn_algorithm(config, checkpoint_state, device=device)
    algorithm.set_eval_mode()
    return _evaluate_q_policy(algorithm.q_network, config, device=device, num_episodes=num_episodes)


def _evaluate_qr_dqn(config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int) -> MetricDict:
    algorithm = _load_qr_dqn_algorithm(config, checkpoint_state, device=device)
    algorithm.set_eval_mode()
    return _evaluate_q_policy(algorithm.q_network, config, device=device, num_episodes=num_episodes)


def _evaluate_iqn(config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int) -> MetricDict:
    algorithm = _load_iqn_algorithm(config, checkpoint_state, device=device)
    algorithm.set_eval_mode()
    return _evaluate_q_policy(algorithm.q_network, config, device=device, num_episodes=num_episodes)


def _evaluate_iql(config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int) -> MetricDict:
    algorithm = _load_iql_algorithm(config, checkpoint_state, device=device)
    return _evaluate_iql_policy(algorithm.model, config, device=device, num_episodes=num_episodes)


def _evaluate_awr(config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int) -> MetricDict:
    algorithm = _load_awr_algorithm(config, checkpoint_state, device=device)
    return _evaluate_iql_policy(algorithm.model, config, device=device, num_episodes=num_episodes)


def _evaluate_marwil(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    num_episodes: int,
) -> MetricDict:
    algorithm = _load_marwil_algorithm(config, checkpoint_state, device=device)
    return _evaluate_iql_policy(algorithm.model, config, device=device, num_episodes=num_episodes)


def _evaluate_xql(config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int) -> MetricDict:
    algorithm = _load_xql_algorithm(config, checkpoint_state, device=device)
    return _evaluate_iql_policy(algorithm.model, config, device=device, num_episodes=num_episodes)


def _evaluate_cal_ql(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    num_episodes: int,
) -> MetricDict:
    algorithm = _load_cal_ql_algorithm(config, checkpoint_state, device=device)
    return _evaluate_sac_policy(algorithm.model, config, device=device, num_episodes=num_episodes)


def _evaluate_sac(config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int) -> MetricDict:
    algorithm = _load_sac_algorithm(config, checkpoint_state, device=device)
    return _evaluate_sac_policy(algorithm.model, config, device=device, num_episodes=num_episodes)


def _evaluate_rlpd(config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int) -> MetricDict:
    algorithm = _load_rlpd_algorithm(config, checkpoint_state, device=device)
    return _evaluate_sac_policy(algorithm.model, config, device=device, num_episodes=num_episodes)


def _evaluate_cql(config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int) -> MetricDict:
    algorithm = _load_cql_algorithm(config, checkpoint_state, device=device)
    return _evaluate_sac_policy(algorithm.model, config, device=device, num_episodes=num_episodes)


def _evaluate_crossq(config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int) -> MetricDict:
    algorithm = _load_crossq_algorithm(config, checkpoint_state, device=device)
    return _evaluate_crossq_policy(algorithm.model, config, device=device, num_episodes=num_episodes)


def _evaluate_tqc(config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int) -> MetricDict:
    algorithm = _load_tqc_algorithm(config, checkpoint_state, device=device)
    return _evaluate_tqc_policy(algorithm.model, config, device=device, num_episodes=num_episodes)


def _evaluate_redq(config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int) -> MetricDict:
    algorithm = _load_redq_algorithm(config, checkpoint_state, device=device)
    return _evaluate_redq_policy(algorithm.model, config, device=device, num_episodes=num_episodes)


def _evaluate_edac(config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int) -> MetricDict:
    algorithm = _load_edac_algorithm(config, checkpoint_state, device=device)
    return _evaluate_redq_policy(algorithm.model, config, device=device, num_episodes=num_episodes)


def _evaluate_ddpg(config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int) -> MetricDict:
    algorithm = _load_ddpg_algorithm(config, checkpoint_state, device=device)
    return _evaluate_ddpg_policy(algorithm.model, config, device=device, num_episodes=num_episodes)


def _evaluate_drqv2(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    num_episodes: int,
) -> MetricDict:
    algorithm = _load_drqv2_algorithm(config, checkpoint_state, device=device)
    return _evaluate_drqv2_policy(algorithm.model, config, device=device, num_episodes=num_episodes)


def _evaluate_discrete_sac(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    num_episodes: int,
) -> MetricDict:
    algorithm = _load_discrete_sac_algorithm(config, checkpoint_state, device=device)
    return _evaluate_discrete_sac_policy(algorithm.model, config, device=device, num_episodes=num_episodes)


def _evaluate_td3(config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int) -> MetricDict:
    algorithm = _load_td3_algorithm(config, checkpoint_state, device=device)
    return _evaluate_td3_policy(algorithm.model, config, device=device, num_episodes=num_episodes)


def _evaluate_td3_bc(
    config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int
) -> MetricDict:
    algorithm = _load_td3_bc_algorithm(config, checkpoint_state, device=device)
    return _evaluate_td3_policy(algorithm.model, config, device=device, num_episodes=num_episodes)


def _predict_a2c(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    algorithm = _load_a2c_algorithm(config, checkpoint_state, device=device)
    obs_tensor = _prepare_observation(obs, device=device)
    with torch.no_grad():
        actions = algorithm.policy.act(obs_tensor, deterministic=deterministic).actions
    return _format_action_output(actions, discrete=True)


def _predict_bc(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    del deterministic
    algorithm = _load_bc_algorithm(config, checkpoint_state, device=device)
    obs_tensor = _prepare_observation(obs, device=device)
    low, high = _continuous_action_bounds(config, device=device)
    with torch.no_grad():
        normalized_actions = algorithm.model.actor(obs_tensor)
        actions = _scale_continuous_actions(normalized_actions, low=low, high=high)
    return _format_action_output(actions, discrete=False)


def _predict_awac(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    algorithm = _load_awac_algorithm(config, checkpoint_state, device=device)
    obs_tensor = _prepare_observation(obs, device=device)
    low, high = _continuous_action_bounds(config, device=device)
    with torch.no_grad():
        normalized_actions = algorithm.model.sample_actions(obs_tensor, deterministic=deterministic).actions
        actions = _scale_continuous_actions(normalized_actions, low=low, high=high)
    return _format_action_output(actions, discrete=False)


def _predict_crr(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    algorithm = _load_crr_algorithm(config, checkpoint_state, device=device)
    obs_tensor = _prepare_observation(obs, device=device)
    low, high = _continuous_action_bounds(config, device=device)
    with torch.no_grad():
        normalized_actions = algorithm.model.sample_actions(obs_tensor, deterministic=deterministic).actions
        actions = _scale_continuous_actions(normalized_actions, low=low, high=high)
    return _format_action_output(actions, discrete=False)


def _predict_rebrac(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    del deterministic
    algorithm = _load_rebrac_algorithm(config, checkpoint_state, device=device)
    obs_tensor = _prepare_observation(obs, device=device)
    low, high = _continuous_action_bounds(config, device=device)
    with torch.no_grad():
        normalized_actions = algorithm.model.actor(obs_tensor)
        actions = _scale_continuous_actions(normalized_actions, low=low, high=high)
    return _format_action_output(actions, discrete=False)


def _predict_bcq(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    algorithm = _load_bcq_algorithm(config, checkpoint_state, device=device)
    obs_tensor = _prepare_observation(obs, device=device)
    low, high = _continuous_action_bounds(config, device=device)
    with torch.no_grad():
        normalized_actions = algorithm.model.select_actions(
            obs_tensor,
            num_action_samples=int(config.algo_kwargs.get("num_action_samples", 10)),
            deterministic=deterministic,
        )
        actions = _scale_continuous_actions(normalized_actions, low=low, high=high)
    return _format_action_output(actions, discrete=False)


def _predict_bear(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    algorithm = _load_bear_algorithm(config, checkpoint_state, device=device)
    obs_tensor = _prepare_observation(obs, device=device)
    low, high = _continuous_action_bounds(config, device=device)
    with torch.no_grad():
        normalized_actions = algorithm.model.sample_actions(obs_tensor, deterministic=deterministic).actions
        actions = _scale_continuous_actions(normalized_actions, low=low, high=high)
    return _format_action_output(actions, discrete=False)


def _predict_her(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    del deterministic
    if not isinstance(obs, Mapping):
        raise TypeError(f"HER predict expects a goal-conditioned observation mapping, got {type(obs)!r}")
    algorithm = _load_her_algorithm(config, checkpoint_state, device=device)
    obs_tensor = torch.as_tensor(flatten_goal_observation(obs), dtype=torch.float32, device=device)
    low, high = _continuous_action_bounds(config, device=device)
    with torch.no_grad():
        normalized_actions = algorithm.model.actor(obs_tensor)
        actions = _scale_continuous_actions(normalized_actions, low=low, high=high)
    return _format_action_output(actions, discrete=False)


def _predict_ppo(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    algorithm = _load_ppo_algorithm(config, checkpoint_state, device=device)
    obs_tensor = _prepare_observation(obs, device=device)
    with torch.no_grad():
        actions = algorithm.policy.act(obs_tensor, deterministic=deterministic).actions
    return _format_action_output(actions, discrete=True)


def _predict_trpo(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    algorithm = _load_trpo_algorithm(config, checkpoint_state, device=device)
    obs_tensor = _prepare_observation(obs, device=device)
    with torch.no_grad():
        actions = algorithm.policy.act(obs_tensor, deterministic=deterministic).actions
    return _format_action_output(actions, discrete=True)


def _predict_recurrent_ppo(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    algorithm = _load_recurrent_ppo_algorithm(config, checkpoint_state, device=device)
    obs_tensor = _prepare_observation(obs, device=device)
    initial_state = algorithm.policy.initial_state(int(obs_tensor.shape[0]), device=device)
    with torch.no_grad():
        actions = algorithm.policy.act(obs_tensor, state=initial_state, deterministic=deterministic).actions
    return _format_action_output(actions, discrete=True)


def _predict_dqn(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    del deterministic
    algorithm = _load_dqn_algorithm(config, checkpoint_state, device=device)
    algorithm.set_eval_mode()
    obs_tensor = _prepare_observation(obs, device=device)
    with torch.no_grad():
        actions = algorithm.q_network.act(obs_tensor, epsilon=0.0)
    return _format_action_output(actions, discrete=True)


def _predict_c51_dqn(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    del deterministic
    algorithm = _load_c51_dqn_algorithm(config, checkpoint_state, device=device)
    algorithm.set_eval_mode()
    obs_tensor = _prepare_observation(obs, device=device)
    with torch.no_grad():
        actions = algorithm.q_network.act(obs_tensor, epsilon=0.0)
    return _format_action_output(actions, discrete=True)


def _predict_qr_dqn(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    del deterministic
    algorithm = _load_qr_dqn_algorithm(config, checkpoint_state, device=device)
    algorithm.set_eval_mode()
    obs_tensor = _prepare_observation(obs, device=device)
    with torch.no_grad():
        actions = algorithm.q_network.act(obs_tensor, epsilon=0.0)
    return _format_action_output(actions, discrete=True)


def _predict_iqn(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    del deterministic
    algorithm = _load_iqn_algorithm(config, checkpoint_state, device=device)
    algorithm.set_eval_mode()
    obs_tensor = _prepare_observation(obs, device=device)
    with torch.no_grad():
        actions = algorithm.q_network.act(obs_tensor, epsilon=0.0)
    return _format_action_output(actions, discrete=True)


def _predict_iql(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    algorithm = _load_iql_algorithm(config, checkpoint_state, device=device)
    obs_tensor = _prepare_observation(obs, device=device)
    low, high = _continuous_action_bounds(config, device=device)
    with torch.no_grad():
        normalized_actions = algorithm.model.sample_actions(obs_tensor, deterministic=deterministic).actions
        actions = _scale_continuous_actions(normalized_actions, low=low, high=high)
    return _format_action_output(actions, discrete=False)


def _predict_awr(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    algorithm = _load_awr_algorithm(config, checkpoint_state, device=device)
    obs_tensor = _prepare_observation(obs, device=device)
    low, high = _continuous_action_bounds(config, device=device)
    with torch.no_grad():
        normalized_actions = algorithm.model.sample_actions(obs_tensor, deterministic=deterministic).actions
        actions = _scale_continuous_actions(normalized_actions, low=low, high=high)
    return _format_action_output(actions, discrete=False)


def _predict_marwil(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    algorithm = _load_marwil_algorithm(config, checkpoint_state, device=device)
    obs_tensor = _prepare_observation(obs, device=device)
    low, high = _continuous_action_bounds(config, device=device)
    with torch.no_grad():
        normalized_actions = algorithm.model.sample_actions(obs_tensor, deterministic=deterministic).actions
        actions = _scale_continuous_actions(normalized_actions, low=low, high=high)
    return _format_action_output(actions, discrete=False)


def _predict_xql(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    algorithm = _load_xql_algorithm(config, checkpoint_state, device=device)
    obs_tensor = _prepare_observation(obs, device=device)
    low, high = _continuous_action_bounds(config, device=device)
    with torch.no_grad():
        normalized_actions = algorithm.model.sample_actions(obs_tensor, deterministic=deterministic).actions
        actions = _scale_continuous_actions(normalized_actions, low=low, high=high)
    return _format_action_output(actions, discrete=False)


def _predict_cal_ql(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    algorithm = _load_cal_ql_algorithm(config, checkpoint_state, device=device)
    obs_tensor = _prepare_observation(obs, device=device)
    low, high = _continuous_action_bounds(config, device=device)
    with torch.no_grad():
        normalized_actions = algorithm.model.sample_actions(obs_tensor, deterministic=deterministic).actions
        actions = _scale_continuous_actions(normalized_actions, low=low, high=high)
    return _format_action_output(actions, discrete=False)


def _predict_sac(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    algorithm = _load_sac_algorithm(config, checkpoint_state, device=device)
    obs_tensor = _prepare_observation(obs, device=device)
    low, high = _continuous_action_bounds(config, device=device)
    with torch.no_grad():
        normalized_actions = algorithm.model.sample_actions(obs_tensor, deterministic=deterministic).actions
        actions = _scale_continuous_actions(normalized_actions, low=low, high=high)
    return _format_action_output(actions, discrete=False)


def _predict_rlpd(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    algorithm = _load_rlpd_algorithm(config, checkpoint_state, device=device)
    obs_tensor = _prepare_observation(obs, device=device)
    low, high = _continuous_action_bounds(config, device=device)
    with torch.no_grad():
        normalized_actions = algorithm.model.sample_actions(obs_tensor, deterministic=deterministic).actions
        actions = _scale_continuous_actions(normalized_actions, low=low, high=high)
    return _format_action_output(actions, discrete=False)


def _predict_cql(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    algorithm = _load_cql_algorithm(config, checkpoint_state, device=device)
    obs_tensor = _prepare_observation(obs, device=device)
    low, high = _continuous_action_bounds(config, device=device)
    with torch.no_grad():
        normalized_actions = algorithm.model.sample_actions(obs_tensor, deterministic=deterministic).actions
        actions = _scale_continuous_actions(normalized_actions, low=low, high=high)
    return _format_action_output(actions, discrete=False)


def _predict_tqc(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    algorithm = _load_tqc_algorithm(config, checkpoint_state, device=device)
    obs_tensor = _prepare_observation(obs, device=device)
    low, high = _continuous_action_bounds(config, device=device)
    with torch.no_grad():
        normalized_actions = algorithm.model.sample_actions(obs_tensor, deterministic=deterministic).actions
        actions = _scale_continuous_actions(normalized_actions, low=low, high=high)
    return _format_action_output(actions, discrete=False)


def _predict_redq(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    algorithm = _load_redq_algorithm(config, checkpoint_state, device=device)
    obs_tensor = _prepare_observation(obs, device=device)
    low, high = _continuous_action_bounds(config, device=device)
    with torch.no_grad():
        normalized_actions = algorithm.model.sample_actions(obs_tensor, deterministic=deterministic).actions
        actions = _scale_continuous_actions(normalized_actions, low=low, high=high)
    return _format_action_output(actions, discrete=False)


def _predict_edac(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    algorithm = _load_edac_algorithm(config, checkpoint_state, device=device)
    obs_tensor = _prepare_observation(obs, device=device)
    low, high = _continuous_action_bounds(config, device=device)
    with torch.no_grad():
        normalized_actions = algorithm.model.sample_actions(obs_tensor, deterministic=deterministic).actions
        actions = _scale_continuous_actions(normalized_actions, low=low, high=high)
    return _format_action_output(actions, discrete=False)


def _predict_ddpg(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    del deterministic
    algorithm = _load_ddpg_algorithm(config, checkpoint_state, device=device)
    obs_tensor = _prepare_observation(obs, device=device)
    low, high = _continuous_action_bounds(config, device=device)
    with torch.no_grad():
        normalized_actions = algorithm.model.actor(obs_tensor)
        actions = _scale_continuous_actions(normalized_actions, low=low, high=high)
    return _format_action_output(actions, discrete=False)


def _predict_drqv2(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    algorithm = _load_drqv2_algorithm(config, checkpoint_state, device=device)
    obs_tensor = _prepare_observation(obs, device=device)
    low, high = _continuous_action_bounds(config, device=device)
    with torch.no_grad():
        if deterministic:
            normalized_actions = algorithm.model.actor(obs_tensor)
        else:
            normalized_actions = algorithm.model.sample_actions(
                obs_tensor,
                std=float(config.algo_kwargs.get("exploration_noise", 0.1)),
                clip=float(config.algo_kwargs.get("exploration_noise_clip", 0.3)),
            ).actions
        actions = _scale_continuous_actions(normalized_actions, low=low, high=high)
    return _format_action_output(actions, discrete=False)


def _predict_crossq(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    algorithm = _load_crossq_algorithm(config, checkpoint_state, device=device)
    obs_tensor = _prepare_observation(obs, device=device)
    low, high = _continuous_action_bounds(config, device=device)
    with torch.no_grad():
        normalized_actions = algorithm.model.sample_actions(obs_tensor, deterministic=deterministic).actions
        actions = _scale_continuous_actions(normalized_actions, low=low, high=high)
    return _format_action_output(actions, discrete=False)


def _predict_discrete_sac(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    algorithm = _load_discrete_sac_algorithm(config, checkpoint_state, device=device)
    obs_tensor = _prepare_observation(obs, device=device)
    with torch.no_grad():
        actions = algorithm.model.sample_actions(obs_tensor, deterministic=deterministic).actions
    return _format_action_output(actions, discrete=True)


def _predict_td3(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    del deterministic
    algorithm = _load_td3_algorithm(config, checkpoint_state, device=device)
    obs_tensor = _prepare_observation(obs, device=device)
    low, high = _continuous_action_bounds(config, device=device)
    with torch.no_grad():
        normalized_actions = algorithm.model.actor(obs_tensor)
        actions = _scale_continuous_actions(normalized_actions, low=low, high=high)
    return _format_action_output(actions, discrete=False)


def _predict_td3_bc(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    del deterministic
    algorithm = _load_td3_bc_algorithm(config, checkpoint_state, device=device)
    obs_tensor = _prepare_observation(obs, device=device)
    low, high = _continuous_action_bounds(config, device=device)
    with torch.no_grad():
        normalized_actions = algorithm.model.actor(obs_tensor)
        actions = _scale_continuous_actions(normalized_actions, low=low, high=high)
    return _format_action_output(actions, discrete=False)


_ALGORITHM_REGISTRY: dict[str, AlgorithmSpec] = {
    "a2c": AlgorithmSpec(
        name="a2c",
        train_fn=train_a2c,
        evaluate_fn=_evaluate_a2c,
        predict_fn=_predict_a2c,
    ),
    "awac": AlgorithmSpec(
        name="awac",
        train_fn=train_awac,
        evaluate_fn=_evaluate_awac,
        predict_fn=_predict_awac,
    ),
    "crr": AlgorithmSpec(
        name="crr",
        train_fn=train_crr,
        evaluate_fn=_evaluate_crr,
        predict_fn=_predict_crr,
    ),
    "bc": AlgorithmSpec(
        name="bc",
        train_fn=train_bc,
        evaluate_fn=_evaluate_bc,
        predict_fn=_predict_bc,
    ),
    "bcq": AlgorithmSpec(
        name="bcq",
        train_fn=train_bcq,
        evaluate_fn=_evaluate_bcq,
        predict_fn=_predict_bcq,
    ),
    "bear": AlgorithmSpec(
        name="bear",
        train_fn=train_bear,
        evaluate_fn=_evaluate_bear,
        predict_fn=_predict_bear,
    ),
    "her": AlgorithmSpec(
        name="her",
        train_fn=train_her,
        evaluate_fn=_evaluate_her,
        predict_fn=_predict_her,
    ),
    "ppo": AlgorithmSpec(
        name="ppo",
        train_fn=train_ppo,
        evaluate_fn=_evaluate_ppo,
        predict_fn=_predict_ppo,
    ),
    "trpo": AlgorithmSpec(
        name="trpo",
        train_fn=train_trpo,
        evaluate_fn=_evaluate_trpo,
        predict_fn=_predict_trpo,
    ),
    "recurrent_ppo": AlgorithmSpec(
        name="recurrent_ppo",
        train_fn=train_recurrent_ppo,
        evaluate_fn=_evaluate_recurrent_ppo,
        predict_fn=_predict_recurrent_ppo,
    ),
    "dqn": AlgorithmSpec(
        name="dqn",
        train_fn=train_dqn,
        evaluate_fn=_evaluate_dqn,
        predict_fn=_predict_dqn,
    ),
    "c51_dqn": AlgorithmSpec(
        name="c51_dqn",
        train_fn=train_dqn,
        evaluate_fn=_evaluate_c51_dqn,
        predict_fn=_predict_c51_dqn,
    ),
    "n_step_dqn": AlgorithmSpec(
        name="n_step_dqn",
        train_fn=train_dqn,
        evaluate_fn=_evaluate_dqn,
        predict_fn=_predict_dqn,
    ),
    "noisy_dqn": AlgorithmSpec(
        name="noisy_dqn",
        train_fn=train_dqn,
        evaluate_fn=_evaluate_dqn,
        predict_fn=_predict_dqn,
    ),
    "prioritized_dqn": AlgorithmSpec(
        name="prioritized_dqn",
        train_fn=train_dqn,
        evaluate_fn=_evaluate_dqn,
        predict_fn=_predict_dqn,
    ),
    "rainbow_dqn": AlgorithmSpec(
        name="rainbow_dqn",
        train_fn=train_dqn,
        evaluate_fn=_evaluate_dqn,
        predict_fn=_predict_dqn,
    ),
    "qr_dqn": AlgorithmSpec(
        name="qr_dqn",
        train_fn=train_dqn,
        evaluate_fn=_evaluate_qr_dqn,
        predict_fn=_predict_qr_dqn,
    ),
    "iqn": AlgorithmSpec(
        name="iqn",
        train_fn=train_dqn,
        evaluate_fn=_evaluate_iqn,
        predict_fn=_predict_iqn,
    ),
    "cal_ql": AlgorithmSpec(
        name="cal_ql",
        train_fn=train_cal_ql,
        evaluate_fn=_evaluate_cal_ql,
        predict_fn=_predict_cal_ql,
    ),
    "awr": AlgorithmSpec(
        name="awr",
        train_fn=train_awr,
        evaluate_fn=_evaluate_awr,
        predict_fn=_predict_awr,
    ),
    "marwil": AlgorithmSpec(
        name="marwil",
        train_fn=train_marwil,
        evaluate_fn=_evaluate_marwil,
        predict_fn=_predict_marwil,
    ),
    "iql": AlgorithmSpec(
        name="iql",
        train_fn=train_iql,
        evaluate_fn=_evaluate_iql,
        predict_fn=_predict_iql,
    ),
    "xql": AlgorithmSpec(
        name="xql",
        train_fn=train_xql,
        evaluate_fn=_evaluate_xql,
        predict_fn=_predict_xql,
    ),
    "ddpg": AlgorithmSpec(
        name="ddpg",
        train_fn=train_ddpg,
        evaluate_fn=_evaluate_ddpg,
        predict_fn=_predict_ddpg,
    ),
    "drqv2": AlgorithmSpec(
        name="drqv2",
        train_fn=train_drqv2,
        evaluate_fn=_evaluate_drqv2,
        predict_fn=_predict_drqv2,
    ),
    "double_dqn": AlgorithmSpec(
        name="double_dqn",
        train_fn=train_dqn,
        evaluate_fn=_evaluate_dqn,
        predict_fn=_predict_dqn,
    ),
    "dueling_dqn": AlgorithmSpec(
        name="dueling_dqn",
        train_fn=train_dqn,
        evaluate_fn=_evaluate_dqn,
        predict_fn=_predict_dqn,
    ),
    "sac": AlgorithmSpec(
        name="sac",
        train_fn=train_sac,
        evaluate_fn=_evaluate_sac,
        predict_fn=_predict_sac,
    ),
    "rlpd": AlgorithmSpec(
        name="rlpd",
        train_fn=train_rlpd,
        evaluate_fn=_evaluate_rlpd,
        predict_fn=_predict_rlpd,
    ),
    "cql": AlgorithmSpec(
        name="cql",
        train_fn=train_cql,
        evaluate_fn=_evaluate_cql,
        predict_fn=_predict_cql,
    ),
    "crossq": AlgorithmSpec(
        name="crossq",
        train_fn=train_crossq,
        evaluate_fn=_evaluate_crossq,
        predict_fn=_predict_crossq,
    ),
    "discrete_sac": AlgorithmSpec(
        name="discrete_sac",
        train_fn=train_discrete_sac,
        evaluate_fn=_evaluate_discrete_sac,
        predict_fn=_predict_discrete_sac,
    ),
    "tqc": AlgorithmSpec(
        name="tqc",
        train_fn=train_tqc,
        evaluate_fn=_evaluate_tqc,
        predict_fn=_predict_tqc,
    ),
    "redq": AlgorithmSpec(
        name="redq",
        train_fn=train_redq,
        evaluate_fn=_evaluate_redq,
        predict_fn=_predict_redq,
    ),
    "edac": AlgorithmSpec(
        name="edac",
        train_fn=train_edac,
        evaluate_fn=_evaluate_edac,
        predict_fn=_predict_edac,
    ),
    "td3": AlgorithmSpec(
        name="td3",
        train_fn=train_td3,
        evaluate_fn=_evaluate_td3,
        predict_fn=_predict_td3,
    ),
    "td3_bc": AlgorithmSpec(
        name="td3_bc",
        train_fn=train_td3_bc,
        evaluate_fn=_evaluate_td3_bc,
        predict_fn=_predict_td3_bc,
    ),
    "rebrac": AlgorithmSpec(
        name="rebrac",
        train_fn=train_rebrac,
        evaluate_fn=_evaluate_rebrac,
        predict_fn=_predict_rebrac,
    ),
}


def get_algorithm_spec(name: str) -> AlgorithmSpec:
    try:
        return _ALGORITHM_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"unknown algorithm: {name!r}") from exc


def list_algorithm_specs() -> tuple[AlgorithmSpec, ...]:
    return tuple(_ALGORITHM_REGISTRY.values())
