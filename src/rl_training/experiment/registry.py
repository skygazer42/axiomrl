from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np
import torch

from rl_training.algorithms.a2c import A2C as A2CAlgorithm
from rl_training.algorithms.ars import ARS as ARSAlgorithm
from rl_training.algorithms.openai_es import OpenAIES as OpenAIESAlgorithm
from rl_training.algorithms.impala import IMPALA as IMPALAAlgorithm
from rl_training.algorithms.appo import APPO as APPOAlgorithm
from rl_training.algorithms.awr import AWR as AWRAlgorithm
from rl_training.algorithms.awac import AWAC as AWACAlgorithm
from rl_training.algorithms.marwil import MARWIL as MARWILAlgorithm
from rl_training.algorithms.bc import BC as BCAlgorithm
from rl_training.algorithms.decision_transformer import DecisionTransformer as DecisionTransformerAlgorithm
from rl_training.algorithms.mopo import MOPO as MOPOAlgorithm
from rl_training.algorithms.pets import PETS as PETSAlgorithm
from rl_training.algorithms.bcq import BCQ as BCQAlgorithm
from rl_training.algorithms.bear import BEAR as BEARAlgorithm
from rl_training.algorithms.cal_ql import CalQL as CalQLAlgorithm
from rl_training.algorithms.c51_dqn import C51DQN as C51DQNAlgorithm
from rl_training.algorithms.crossq import CrossQ as CrossQAlgorithm
from rl_training.algorithms.crr import CRR as CRRAlgorithm
from rl_training.algorithms.cql import CQL as CQLAlgorithm
from rl_training.algorithms.curl import CURL as CURLAlgorithm
from rl_training.algorithms.d4pg import D4PG as D4PGAlgorithm
from rl_training.algorithms.ddpg import DDPG as DDPGAlgorithm
from rl_training.algorithms.drqn import DRQN as DRQNAlgorithm
from rl_training.algorithms.agent57 import Agent57 as Agent57Algorithm
from rl_training.algorithms.r2d2 import R2D2 as R2D2Algorithm
from rl_training.algorithms.naf import NAF as NAFAlgorithm
from rl_training.algorithms.edac import EDAC as EDACAlgorithm
from rl_training.algorithms.drq import DrQ as DrQAlgorithm
from rl_training.algorithms.drqv2 import DrQv2 as DrQv2Algorithm
from rl_training.algorithms.discrete_sac import DiscreteSAC as DiscreteSACAlgorithm
from rl_training.algorithms.efficientzero import EfficientZero as EfficientZeroAlgorithm
from rl_training.algorithms.ppg import PPG as PPGAlgorithm
from rl_training.algorithms.scalezero import ScaleZero as ScaleZeroAlgorithm
from rl_training.algorithms.dqn import DQN as DQNAlgorithm
from rl_training.algorithms.dqn import AdvantageLearningDQN as AdvantageLearningDQNAlgorithm
from rl_training.algorithms.dqn import BoltzmannDQN as BoltzmannDQNAlgorithm
from rl_training.algorithms.dqn import BoltzmannDoubleDQN as BoltzmannDoubleDQNAlgorithm
from rl_training.algorithms.dqn import CQLDQN as CQLDQNAlgorithm
from rl_training.algorithms.dqn import CQLDoubleDQN as CQLDoubleDQNAlgorithm
from rl_training.algorithms.dqn import ClippedDoubleDQN as ClippedDoubleDQNAlgorithm
from rl_training.algorithms.dqn import DoubleDQN as DoubleDQNAlgorithm
from rl_training.algorithms.dqn import DuelingDQN as DuelingDQNAlgorithm
from rl_training.algorithms.dqn import ExpectedDoubleDQN as ExpectedDoubleDQNAlgorithm
from rl_training.algorithms.dqn import ExpectedSARSA as ExpectedSARSAAlgorithm
from rl_training.algorithms.her import HER as HERAlgorithm
from rl_training.algorithms.iql import IQL as IQLAlgorithm
from rl_training.algorithms.iqn import IQN as IQNAlgorithm
from rl_training.algorithms.dqn import HystereticDQN as HystereticDQNAlgorithm
from rl_training.algorithms.dqn import MunchausenDQN as MunchausenDQNAlgorithm
from rl_training.algorithms.dqn import MunchausenDoubleDQN as MunchausenDoubleDQNAlgorithm
from rl_training.algorithms.dqn import PersistentAdvantageLearningDQN as PersistentAdvantageLearningDQNAlgorithm
from rl_training.algorithms.xql import XQL as XQLAlgorithm
from rl_training.algorithms.rlpd import RLPD as RLPDAlgorithm
from rl_training.algorithms.dqn import MellowmaxDQN as MellowmaxDQNAlgorithm
from rl_training.algorithms.dqn import NoisyDQN as NoisyDQNAlgorithm
from rl_training.algorithms.dqn import PrioritizedDQN as PrioritizedDQNAlgorithm
from rl_training.algorithms.dqn import RainbowDQN as RainbowDQNAlgorithm
from rl_training.algorithms.dqn import SoftDQN as SoftDQNAlgorithm
from rl_training.algorithms.dqn import SoftDoubleDQN as SoftDoubleDQNAlgorithm
from rl_training.algorithms.diamond import Diamond as DiamondAlgorithm
from rl_training.algorithms.horizon_imagination import HorizonImagination as HorizonImaginationAlgorithm
from rl_training.algorithms.po_dreamer import PODreamer as PODreamerAlgorithm
from rl_training.algorithms.twisted import Twisted as TwistedAlgorithm
from rl_training.algorithms.dreamer import Dreamer as DreamerAlgorithm
from rl_training.algorithms.dreamerv3 import DreamerV3 as DreamerV3Algorithm
from rl_training.algorithms.eadream import EADream as EADreamAlgorithm
from rl_training.algorithms.mow import MoW as MoWAlgorithm
from rl_training.algorithms.fqf import FQF as FQFAlgorithm
from rl_training.algorithms.jowa import JOWA as JOWAAlgorithm
from rl_training.algorithms.gail import GAIL as GAILAlgorithm
from rl_training.algorithms.gumbel_muzero import GumbelMuZero as GumbelMuZeroAlgorithm
from rl_training.algorithms.mbpo import MBPO as MBPOAlgorithm
from rl_training.algorithms.muzero import MuZero as MuZeroAlgorithm
from rl_training.algorithms.muzero import MuZeroMCTSConfig
from rl_training.algorithms.ppo import PPO as PPOAlgorithm
from rl_training.algorithms.qr_dqn import QRDQN as QRDQNAlgorithm
from rl_training.algorithms.spr import SPR as SPRAlgorithm
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
from rl_training.models.cnn import (
    CNNC51QNetwork,
    CNNDuelingNoisyQNetwork,
    CNNDuelingQNetwork,
    CNNFQFNetwork,
    CNNIQNetwork,
    CNNJOWAQNetwork,
    CNNNoisyQNetwork,
    CNNActorCritic,
    CNNPPGModel,
    CNNCURLModel,
    CNNDrQModel,
    CNNDrQv2Model,
    CNNQNetwork,
    CNNQRQNetwork,
    CNNSPRQNetwork,
)
from rl_training.models.decision_transformer import DecisionTransformerModel
from rl_training.models.mlp_actor_critic import MLPActorCritic
from rl_training.models.mlp_ars import MLPARSModel
from rl_training.models.mlp_bc import MLPBCModel
from rl_training.models.mlp_bcq import MLPBCQModel
from rl_training.models.mlp_bear import MLPBEARModel
from rl_training.models.mlp_c51_q_network import MLPC51QNetwork
from rl_training.models.mlp_crossq import MLPCrossQModel
from rl_training.models.mlp_d4pg import MLPD4PGModel
from rl_training.models.mlp_ddpg import MLPDDPGModel
from rl_training.models.mlp_discrete_sac import MLPDiscreteSACModel
from rl_training.models.mlp_dueling_noisy_q_network import MLPDuelingNoisyQNetwork
from rl_training.models.mlp_dueling_q_network import MLPDuelingQNetwork
from rl_training.models.mlp_iql import MLPIQLModel
from rl_training.models.mlp_iqn_network import MLPIQNetwork
from rl_training.models.mlp_mopo import MLPMOPOEnsembleModel
from rl_training.models.mlp_naf import MLPNAFModel
from rl_training.models.mlp_noisy_q_network import MLPNoisyQNetwork
from rl_training.models.mlp_ppg import MLPPPGModel
from rl_training.models.mlp_q_network import MLPQNetwork
from rl_training.models.mlp_fqf_network import MLPFQFNetwork
from rl_training.models.dreamer import DreamerModel
from rl_training.models.eadream import EADreamModel
from rl_training.models.mow import MoWModel
from rl_training.models.po_dreamer import PODreamerModel
from rl_training.models.mlp_gail_discriminator import CNNGAILDiscriminator, MLPGAILDiscriminator
from rl_training.models.muzero import MuZeroModel
from rl_training.models.scalezero import ScaleZeroModel
from rl_training.models.rnd import RNDModel
from rl_training.models.mlp_qr_q_network import MLPQRQNetwork
from rl_training.models.mlp_redq import MLPREDQModel
from rl_training.models.mlp_sac import MLPSACModel
from rl_training.models.mlp_tqc import MLPTQCModel
from rl_training.models.mlp_td3 import MLPTD3Model
from rl_training.models.recurrent import LSTMActorCritic, LSTMQNetwork
from rl_training.runtime.a2c_trainer import _evaluate_policy as _evaluate_a2c_policy
from rl_training.runtime.a2c_trainer import train_a2c
from rl_training.runtime.ars_trainer import _evaluate_ars_policy, train_ars
from rl_training.runtime.openai_es_trainer import _evaluate_openai_es_policy, train_openai_es
from rl_training.runtime.impala_trainer import _evaluate_impala_policy, train_impala
from rl_training.runtime.appo_trainer import _evaluate_appo_policy, train_appo
from rl_training.runtime.awr_trainer import train_awr
from rl_training.runtime.awac_trainer import train_awac
from rl_training.runtime.marwil_trainer import train_marwil
from rl_training.runtime.bc_trainer import _evaluate_bc_policy, train_bc
from rl_training.runtime.decision_transformer_trainer import (
    _build_autoregressive_window,
    _evaluate_decision_transformer_policy,
    train_decision_transformer,
)
from rl_training.runtime.mopo_trainer import train_mopo
from rl_training.runtime.mbpo_trainer import train_mbpo
from rl_training.runtime.pets_trainer import _evaluate_pets_policy, train_pets
from rl_training.runtime.bcq_trainer import _evaluate_bcq_policy, train_bcq
from rl_training.runtime.bear_trainer import train_bear
from rl_training.runtime.cal_ql_trainer import train_cal_ql
from rl_training.runtime.cql_trainer import train_cql
from rl_training.runtime.curl_trainer import _evaluate_curl_policy, train_curl
from rl_training.runtime.crossq_trainer import _evaluate_crossq_policy, train_crossq
from rl_training.runtime.crr_trainer import train_crr
from rl_training.runtime.d4pg_trainer import _evaluate_d4pg_policy, train_d4pg
from rl_training.runtime.ddpg_trainer import _evaluate_ddpg_policy, train_ddpg
from rl_training.runtime.drqn_trainer import _evaluate_drqn_policy, train_drqn
from rl_training.runtime.agent57_trainer import train_agent57
from rl_training.runtime.r2d2_trainer import _evaluate_r2d2_policy, train_r2d2
from rl_training.runtime.edac_trainer import train_edac
from rl_training.runtime.drq_trainer import _evaluate_drq_policy, train_drq
from rl_training.runtime.drqv2_trainer import _evaluate_drqv2_policy, train_drqv2
from rl_training.runtime.discrete_sac_trainer import _evaluate_discrete_sac_policy, train_discrete_sac
from rl_training.runtime.dqn_trainer import _evaluate_q_policy, train_dqn
from rl_training.runtime.apex_dqn_trainer import train_apex_dqn
from rl_training.runtime.dreamer_trainer import train_dreamer
from rl_training.runtime.efficientzero_trainer import train_efficientzero
from rl_training.runtime.gail_trainer import train_gail
from rl_training.runtime.muzero_trainer import _evaluate_muzero_policy, train_muzero
from rl_training.runtime.naf_trainer import _evaluate_naf_policy, train_naf
from rl_training.runtime.her_trainer import _evaluate_her_policy, _infer_her_spaces, train_her
from rl_training.runtime.iql_trainer import _evaluate_iql_policy, train_iql
from rl_training.runtime.xql_trainer import train_xql
from rl_training.runtime.ppo_trainer import _evaluate_policy, train_ppo
from rl_training.runtime.ppg_trainer import _evaluate_ppg_policy, train_ppg
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
    if len(obs_shape) == 1:
        policy: MLPActorCritic | CNNActorCritic = MLPActorCritic(
            obs_dim=obs_shape[0],
            action_dim=action_dim,
            hidden_sizes=tuple(config.algo_kwargs.get("hidden_sizes", (64, 64))),
        ).to(device)
    else:
        policy = CNNActorCritic(
            obs_shape=obs_shape,
            action_dim=action_dim,
            hidden_sizes=tuple(
                config.algo_kwargs.get("head_hidden_sizes", config.algo_kwargs.get("hidden_sizes", (512,)))
            ),
            features_dim=int(config.algo_kwargs.get("features_dim", 512)),
        ).to(device)
    algorithm = A2CAlgorithm(
        policy=policy,
        learning_rate=float(config.algo_kwargs.get("learning_rate", 3e-4)),
        ent_coef=float(config.algo_kwargs.get("ent_coef", 0.01)),
        vf_coef=float(config.algo_kwargs.get("vf_coef", 0.5)),
        max_grad_norm=float(config.algo_kwargs.get("max_grad_norm", 0.5)),
    )
    algorithm.load_state_dict(checkpoint_state.algorithm_state)
    return algorithm


def _load_ars_algorithm(config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device) -> ARSAlgorithm:
    obs_dim, action_dim = _infer_continuous_env_spaces(config)
    hidden_sizes = tuple(config.algo_kwargs.get("hidden_sizes", (64, 64)))
    algorithm = ARSAlgorithm(
        model=MLPARSModel(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=hidden_sizes).to(device),
        step_size=float(config.algo_kwargs.get("step_size", 0.02)),
        noise_std=float(config.algo_kwargs.get("noise_std", 0.03)),
        num_top_directions=int(
            config.algo_kwargs.get(
                "num_top_directions",
                max(1, int(config.algo_kwargs.get("num_directions", 8)) // 2),
            )
        ),
    )
    algorithm.load_state_dict(checkpoint_state.algorithm_state)
    return algorithm


def _load_openai_es_algorithm(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    *,
    device: torch.device,
) -> OpenAIESAlgorithm:
    obs_dim, action_dim = _infer_continuous_env_spaces(config)
    hidden_sizes = tuple(config.algo_kwargs.get("hidden_sizes", (64, 64)))
    algorithm = OpenAIESAlgorithm(
        model=MLPARSModel(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=hidden_sizes).to(device),
        step_size=float(config.algo_kwargs.get("step_size", 0.02)),
        noise_std=float(config.algo_kwargs.get("noise_std", 0.03)),
        weight_decay=float(config.algo_kwargs.get("weight_decay", 0.0)),
    )
    algorithm.load_state_dict(checkpoint_state.algorithm_state)
    return algorithm


def _load_impala_algorithm(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    *,
    device: torch.device,
) -> IMPALAAlgorithm:
    obs_shape, action_dim = _infer_discrete_env_spaces(config)
    if len(obs_shape) == 1:
        policy: MLPActorCritic | CNNActorCritic = MLPActorCritic(
            obs_dim=obs_shape[0],
            action_dim=action_dim,
            hidden_sizes=tuple(config.algo_kwargs.get("hidden_sizes", (64, 64))),
        ).to(device)
    else:
        policy = CNNActorCritic(
            obs_shape=obs_shape,
            action_dim=action_dim,
            hidden_sizes=tuple(
                config.algo_kwargs.get("head_hidden_sizes", config.algo_kwargs.get("hidden_sizes", (512,)))
            ),
            features_dim=int(config.algo_kwargs.get("features_dim", 512)),
        ).to(device)
    algorithm = IMPALAAlgorithm(
        policy=policy,
        learning_rate=float(config.algo_kwargs.get("learning_rate", 3e-4)),
        ent_coef=float(config.algo_kwargs.get("ent_coef", 0.01)),
        vf_coef=float(config.algo_kwargs.get("vf_coef", 0.5)),
        gamma=float(config.algo_kwargs.get("gamma", 0.99)),
        rho_clip=float(config.algo_kwargs.get("rho_clip", 1.0)),
        c_clip=float(config.algo_kwargs.get("c_clip", 1.0)),
        pg_rho_clip=float(config.algo_kwargs.get("pg_rho_clip", config.algo_kwargs.get("rho_clip", 1.0))),
        max_grad_norm=float(config.algo_kwargs.get("max_grad_norm", 0.5)),
    )
    algorithm.load_state_dict(checkpoint_state.algorithm_state)
    return algorithm


def _load_appo_algorithm(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    *,
    device: torch.device,
) -> APPOAlgorithm:
    obs_shape, action_dim = _infer_discrete_env_spaces(config)
    if len(obs_shape) != 1:
        raise ValueError("appo checkpoint loading currently supports flat observations only")
    obs_dim = obs_shape[0]
    hidden_sizes = tuple(config.algo_kwargs.get("hidden_sizes", (64, 64)))
    algorithm = APPOAlgorithm(
        policy=MLPActorCritic(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=hidden_sizes).to(device),
        learning_rate=float(config.algo_kwargs.get("learning_rate", 3e-4)),
        clip_coef=float(config.algo_kwargs.get("clip_coef", 0.2)),
        ent_coef=float(config.algo_kwargs.get("ent_coef", 0.01)),
        vf_coef=float(config.algo_kwargs.get("vf_coef", 0.5)),
        gamma=float(config.algo_kwargs.get("gamma", 0.99)),
        rho_clip=float(config.algo_kwargs.get("rho_clip", 1.0)),
        c_clip=float(config.algo_kwargs.get("c_clip", 1.0)),
        pg_rho_clip=float(config.algo_kwargs.get("pg_rho_clip", config.algo_kwargs.get("rho_clip", 1.0))),
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


def _load_decision_transformer_algorithm(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    *,
    device: torch.device,
) -> DecisionTransformerAlgorithm:
    obs_dim, action_dim = _infer_continuous_env_spaces(config)
    algorithm = DecisionTransformerAlgorithm(
        model=DecisionTransformerModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            context_length=int(config.algo_kwargs.get("context_length", 20)),
            hidden_size=int(config.algo_kwargs.get("hidden_size", 128)),
            num_layers=int(config.algo_kwargs.get("num_layers", 3)),
            num_heads=int(config.algo_kwargs.get("num_heads", 4)),
            max_timestep=int(config.algo_kwargs.get("max_timestep", 1024)),
            dropout=float(config.algo_kwargs.get("dropout", 0.1)),
        ).to(device),
        learning_rate=float(config.algo_kwargs.get("learning_rate", 1e-4)),
    )
    algorithm.load_state_dict(checkpoint_state.algorithm_state)
    return algorithm


def _load_mopo_algorithm(config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device) -> MOPOAlgorithm:
    obs_dim, action_dim = _infer_continuous_env_spaces(config)
    hidden_sizes = tuple(config.algo_kwargs.get("hidden_sizes", (256, 256)))
    model_hidden_sizes = tuple(config.algo_kwargs.get("model_hidden_sizes", hidden_sizes))
    algorithm = MOPOAlgorithm(
        policy_model=MLPSACModel(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=hidden_sizes).to(device),
        dynamics_model=MLPMOPOEnsembleModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=model_hidden_sizes,
            num_ensembles=int(config.algo_kwargs.get("num_ensembles", 5)),
        ).to(device),
        policy_learning_rate=float(config.algo_kwargs.get("policy_learning_rate", 3e-4)),
        model_learning_rate=float(config.algo_kwargs.get("model_learning_rate", 1e-3)),
        gamma=float(config.algo_kwargs.get("gamma", 0.99)),
        alpha=float(config.algo_kwargs.get("alpha", 0.2)),
        tau=float(config.algo_kwargs.get("tau", 0.005)),
        penalty_coef=float(config.algo_kwargs.get("penalty_coef", 1.0)),
    )
    algorithm.load_state_dict(checkpoint_state.algorithm_state)
    return algorithm


def _load_mbpo_algorithm(config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device) -> MBPOAlgorithm:
    obs_dim, action_dim = _infer_continuous_env_spaces(config)
    hidden_sizes = tuple(config.algo_kwargs.get("hidden_sizes", (256, 256)))
    model_hidden_sizes = tuple(config.algo_kwargs.get("model_hidden_sizes", hidden_sizes))
    algorithm = MBPOAlgorithm(
        policy_model=MLPSACModel(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=hidden_sizes).to(device),
        dynamics_model=MLPMOPOEnsembleModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=model_hidden_sizes,
            num_ensembles=int(config.algo_kwargs.get("num_ensembles", 5)),
        ).to(device),
        policy_learning_rate=float(config.algo_kwargs.get("policy_learning_rate", 3e-4)),
        model_learning_rate=float(config.algo_kwargs.get("model_learning_rate", 1e-3)),
        gamma=float(config.algo_kwargs.get("gamma", 0.99)),
        alpha=float(config.algo_kwargs.get("alpha", 0.2)),
        tau=float(config.algo_kwargs.get("tau", 0.005)),
    )
    algorithm.load_state_dict(checkpoint_state.algorithm_state)
    return algorithm


def _load_pets_algorithm(config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device) -> PETSAlgorithm:
    obs_dim, action_dim = _infer_continuous_env_spaces(config)
    model_hidden_sizes = tuple(config.algo_kwargs.get("model_hidden_sizes", (256, 256)))
    algorithm = PETSAlgorithm(
        dynamics_model=MLPMOPOEnsembleModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=model_hidden_sizes,
            num_ensembles=int(config.algo_kwargs.get("num_ensembles", 5)),
        ).to(device),
        learning_rate=float(config.algo_kwargs.get("model_learning_rate", 1e-3)),
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


def _load_gail_algorithm(config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device) -> GAILAlgorithm:
    obs_shape, action_dim = _infer_discrete_env_spaces(config)
    if len(obs_shape) == 1:
        hidden_sizes = tuple(config.algo_kwargs.get("hidden_sizes", (64, 64)))
        policy = MLPActorCritic(obs_dim=obs_shape[0], action_dim=action_dim, hidden_sizes=hidden_sizes).to(device)
        discriminator_hidden_sizes = tuple(config.algo_kwargs.get("discriminator_hidden_sizes", hidden_sizes))
        discriminator = MLPGAILDiscriminator(
            obs_dim=obs_shape[0],
            action_dim=action_dim,
            hidden_sizes=discriminator_hidden_sizes,
        ).to(device)
    else:
        policy = CNNActorCritic(
            obs_shape=obs_shape,
            action_dim=action_dim,
            hidden_sizes=tuple(config.algo_kwargs.get("head_hidden_sizes", config.algo_kwargs.get("hidden_sizes", (512,)))),
            features_dim=int(config.algo_kwargs.get("features_dim", 512)),
        ).to(device)
        discriminator_hidden_sizes = tuple(
            config.algo_kwargs.get(
                "discriminator_head_hidden_sizes",
                config.algo_kwargs.get("head_hidden_sizes", config.algo_kwargs.get("hidden_sizes", (512,))),
            )
        )
        discriminator = CNNGAILDiscriminator(
            obs_shape=obs_shape,
            action_dim=action_dim,
            hidden_sizes=discriminator_hidden_sizes,
            features_dim=int(config.algo_kwargs.get("discriminator_features_dim", config.algo_kwargs.get("features_dim", 512))),
        ).to(device)

    algorithm = GAILAlgorithm(
        policy=policy,
        discriminator=discriminator,
        learning_rate=float(config.algo_kwargs.get("learning_rate", 3e-4)),
        clip_coef=float(config.algo_kwargs.get("clip_coef", 0.2)),
        ent_coef=float(config.algo_kwargs.get("ent_coef", 0.01)),
        vf_coef=float(config.algo_kwargs.get("vf_coef", 0.5)),
        discriminator_learning_rate=float(config.algo_kwargs.get("discriminator_learning_rate", config.algo_kwargs.get("learning_rate", 3e-4))),
        max_grad_norm=float(config.algo_kwargs.get("max_grad_norm", 0.5)),
    )
    algorithm.load_state_dict(checkpoint_state.algorithm_state)
    return algorithm


def _load_dreamer_algorithm(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    *,
    device: torch.device,
) -> (
    DreamerAlgorithm
    | DreamerV3Algorithm
    | DiamondAlgorithm
    | HorizonImaginationAlgorithm
    | PODreamerAlgorithm
    | TwistedAlgorithm
    | EADreamAlgorithm
    | MoWAlgorithm
):
    obs_shape, action_dim = _infer_discrete_env_spaces(config)
    if len(obs_shape) != 3:
        raise ValueError("dreamer checkpoint loading requires channel-first image observations")

    if config.algo == "po_dreamer":
        model_cls = PODreamerModel
    elif config.algo == "mow":
        model_cls = MoWModel
    elif config.algo == "eadream":
        model_cls = EADreamModel
    else:
        model_cls = DreamerModel
    model_kwargs = {
        "obs_shape": obs_shape,
        "action_dim": action_dim,
        "features_dim": int(config.algo_kwargs.get("features_dim", 128)),
        "action_embed_dim": int(config.algo_kwargs.get("action_embed_dim", 32)),
        "actor_hidden_sizes": tuple(config.algo_kwargs.get("actor_hidden_sizes", (256, 256))),
        "critic_hidden_sizes": tuple(config.algo_kwargs.get("critic_hidden_sizes", (256, 256))),
        "reward_hidden_sizes": tuple(config.algo_kwargs.get("reward_hidden_sizes", (256, 256))),
    }
    if model_cls is PODreamerModel:
        model_kwargs["memory_dim"] = int(config.algo_kwargs.get("memory_dim", 64))
        model_kwargs["memory_hidden_size"] = int(config.algo_kwargs.get("memory_hidden_size", 128))
        model_kwargs["memory_mix"] = float(config.algo_kwargs.get("memory_mix", 0.35))
    if model_cls is MoWModel:
        model_kwargs["num_experts"] = int(config.algo_kwargs.get("num_experts", 4))
        model_kwargs["gating_hidden_size"] = int(config.algo_kwargs.get("gating_hidden_size", 128))
    if model_cls is EADreamModel:
        model_kwargs["event_hidden_sizes"] = tuple(config.algo_kwargs.get("event_hidden_sizes", (128,)))
        model_kwargs["event_scale"] = float(config.algo_kwargs.get("event_scale", 1.0))
    model = model_cls(**model_kwargs).to(device)
    if config.algo == "dreamerv3":
        algorithm_cls = DreamerV3Algorithm
    elif config.algo == "diamond":
        algorithm_cls = DiamondAlgorithm
    elif config.algo == "horizon_imagination":
        algorithm_cls = HorizonImaginationAlgorithm
    elif config.algo == "po_dreamer":
        algorithm_cls = PODreamerAlgorithm
    elif config.algo == "twisted":
        algorithm_cls = TwistedAlgorithm
    elif config.algo == "eadream":
        algorithm_cls = EADreamAlgorithm
    elif config.algo == "mow":
        algorithm_cls = MoWAlgorithm
    else:
        algorithm_cls = DreamerAlgorithm
    algorithm_kwargs = {
        "model": model,
        "world_model_learning_rate": float(config.algo_kwargs.get("world_model_learning_rate", 1e-3)),
        "actor_learning_rate": float(config.algo_kwargs.get("actor_learning_rate", 3e-4)),
        "critic_learning_rate": float(config.algo_kwargs.get("critic_learning_rate", 3e-4)),
        "gamma": float(config.algo_kwargs.get("gamma", 0.99)),
        "entropy_coef": float(config.algo_kwargs.get("entropy_coef", 1e-3)),
    }
    if algorithm_cls is DreamerV3Algorithm:
        algorithm_kwargs["unimix_ratio"] = float(config.algo_kwargs.get("unimix_ratio", 0.01))
    if algorithm_cls in {DiamondAlgorithm, HorizonImaginationAlgorithm}:
        algorithm_kwargs["denoising_loss_coef"] = float(config.algo_kwargs.get("denoising_loss_coef", 0.5))
        algorithm_kwargs["noise_scale"] = float(config.algo_kwargs.get("noise_scale", 0.15))
        algorithm_kwargs["denoiser_hidden_channels"] = int(config.algo_kwargs.get("denoiser_hidden_channels", 64))
    if algorithm_cls is HorizonImaginationAlgorithm:
        algorithm_kwargs["stabilization_coef"] = float(config.algo_kwargs.get("stabilization_coef", 0.25))
        algorithm_kwargs["schedule_bias"] = float(config.algo_kwargs.get("schedule_bias", 0.5))
        algorithm_kwargs["subframe_budget_ratio"] = float(config.algo_kwargs.get("subframe_budget_ratio", 0.5))
    if algorithm_cls is PODreamerAlgorithm:
        algorithm_kwargs["memory_loss_coef"] = float(config.algo_kwargs.get("memory_loss_coef", 0.5))
    if algorithm_cls is TwistedAlgorithm:
        algorithm_kwargs["reuse_loss_coef"] = float(config.algo_kwargs.get("reuse_loss_coef", 0.5))
        algorithm_kwargs["reuse_threshold"] = float(config.algo_kwargs.get("reuse_threshold", 0.03))
        algorithm_kwargs["transport_temperature"] = float(
            config.algo_kwargs.get("transport_temperature", 0.5)
        )
    if algorithm_cls is EADreamAlgorithm:
        algorithm_kwargs["event_loss_coef"] = float(config.algo_kwargs.get("event_loss_coef", 0.5))
        algorithm_kwargs["event_threshold"] = float(config.algo_kwargs.get("event_threshold", 0.01))
    algorithm = algorithm_cls(**algorithm_kwargs)
    algorithm.load_state_dict(checkpoint_state.algorithm_state)
    return algorithm


def _load_muzero_algorithm(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    *,
    device: torch.device,
) -> MuZeroAlgorithm | ScaleZeroAlgorithm:
    obs_shape, action_dim = _infer_discrete_env_spaces(config)
    if len(obs_shape) != 3:
        raise ValueError("muzero checkpoint loading requires channel-first image observations")

    model_cls = ScaleZeroModel if config.algo == "scalezero" else MuZeroModel
    model_kwargs = {
        "obs_shape": obs_shape,
        "action_dim": action_dim,
        "latent_dim": int(config.algo_kwargs.get("latent_dim", 256)),
        "action_embed_dim": int(config.algo_kwargs.get("action_embed_dim", 64)),
        "dynamics_hidden_sizes": tuple(config.algo_kwargs.get("dynamics_hidden_sizes", (256,))),
        "prediction_hidden_sizes": tuple(config.algo_kwargs.get("prediction_hidden_sizes", (256,))),
        "normalize_latent": bool(config.algo_kwargs.get("normalize_latent", True)),
    }
    if model_cls is ScaleZeroModel:
        model_kwargs["num_experts"] = int(config.algo_kwargs.get("num_experts", 4))
        model_kwargs["gating_hidden_size"] = int(config.algo_kwargs.get("gating_hidden_size", 128))
    model = model_cls(**model_kwargs).to(device)
    algorithm_cls = ScaleZeroAlgorithm if config.algo == "scalezero" else MuZeroAlgorithm
    algorithm = algorithm_cls(
        model=model,
        learning_rate=float(config.algo_kwargs.get("learning_rate", 1e-3)),
        gamma=float(config.algo_kwargs.get("gamma", 0.997)),
        mcts_config=MuZeroMCTSConfig(
            num_simulations=int(config.algo_kwargs.get("num_simulations", 25)),
            pb_c_base=float(config.algo_kwargs.get("pb_c_base", 19652.0)),
            pb_c_init=float(config.algo_kwargs.get("pb_c_init", 1.25)),
            root_dirichlet_alpha=float(config.algo_kwargs.get("root_dirichlet_alpha", 0.3)),
            root_exploration_fraction=float(config.algo_kwargs.get("root_exploration_fraction", 0.25)),
        ),
        unroll_steps=int(config.algo_kwargs.get("unroll_steps", 5)),
        value_loss_weight=float(config.algo_kwargs.get("value_loss_weight", 1.0)),
        reward_loss_weight=float(config.algo_kwargs.get("reward_loss_weight", 1.0)),
        policy_loss_weight=float(config.algo_kwargs.get("policy_loss_weight", 1.0)),
        max_grad_norm=float(config.algo_kwargs.get("max_grad_norm", 10.0)),
    )
    algorithm.load_state_dict(checkpoint_state.algorithm_state)
    return algorithm


def _load_gumbel_muzero_algorithm(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    *,
    device: torch.device,
) -> GumbelMuZeroAlgorithm:
    obs_shape, action_dim = _infer_discrete_env_spaces(config)
    if len(obs_shape) != 3:
        raise ValueError("gumbel_muzero checkpoint loading requires channel-first image observations")

    model = MuZeroModel(
        obs_shape=obs_shape,
        action_dim=action_dim,
        latent_dim=int(config.algo_kwargs.get("latent_dim", 256)),
        action_embed_dim=int(config.algo_kwargs.get("action_embed_dim", 64)),
        dynamics_hidden_sizes=tuple(config.algo_kwargs.get("dynamics_hidden_sizes", (256,))),
        prediction_hidden_sizes=tuple(config.algo_kwargs.get("prediction_hidden_sizes", (256,))),
        normalize_latent=bool(config.algo_kwargs.get("normalize_latent", True)),
    ).to(device)
    algorithm = GumbelMuZeroAlgorithm(
        model=model,
        learning_rate=float(config.algo_kwargs.get("learning_rate", 1e-3)),
        gamma=float(config.algo_kwargs.get("gamma", 0.997)),
        mcts_config=MuZeroMCTSConfig(
            num_simulations=int(config.algo_kwargs.get("num_simulations", 25)),
            pb_c_base=float(config.algo_kwargs.get("pb_c_base", 19652.0)),
            pb_c_init=float(config.algo_kwargs.get("pb_c_init", 1.25)),
            root_dirichlet_alpha=float(config.algo_kwargs.get("root_dirichlet_alpha", 0.3)),
            root_exploration_fraction=float(config.algo_kwargs.get("root_exploration_fraction", 0.25)),
        ),
        unroll_steps=int(config.algo_kwargs.get("unroll_steps", 5)),
        value_loss_weight=float(config.algo_kwargs.get("value_loss_weight", 1.0)),
        reward_loss_weight=float(config.algo_kwargs.get("reward_loss_weight", 1.0)),
        policy_loss_weight=float(config.algo_kwargs.get("policy_loss_weight", 1.0)),
        max_grad_norm=float(config.algo_kwargs.get("max_grad_norm", 10.0)),
        gumbel_scale=float(config.algo_kwargs.get("gumbel_scale", 1.0)),
    )
    algorithm.load_state_dict(checkpoint_state.algorithm_state)
    return algorithm


def _load_efficientzero_algorithm(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    *,
    device: torch.device,
) -> EfficientZeroAlgorithm:
    obs_shape, action_dim = _infer_discrete_env_spaces(config)
    if len(obs_shape) != 3:
        raise ValueError("efficientzero checkpoint loading requires channel-first image observations")

    model = MuZeroModel(
        obs_shape=obs_shape,
        action_dim=action_dim,
        latent_dim=int(config.algo_kwargs.get("latent_dim", 256)),
        action_embed_dim=int(config.algo_kwargs.get("action_embed_dim", 64)),
        dynamics_hidden_sizes=tuple(config.algo_kwargs.get("dynamics_hidden_sizes", (256,))),
        prediction_hidden_sizes=tuple(config.algo_kwargs.get("prediction_hidden_sizes", (256,))),
        normalize_latent=bool(config.algo_kwargs.get("normalize_latent", True)),
    ).to(device)
    algorithm = EfficientZeroAlgorithm(
        model=model,
        learning_rate=float(config.algo_kwargs.get("learning_rate", 1e-3)),
        gamma=float(config.algo_kwargs.get("gamma", 0.997)),
        mcts_config=MuZeroMCTSConfig(
            num_simulations=int(config.algo_kwargs.get("num_simulations", 25)),
            pb_c_base=float(config.algo_kwargs.get("pb_c_base", 19652.0)),
            pb_c_init=float(config.algo_kwargs.get("pb_c_init", 1.25)),
            root_dirichlet_alpha=float(config.algo_kwargs.get("root_dirichlet_alpha", 0.3)),
            root_exploration_fraction=float(config.algo_kwargs.get("root_exploration_fraction", 0.25)),
        ),
        unroll_steps=int(config.algo_kwargs.get("unroll_steps", 5)),
        value_loss_weight=float(config.algo_kwargs.get("value_loss_weight", 1.0)),
        reward_loss_weight=float(config.algo_kwargs.get("reward_loss_weight", 1.0)),
        policy_loss_weight=float(config.algo_kwargs.get("policy_loss_weight", 1.0)),
        consistency_loss_weight=float(config.algo_kwargs.get("consistency_loss_weight", 1.0)),
        max_grad_norm=float(config.algo_kwargs.get("max_grad_norm", 10.0)),
    )
    algorithm.load_state_dict(checkpoint_state.algorithm_state)
    return algorithm


def _load_ppg_algorithm(config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device) -> PPGAlgorithm:
    obs_shape, action_dim = _infer_discrete_env_spaces(config)
    if len(obs_shape) == 1:
        model: MLPPPGModel | CNNPPGModel = MLPPPGModel(
            obs_dim=obs_shape[0],
            action_dim=action_dim,
            hidden_sizes=tuple(config.algo_kwargs.get("hidden_sizes", (64, 64))),
        ).to(device)
    else:
        model = CNNPPGModel(
            obs_shape=obs_shape,
            action_dim=action_dim,
            hidden_sizes=tuple(
                config.algo_kwargs.get("head_hidden_sizes", config.algo_kwargs.get("hidden_sizes", (512,)))
            ),
            features_dim=int(config.algo_kwargs.get("features_dim", 512)),
        ).to(device)

    algorithm = PPGAlgorithm(
        model=model,
        learning_rate=float(config.algo_kwargs.get("learning_rate", 3e-4)),
        aux_learning_rate=float(config.algo_kwargs.get("aux_learning_rate", config.algo_kwargs.get("learning_rate", 3e-4))),
        clip_coef=float(config.algo_kwargs.get("clip_coef", 0.2)),
        ent_coef=float(config.algo_kwargs.get("ent_coef", 0.01)),
        vf_coef=float(config.algo_kwargs.get("vf_coef", 0.5)),
        aux_value_coef=float(config.algo_kwargs.get("aux_value_coef", 1.0)),
        behavior_clone_coef=float(config.algo_kwargs.get("behavior_clone_coef", 1.0)),
        value_clone_coef=float(config.algo_kwargs.get("value_clone_coef", 1.0)),
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


def _build_image_dqn_loader(
    config: TrainConfig,
    *,
    obs_shape: tuple[int, ...],
    action_dim: int,
    device: torch.device,
) -> tuple[CNNQNetwork | CNNSPRQNetwork | CNNJOWAQNetwork, type[DQNAlgorithm] | type[SPRAlgorithm] | type[JOWAAlgorithm]]:
    head_hidden_sizes = tuple(config.algo_kwargs.get("head_hidden_sizes", config.algo_kwargs.get("hidden_sizes", (512,))))
    features_dim = int(config.algo_kwargs.get("features_dim", 512))
    sigma_init = float(config.algo_kwargs.get("sigma_init", 0.5))

    if config.algo == "rainbow_dqn":
        q_network = CNNDuelingNoisyQNetwork(
            obs_shape=obs_shape,
            action_dim=action_dim,
            hidden_sizes=head_hidden_sizes,
            sigma_init=sigma_init,
            features_dim=features_dim,
        ).to(device)
        return q_network, RainbowDQNAlgorithm
    if config.algo == "dueling_dqn":
        q_network = CNNDuelingQNetwork(
            obs_shape=obs_shape,
            action_dim=action_dim,
            hidden_sizes=head_hidden_sizes,
            features_dim=features_dim,
        ).to(device)
        return q_network, DuelingDQNAlgorithm
    if config.algo == "noisy_dqn":
        q_network = CNNNoisyQNetwork(
            obs_shape=obs_shape,
            action_dim=action_dim,
            hidden_sizes=head_hidden_sizes,
            sigma_init=sigma_init,
            features_dim=features_dim,
        ).to(device)
        return q_network, NoisyDQNAlgorithm
    if config.algo == "spr":
        q_network = CNNSPRQNetwork(
            obs_shape=obs_shape,
            action_dim=action_dim,
            hidden_sizes=head_hidden_sizes,
            features_dim=features_dim,
            transition_hidden_size=int(config.algo_kwargs.get("spr_hidden_size", features_dim)),
            projection_dim=int(config.algo_kwargs.get("spr_projection_dim", 256)),
            action_embed_dim=int(config.algo_kwargs.get("spr_action_embed_dim", 64)),
        ).to(device)
        return q_network, SPRAlgorithm
    if config.algo == "jowa":
        q_network = CNNJOWAQNetwork(
            obs_shape=obs_shape,
            action_dim=action_dim,
            hidden_sizes=head_hidden_sizes,
            features_dim=features_dim,
            transition_hidden_size=int(config.algo_kwargs.get("jowa_transition_hidden_size", features_dim)),
            reward_hidden_size=int(config.algo_kwargs.get("jowa_reward_hidden_size", features_dim)),
            action_embed_dim=int(config.algo_kwargs.get("jowa_action_embed_dim", 64)),
        ).to(device)
        return q_network, JOWAAlgorithm

    q_network = CNNQNetwork(
        obs_shape=obs_shape,
        action_dim=action_dim,
        hidden_sizes=head_hidden_sizes,
        features_dim=features_dim,
    ).to(device)
    return q_network, _resolve_vector_dqn_algorithm_class(config.algo)


def _resolve_vector_dqn_algorithm_class(algo_name: str) -> type[DQNAlgorithm]:
    return {
        "apex_dqn": DoubleDQNAlgorithm,
        "double_dqn": DoubleDQNAlgorithm,
        "expected_sarsa": ExpectedSARSAAlgorithm,
        "expected_double_dqn": ExpectedDoubleDQNAlgorithm,
        "boltzmann_dqn": BoltzmannDQNAlgorithm,
        "boltzmann_double_dqn": BoltzmannDoubleDQNAlgorithm,
        "mellowmax_dqn": MellowmaxDQNAlgorithm,
        "soft_dqn": SoftDQNAlgorithm,
        "soft_double_dqn": SoftDoubleDQNAlgorithm,
        "advantage_learning_dqn": AdvantageLearningDQNAlgorithm,
        "persistent_advantage_learning_dqn": PersistentAdvantageLearningDQNAlgorithm,
        "munchausen_dqn": MunchausenDQNAlgorithm,
        "munchausen_double_dqn": MunchausenDoubleDQNAlgorithm,
        "cql_dqn": CQLDQNAlgorithm,
        "cql_double_dqn": CQLDoubleDQNAlgorithm,
        "clipped_double_dqn": ClippedDoubleDQNAlgorithm,
        "hysteretic_dqn": HystereticDQNAlgorithm,
        "prioritized_dqn": PrioritizedDQNAlgorithm,
    }.get(algo_name, DQNAlgorithm)


def _build_vector_dqn_loader(
    config: TrainConfig,
    *,
    obs_dim: int,
    action_dim: int,
    hidden_sizes: tuple[int, ...],
    device: torch.device,
) -> tuple[nn.Module, type[DQNAlgorithm]]:
    if config.algo == "rainbow_dqn":
        q_network = MLPDuelingNoisyQNetwork(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=hidden_sizes).to(device)
        return q_network, RainbowDQNAlgorithm
    if config.algo == "dueling_dqn":
        q_network = MLPDuelingQNetwork(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=hidden_sizes).to(device)
        return q_network, DuelingDQNAlgorithm
    if config.algo == "noisy_dqn":
        q_network = MLPNoisyQNetwork(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=hidden_sizes).to(device)
        return q_network, NoisyDQNAlgorithm

    q_network = MLPQNetwork(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=hidden_sizes).to(device)
    return q_network, _resolve_vector_dqn_algorithm_class(config.algo)


def _build_dqn_algorithm_kwargs(
    config: TrainConfig,
    *,
    q_network: nn.Module,
    algorithm_cls: type[DQNAlgorithm] | type[SPRAlgorithm] | type[JOWAAlgorithm],
) -> dict[str, float | int | nn.Module]:
    algorithm_kwargs: dict[str, float | int | nn.Module] = {
        "q_network": q_network,
        "learning_rate": float(config.algo_kwargs.get("learning_rate", 1e-3)),
        "gamma": float(config.algo_kwargs.get("gamma", 0.99)),
        "target_update_interval": int(config.algo_kwargs.get("target_update_interval", 250)),
    }
    if algorithm_cls in {ExpectedSARSAAlgorithm, ExpectedDoubleDQNAlgorithm}:
        algorithm_kwargs["target_epsilon"] = float(config.algo_kwargs.get("target_epsilon", 0.05))
    elif algorithm_cls in {BoltzmannDQNAlgorithm, BoltzmannDoubleDQNAlgorithm}:
        algorithm_kwargs["boltzmann_temperature"] = float(config.algo_kwargs.get("boltzmann_temperature", 0.5))
    elif algorithm_cls is MellowmaxDQNAlgorithm:
        algorithm_kwargs["mellowmax_omega"] = float(config.algo_kwargs.get("mellowmax_omega", 5.0))
    elif algorithm_cls in {SoftDQNAlgorithm, SoftDoubleDQNAlgorithm}:
        algorithm_kwargs["entropy_temperature"] = float(config.algo_kwargs.get("entropy_temperature", 0.03))
    elif algorithm_cls is AdvantageLearningDQNAlgorithm:
        algorithm_kwargs["advantage_alpha"] = float(config.algo_kwargs.get("advantage_alpha", 0.9))
    elif algorithm_cls is PersistentAdvantageLearningDQNAlgorithm:
        algorithm_kwargs["persistent_advantage_alpha"] = float(config.algo_kwargs.get("persistent_advantage_alpha", 0.9))
    elif algorithm_cls in {MunchausenDQNAlgorithm, MunchausenDoubleDQNAlgorithm}:
        algorithm_kwargs["munchausen_alpha"] = float(config.algo_kwargs.get("munchausen_alpha", 0.9))
        algorithm_kwargs["entropy_temperature"] = float(config.algo_kwargs.get("entropy_temperature", 0.03))
        algorithm_kwargs["munchausen_clip_min"] = float(config.algo_kwargs.get("munchausen_clip_min", -1.0))
    elif algorithm_cls in {CQLDQNAlgorithm, CQLDoubleDQNAlgorithm}:
        algorithm_kwargs["cql_alpha"] = float(config.algo_kwargs.get("cql_alpha", 1.0))
    elif algorithm_cls is HystereticDQNAlgorithm:
        algorithm_kwargs["hysteretic_beta"] = float(config.algo_kwargs.get("hysteretic_beta", 0.1))
    elif algorithm_cls is SPRAlgorithm:
        algorithm_kwargs["spr_loss_coef"] = float(config.algo_kwargs.get("spr_loss_coef", 1.0))
    elif algorithm_cls is JOWAAlgorithm:
        algorithm_kwargs["jowa_world_model_loss_coef"] = float(config.algo_kwargs.get("jowa_world_model_loss_coef", 1.0))
        algorithm_kwargs["jowa_reward_loss_coef"] = float(config.algo_kwargs.get("jowa_reward_loss_coef", 1.0))
        algorithm_kwargs["jowa_reconstruction_loss_coef"] = float(config.algo_kwargs.get("jowa_reconstruction_loss_coef", 1.0))
        algorithm_kwargs["jowa_consistency_loss_coef"] = float(config.algo_kwargs.get("jowa_consistency_loss_coef", 0.5))
    return algorithm_kwargs


def _load_dqn_algorithm(config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device) -> DQNAlgorithm:
    obs_shape, action_dim = _infer_discrete_env_spaces(config)
    hidden_sizes = tuple(config.algo_kwargs.get("hidden_sizes", (64, 64)))
    if config.algo in {"spr", "jowa"} and len(obs_shape) != 3:
        raise ValueError(f"{config.algo} checkpoint loading requires channel-first image observations")

    if len(obs_shape) == 3:
        q_network, algorithm_cls = _build_image_dqn_loader(
            config,
            obs_shape=obs_shape,
            action_dim=action_dim,
            device=device,
        )
    else:
        q_network, algorithm_cls = _build_vector_dqn_loader(
            config,
            obs_dim=obs_shape[0],
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
            device=device,
        )

    algorithm_kwargs = _build_dqn_algorithm_kwargs(
        config,
        q_network=q_network,
        algorithm_cls=algorithm_cls,
    )
    algorithm = algorithm_cls(**algorithm_kwargs)
    algorithm.load_state_dict(checkpoint_state.algorithm_state)
    return algorithm


def _load_c51_dqn_algorithm(config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device) -> C51DQNAlgorithm:
    obs_shape, action_dim = _infer_discrete_env_spaces(config)
    v_min = float(config.algo_kwargs.get("v_min", 0.0))
    v_max = float(config.algo_kwargs.get("v_max", 200.0))
    num_atoms = int(config.algo_kwargs.get("num_atoms", 51))

    if len(obs_shape) == 3:
        head_hidden_sizes = tuple(
            config.algo_kwargs.get("head_hidden_sizes", config.algo_kwargs.get("hidden_sizes", (512,)))
        )
        q_network = CNNC51QNetwork(
            obs_shape=obs_shape,
            action_dim=action_dim,
            hidden_sizes=head_hidden_sizes,
            features_dim=int(config.algo_kwargs.get("features_dim", 512)),
            v_min=v_min,
            v_max=v_max,
            num_atoms=num_atoms,
        ).to(device)
    else:
        if len(obs_shape) != 1:
            raise ValueError("c51_dqn checkpoint loading expects flat or 3D image observations")
        obs_dim = obs_shape[0]
        hidden_sizes = tuple(config.algo_kwargs.get("hidden_sizes", (64, 64)))
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
    num_quantiles = int(config.algo_kwargs.get("num_quantiles", 51))
    kappa = float(config.algo_kwargs.get("kappa", 1.0))

    if len(obs_shape) == 3:
        head_hidden_sizes = tuple(
            config.algo_kwargs.get("head_hidden_sizes", config.algo_kwargs.get("hidden_sizes", (512,)))
        )
        q_network = CNNQRQNetwork(
            obs_shape=obs_shape,
            action_dim=action_dim,
            num_quantiles=num_quantiles,
            hidden_sizes=head_hidden_sizes,
            features_dim=int(config.algo_kwargs.get("features_dim", 512)),
        ).to(device)
    else:
        if len(obs_shape) != 1:
            raise ValueError("qr_dqn checkpoint loading expects flat or 3D image observations")
        obs_dim = obs_shape[0]
        hidden_sizes = tuple(config.algo_kwargs.get("hidden_sizes", (64, 64)))
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
    num_quantiles = int(config.algo_kwargs.get("num_quantiles", 32))
    embedding_dim = int(config.algo_kwargs.get("embedding_dim", 64))
    kappa = float(config.algo_kwargs.get("kappa", 1.0))

    if len(obs_shape) == 3:
        head_hidden_sizes = tuple(
            config.algo_kwargs.get("head_hidden_sizes", config.algo_kwargs.get("hidden_sizes", (512,)))
        )
        q_network = CNNIQNetwork(
            obs_shape=obs_shape,
            action_dim=action_dim,
            num_quantiles=num_quantiles,
            hidden_sizes=head_hidden_sizes,
            embedding_dim=embedding_dim,
            features_dim=int(config.algo_kwargs.get("features_dim", 512)),
        ).to(device)
    else:
        if len(obs_shape) != 1:
            raise ValueError("iqn checkpoint loading expects flat or 3D image observations")
        obs_dim = obs_shape[0]
        hidden_sizes = tuple(config.algo_kwargs.get("hidden_sizes", (64, 64)))
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


def _load_fqf_algorithm(config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device) -> FQFAlgorithm:
    obs_shape, action_dim = _infer_discrete_env_spaces(config)
    num_quantiles = int(config.algo_kwargs.get("num_quantiles", 32))
    embedding_dim = int(config.algo_kwargs.get("embedding_dim", 64))
    kappa = float(config.algo_kwargs.get("kappa", 1.0))
    entropy_coef = float(config.algo_kwargs.get("entropy_coef", 1e-3))

    if len(obs_shape) == 3:
        head_hidden_sizes = tuple(
            config.algo_kwargs.get("head_hidden_sizes", config.algo_kwargs.get("hidden_sizes", (512,)))
        )
        q_network = CNNFQFNetwork(
            obs_shape=obs_shape,
            action_dim=action_dim,
            num_quantiles=num_quantiles,
            hidden_sizes=head_hidden_sizes,
            embedding_dim=embedding_dim,
            features_dim=int(config.algo_kwargs.get("features_dim", 512)),
        ).to(device)
    else:
        if len(obs_shape) != 1:
            raise ValueError("fqf checkpoint loading expects flat or 3D image observations")
        obs_dim = obs_shape[0]
        hidden_sizes = tuple(config.algo_kwargs.get("hidden_sizes", (64, 64)))
        q_network = MLPFQFNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            num_quantiles=num_quantiles,
            hidden_sizes=hidden_sizes,
            embedding_dim=embedding_dim,
        ).to(device)
    algorithm = FQFAlgorithm(
        q_network=q_network,
        learning_rate=float(config.algo_kwargs.get("learning_rate", 1e-3)),
        fraction_learning_rate=float(config.algo_kwargs.get("fraction_learning_rate", config.algo_kwargs.get("learning_rate", 1e-3))),
        gamma=float(config.algo_kwargs.get("gamma", 0.99)),
        target_update_interval=int(config.algo_kwargs.get("target_update_interval", 250)),
        num_quantiles=num_quantiles,
        kappa=kappa,
        entropy_coef=entropy_coef,
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


def _load_naf_algorithm(config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device) -> NAFAlgorithm:
    obs_dim, action_dim = _infer_continuous_env_spaces(config)
    hidden_sizes = tuple(config.algo_kwargs.get("hidden_sizes", (256, 256)))
    algorithm = NAFAlgorithm(
        model=MLPNAFModel(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=hidden_sizes).to(device),
        learning_rate=float(config.algo_kwargs.get("learning_rate", 3e-4)),
        gamma=float(config.algo_kwargs.get("gamma", 0.99)),
        tau=float(config.algo_kwargs.get("tau", 0.005)),
    )
    algorithm.load_state_dict(checkpoint_state.algorithm_state)
    return algorithm


def _load_d4pg_algorithm(config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device) -> D4PGAlgorithm:
    obs_dim, action_dim = _infer_continuous_env_spaces(config)
    hidden_sizes = tuple(config.algo_kwargs.get("hidden_sizes", (256, 256)))
    v_min = float(config.algo_kwargs.get("v_min", -100.0))
    v_max = float(config.algo_kwargs.get("v_max", 100.0))
    num_atoms = int(config.algo_kwargs.get("num_atoms", 51))
    algorithm = D4PGAlgorithm(
        model=MLPD4PGModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
            v_min=v_min,
            v_max=v_max,
            num_atoms=num_atoms,
        ).to(device),
        learning_rate=float(config.algo_kwargs.get("learning_rate", 3e-4)),
        gamma=float(config.algo_kwargs.get("gamma", 0.99)),
        tau=float(config.algo_kwargs.get("tau", 0.005)),
        v_min=v_min,
        v_max=v_max,
        num_atoms=num_atoms,
    )
    algorithm.load_state_dict(checkpoint_state.algorithm_state)
    return algorithm


def _load_drqn_algorithm(config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device) -> DRQNAlgorithm:
    obs_shape, action_dim = _infer_discrete_env_spaces(config)
    if len(obs_shape) != 1:
        raise ValueError("drqn checkpoint loading currently supports flat observations only")

    algorithm = DRQNAlgorithm(
        q_network=LSTMQNetwork(
            obs_shape=obs_shape,
            action_dim=action_dim,
            features_dim=int(config.algo_kwargs.get("features_dim", 256)),
            encoder_hidden_sizes=tuple(config.algo_kwargs.get("hidden_sizes", (128,))),
            head_hidden_sizes=tuple(config.algo_kwargs.get("head_hidden_sizes", (128,))),
            hidden_size=int(config.algo_kwargs.get("recurrent_hidden_size", 256)),
            num_layers=int(config.algo_kwargs.get("recurrent_num_layers", 1)),
        ).to(device),
        learning_rate=float(config.algo_kwargs.get("learning_rate", 1e-3)),
        gamma=float(config.algo_kwargs.get("gamma", 0.99)),
        target_update_interval=int(config.algo_kwargs.get("target_update_interval", 250)),
    )
    algorithm.load_state_dict(checkpoint_state.algorithm_state)
    return algorithm


def _load_r2d2_algorithm(config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device) -> R2D2Algorithm:
    obs_shape, action_dim = _infer_discrete_env_spaces(config)

    algorithm = R2D2Algorithm(
        q_network=LSTMQNetwork(
            obs_shape=obs_shape,
            action_dim=action_dim,
            features_dim=int(config.algo_kwargs.get("features_dim", 256)),
            encoder_hidden_sizes=tuple(config.algo_kwargs.get("hidden_sizes", (128,))),
            head_hidden_sizes=tuple(config.algo_kwargs.get("head_hidden_sizes", (128,))),
            hidden_size=int(config.algo_kwargs.get("recurrent_hidden_size", 256)),
            num_layers=int(config.algo_kwargs.get("recurrent_num_layers", 1)),
        ).to(device),
        learning_rate=float(config.algo_kwargs.get("learning_rate", 1e-3)),
        gamma=float(config.algo_kwargs.get("gamma", 0.99)) ** int(config.algo_kwargs.get("n_step", 3)),
        target_update_interval=int(config.algo_kwargs.get("target_update_interval", 250)),
        double_q=True,
        priority_eta=float(config.algo_kwargs.get("priority_eta", 0.9)),
    )
    algorithm.load_state_dict(checkpoint_state.algorithm_state)
    return algorithm


def _load_agent57_algorithm(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    *,
    device: torch.device,
) -> Agent57Algorithm:
    obs_shape, action_dim = _infer_discrete_env_spaces(config)

    algorithm = Agent57Algorithm(
        q_network=LSTMQNetwork(
            obs_shape=obs_shape,
            action_dim=action_dim,
            features_dim=int(config.algo_kwargs.get("features_dim", 256)),
            encoder_hidden_sizes=tuple(config.algo_kwargs.get("hidden_sizes", (128,))),
            head_hidden_sizes=tuple(config.algo_kwargs.get("head_hidden_sizes", (128,))),
            hidden_size=int(config.algo_kwargs.get("recurrent_hidden_size", 256)),
            num_layers=int(config.algo_kwargs.get("recurrent_num_layers", 1)),
        ).to(device),
        rnd_model=RNDModel(
            obs_shape=obs_shape,
            hidden_sizes=tuple(config.algo_kwargs.get("rnd_hidden_sizes", (256,))),
            embedding_dim=int(config.algo_kwargs.get("rnd_embedding_dim", 128)),
        ).to(device),
        learning_rate=float(config.algo_kwargs.get("learning_rate", 1e-3)),
        rnd_learning_rate=float(config.algo_kwargs.get("rnd_learning_rate", 1e-4)),
        gamma=float(config.algo_kwargs.get("gamma", 0.99)) ** int(config.algo_kwargs.get("n_step", 3)),
        target_update_interval=int(config.algo_kwargs.get("target_update_interval", 250)),
        double_q=True,
        priority_eta=float(config.algo_kwargs.get("priority_eta", 0.9)),
        intrinsic_reward_coef=float(config.algo_kwargs.get("intrinsic_reward_coef", 0.1)),
    )
    algorithm.load_state_dict(checkpoint_state.algorithm_state)
    return algorithm


def _load_drq_algorithm(config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device) -> DrQAlgorithm:
    obs_shape, action_dim = _infer_image_continuous_env_spaces(config)
    algorithm = DrQAlgorithm(
        model=CNNDrQModel(
            obs_shape=obs_shape,
            action_dim=action_dim,
            features_dim=int(config.algo_kwargs.get("features_dim", 256)),
            actor_hidden_sizes=tuple(config.algo_kwargs.get("actor_hidden_sizes", (256, 256))),
            critic_hidden_sizes=tuple(config.algo_kwargs.get("critic_hidden_sizes", (256, 256))),
        ).to(device),
        learning_rate=float(config.algo_kwargs.get("learning_rate", 1e-4)),
        gamma=float(config.algo_kwargs.get("gamma", 0.99)),
        alpha=float(config.algo_kwargs.get("alpha", 0.1)),
        tau=float(config.algo_kwargs.get("tau", 0.01)),
        augmentation_pad=int(config.algo_kwargs.get("augmentation_pad", 4)),
    )
    algorithm.load_state_dict(checkpoint_state.algorithm_state)
    return algorithm


def _load_curl_algorithm(config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device) -> CURLAlgorithm:
    obs_shape, action_dim = _infer_image_continuous_env_spaces(config)
    algorithm = CURLAlgorithm(
        model=CNNCURLModel(
            obs_shape=obs_shape,
            action_dim=action_dim,
            features_dim=int(config.algo_kwargs.get("features_dim", 256)),
            actor_hidden_sizes=tuple(config.algo_kwargs.get("actor_hidden_sizes", (256, 256))),
            critic_hidden_sizes=tuple(config.algo_kwargs.get("critic_hidden_sizes", (256, 256))),
            projection_dim=int(config.algo_kwargs.get("projection_dim", 128)),
        ).to(device),
        learning_rate=float(config.algo_kwargs.get("learning_rate", 1e-4)),
        gamma=float(config.algo_kwargs.get("gamma", 0.99)),
        alpha=float(config.algo_kwargs.get("alpha", 0.1)),
        tau=float(config.algo_kwargs.get("tau", 0.01)),
        augmentation_pad=int(config.algo_kwargs.get("augmentation_pad", 4)),
        curl_temperature=float(config.algo_kwargs.get("curl_temperature", 0.1)),
        curl_coef=float(config.algo_kwargs.get("curl_coef", 1.0)),
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


def _evaluate_ars(config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int) -> MetricDict:
    algorithm = _load_ars_algorithm(config, checkpoint_state, device=device)
    return _evaluate_ars_policy(algorithm.model, config, device=device, num_episodes=num_episodes)


def _evaluate_openai_es(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    num_episodes: int,
) -> MetricDict:
    algorithm = _load_openai_es_algorithm(config, checkpoint_state, device=device)
    return _evaluate_openai_es_policy(algorithm.model, config, device=device, num_episodes=num_episodes)


def _evaluate_impala(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    num_episodes: int,
) -> MetricDict:
    algorithm = _load_impala_algorithm(config, checkpoint_state, device=device)
    return _evaluate_impala_policy(algorithm.policy, config, device=device, num_episodes=num_episodes)


def _evaluate_appo(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    num_episodes: int,
) -> MetricDict:
    algorithm = _load_appo_algorithm(config, checkpoint_state, device=device)
    return _evaluate_appo_policy(algorithm.policy, config, device=device, num_episodes=num_episodes)


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


def _evaluate_decision_transformer(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    num_episodes: int,
) -> MetricDict:
    algorithm = _load_decision_transformer_algorithm(config, checkpoint_state, device=device)
    return _evaluate_decision_transformer_policy(
        algorithm.model,
        config,
        device=device,
        num_episodes=num_episodes,
        context_length=int(config.algo_kwargs.get("context_length", 20)),
        target_return=float(config.algo_kwargs.get("target_return", 0.0)),
        max_timestep=int(config.algo_kwargs.get("max_timestep", 1024)),
        gamma=float(config.algo_kwargs.get("gamma", 0.99)),
    )


def _evaluate_mopo(config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int) -> MetricDict:
    algorithm = _load_mopo_algorithm(config, checkpoint_state, device=device)
    return _evaluate_sac_policy(algorithm.policy_model, config, device=device, num_episodes=num_episodes)


def _evaluate_mbpo(config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int) -> MetricDict:
    algorithm = _load_mbpo_algorithm(config, checkpoint_state, device=device)
    return _evaluate_sac_policy(algorithm.model, config, device=device, num_episodes=num_episodes)


def _evaluate_pets(config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int) -> MetricDict:
    algorithm = _load_pets_algorithm(config, checkpoint_state, device=device)
    return _evaluate_pets_policy(algorithm, config, device=device, num_episodes=num_episodes)


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


def _evaluate_gail(config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int) -> MetricDict:
    algorithm = _load_gail_algorithm(config, checkpoint_state, device=device)
    return _evaluate_policy(algorithm.policy, config, device=device, num_episodes=num_episodes)


def _evaluate_dreamer(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    num_episodes: int,
) -> MetricDict:
    algorithm = _load_dreamer_algorithm(config, checkpoint_state, device=device)
    return _evaluate_policy(algorithm, config, device=device, num_episodes=num_episodes)


def _evaluate_dreamerv3(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    num_episodes: int,
) -> MetricDict:
    algorithm = _load_dreamer_algorithm(config, checkpoint_state, device=device)
    return _evaluate_policy(algorithm, config, device=device, num_episodes=num_episodes)


def _evaluate_diamond(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    num_episodes: int,
) -> MetricDict:
    algorithm = _load_dreamer_algorithm(config, checkpoint_state, device=device)
    return _evaluate_policy(algorithm, config, device=device, num_episodes=num_episodes)


def _evaluate_horizon_imagination(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    num_episodes: int,
) -> MetricDict:
    algorithm = _load_dreamer_algorithm(config, checkpoint_state, device=device)
    return _evaluate_policy(algorithm, config, device=device, num_episodes=num_episodes)


def _evaluate_po_dreamer(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    num_episodes: int,
) -> MetricDict:
    algorithm = _load_dreamer_algorithm(config, checkpoint_state, device=device)
    return _evaluate_policy(algorithm, config, device=device, num_episodes=num_episodes)


def _evaluate_twisted(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    num_episodes: int,
) -> MetricDict:
    algorithm = _load_dreamer_algorithm(config, checkpoint_state, device=device)
    return _evaluate_policy(algorithm, config, device=device, num_episodes=num_episodes)


def _evaluate_mow(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    num_episodes: int,
) -> MetricDict:
    algorithm = _load_dreamer_algorithm(config, checkpoint_state, device=device)
    return _evaluate_policy(algorithm, config, device=device, num_episodes=num_episodes)


def _evaluate_eadream(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    num_episodes: int,
) -> MetricDict:
    algorithm = _load_dreamer_algorithm(config, checkpoint_state, device=device)
    return _evaluate_policy(algorithm, config, device=device, num_episodes=num_episodes)


def _evaluate_muzero(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    num_episodes: int,
) -> MetricDict:
    algorithm = _load_muzero_algorithm(config, checkpoint_state, device=device)
    return _evaluate_muzero_policy(algorithm, config, device=device, num_episodes=num_episodes)


def _evaluate_gumbel_muzero(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    num_episodes: int,
) -> MetricDict:
    algorithm = _load_gumbel_muzero_algorithm(config, checkpoint_state, device=device)
    return _evaluate_muzero_policy(algorithm, config, device=device, num_episodes=num_episodes)


def _evaluate_efficientzero(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    num_episodes: int,
) -> MetricDict:
    algorithm = _load_efficientzero_algorithm(config, checkpoint_state, device=device)
    return _evaluate_muzero_policy(algorithm, config, device=device, num_episodes=num_episodes)


def _evaluate_scalezero(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    num_episodes: int,
) -> MetricDict:
    algorithm = _load_muzero_algorithm(config, checkpoint_state, device=device)
    return _evaluate_muzero_policy(algorithm, config, device=device, num_episodes=num_episodes)


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


def _evaluate_fqf(config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int) -> MetricDict:
    algorithm = _load_fqf_algorithm(config, checkpoint_state, device=device)
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


def _evaluate_naf(config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int) -> MetricDict:
    algorithm = _load_naf_algorithm(config, checkpoint_state, device=device)
    return _evaluate_naf_policy(algorithm.model, config, device=device, num_episodes=num_episodes)


def _evaluate_d4pg(config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int) -> MetricDict:
    algorithm = _load_d4pg_algorithm(config, checkpoint_state, device=device)
    return _evaluate_d4pg_policy(algorithm.model, config, device=device, num_episodes=num_episodes)


def _evaluate_drqn(config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int) -> MetricDict:
    algorithm = _load_drqn_algorithm(config, checkpoint_state, device=device)
    algorithm.set_eval_mode()
    return _evaluate_drqn_policy(algorithm.q_network, config, device=device, num_episodes=num_episodes)


def _evaluate_r2d2(config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int) -> MetricDict:
    algorithm = _load_r2d2_algorithm(config, checkpoint_state, device=device)
    algorithm.set_eval_mode()
    return _evaluate_r2d2_policy(algorithm.q_network, config, device=device, num_episodes=num_episodes)


def _evaluate_agent57(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    num_episodes: int,
) -> MetricDict:
    algorithm = _load_agent57_algorithm(config, checkpoint_state, device=device)
    algorithm.set_eval_mode()
    return _evaluate_r2d2_policy(algorithm.q_network, config, device=device, num_episodes=num_episodes)


def _evaluate_drq(config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int) -> MetricDict:
    algorithm = _load_drq_algorithm(config, checkpoint_state, device=device)
    return _evaluate_drq_policy(algorithm.model, config, device=device, num_episodes=num_episodes)


def _evaluate_curl(config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int) -> MetricDict:
    algorithm = _load_curl_algorithm(config, checkpoint_state, device=device)
    return _evaluate_curl_policy(algorithm.model, config, device=device, num_episodes=num_episodes)


def _evaluate_ppg(config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int) -> MetricDict:
    algorithm = _load_ppg_algorithm(config, checkpoint_state, device=device)
    return _evaluate_ppg_policy(algorithm.model, config, device=device, num_episodes=num_episodes)


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


def _predict_ars(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    del deterministic
    algorithm = _load_ars_algorithm(config, checkpoint_state, device=device)
    obs_tensor = _prepare_observation(obs, device=device)
    low, high = _continuous_action_bounds(config, device=device)
    with torch.no_grad():
        normalized_actions = algorithm.model.actor(obs_tensor)
        actions = _scale_continuous_actions(normalized_actions, low=low, high=high)
    return _format_action_output(actions, discrete=False)


def _predict_openai_es(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    del deterministic
    algorithm = _load_openai_es_algorithm(config, checkpoint_state, device=device)
    obs_tensor = _prepare_observation(obs, device=device)
    low, high = _continuous_action_bounds(config, device=device)
    with torch.no_grad():
        normalized_actions = algorithm.model.actor(obs_tensor)
        actions = _scale_continuous_actions(normalized_actions, low=low, high=high)
    return _format_action_output(actions, discrete=False)


def _predict_impala(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    algorithm = _load_impala_algorithm(config, checkpoint_state, device=device)
    obs_tensor = _prepare_observation(obs, device=device)
    with torch.no_grad():
        actions = algorithm.policy.act(obs_tensor, deterministic=deterministic).actions
    return _format_action_output(actions, discrete=True)


def _predict_appo(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    algorithm = _load_appo_algorithm(config, checkpoint_state, device=device)
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


def _predict_decision_transformer(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    del deterministic
    algorithm = _load_decision_transformer_algorithm(config, checkpoint_state, device=device)
    low, high = _continuous_action_bounds(config, device=device)
    obs_array = np.asarray(obs, dtype=np.float32)
    action_dim = int(low.numel())
    autoregressive_batch = _build_autoregressive_window(
        [obs_array],
        [],
        [float(config.algo_kwargs.get("target_return", 0.0))],
        context_length=int(config.algo_kwargs.get("context_length", 20)),
        action_dim=action_dim,
        max_timestep=int(config.algo_kwargs.get("max_timestep", 1024)),
        device=device,
    )
    with torch.no_grad():
        normalized_actions = torch.nan_to_num(
            algorithm.model.predict_last_action(**autoregressive_batch),
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )
        actions = _scale_continuous_actions(normalized_actions, low=low, high=high)
    return _format_action_output(actions, discrete=False)


def _predict_mopo(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    algorithm = _load_mopo_algorithm(config, checkpoint_state, device=device)
    obs_tensor = _prepare_observation(obs, device=device)
    low, high = _continuous_action_bounds(config, device=device)
    with torch.no_grad():
        normalized_actions = algorithm.policy_model.sample_actions(obs_tensor, deterministic=deterministic).actions
        actions = _scale_continuous_actions(normalized_actions, low=low, high=high)
    return _format_action_output(actions, discrete=False)


def _predict_pets(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    algorithm = _load_pets_algorithm(config, checkpoint_state, device=device)
    low, high = _continuous_action_bounds(config, device=device)
    return algorithm.plan_action(
        obs,
        action_low=low.detach().cpu().numpy(),
        action_high=high.detach().cpu().numpy(),
        horizon=int(config.algo_kwargs.get("planning_horizon", 5)),
        num_candidates=int(config.algo_kwargs.get("planning_candidates", 256)),
        num_iterations=int(config.algo_kwargs.get("planning_iterations", 4)),
        num_topk=int(config.algo_kwargs.get("planning_topk", 32)),
        num_particles=int(config.algo_kwargs.get("planning_particles", 8)),
        deterministic=deterministic,
    )


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


def _predict_gail(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    algorithm = _load_gail_algorithm(config, checkpoint_state, device=device)
    obs_tensor = _prepare_observation(obs, device=device)
    with torch.no_grad():
        actions = algorithm.policy.act(obs_tensor, deterministic=deterministic).actions
    return _format_action_output(actions, discrete=True)


def _predict_dreamer(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    algorithm = _load_dreamer_algorithm(config, checkpoint_state, device=device)
    obs_tensor = _prepare_observation(obs, device=device)
    with torch.no_grad():
        actions = algorithm.act(obs_tensor, deterministic=deterministic).actions
    return _format_action_output(actions, discrete=True)


def _predict_dreamerv3(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    algorithm = _load_dreamer_algorithm(config, checkpoint_state, device=device)
    obs_tensor = _prepare_observation(obs, device=device)
    with torch.no_grad():
        actions = algorithm.act(obs_tensor, deterministic=deterministic).actions
    return _format_action_output(actions, discrete=True)


def _predict_diamond(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    algorithm = _load_dreamer_algorithm(config, checkpoint_state, device=device)
    obs_tensor = _prepare_observation(obs, device=device)
    with torch.no_grad():
        actions = algorithm.act(obs_tensor, deterministic=deterministic).actions
    return _format_action_output(actions, discrete=True)


def _predict_horizon_imagination(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    algorithm = _load_dreamer_algorithm(config, checkpoint_state, device=device)
    obs_tensor = _prepare_observation(obs, device=device)
    with torch.no_grad():
        actions = algorithm.act(obs_tensor, deterministic=deterministic).actions
    return _format_action_output(actions, discrete=True)


def _predict_po_dreamer(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    algorithm = _load_dreamer_algorithm(config, checkpoint_state, device=device)
    obs_tensor = _prepare_observation(obs, device=device)
    with torch.no_grad():
        actions = algorithm.act(obs_tensor, deterministic=deterministic).actions
    return _format_action_output(actions, discrete=True)


def _predict_twisted(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    algorithm = _load_dreamer_algorithm(config, checkpoint_state, device=device)
    obs_tensor = _prepare_observation(obs, device=device)
    with torch.no_grad():
        actions = algorithm.act(obs_tensor, deterministic=deterministic).actions
    return _format_action_output(actions, discrete=True)


def _predict_mow(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    algorithm = _load_dreamer_algorithm(config, checkpoint_state, device=device)
    obs_tensor = _prepare_observation(obs, device=device)
    with torch.no_grad():
        actions = algorithm.act(obs_tensor, deterministic=deterministic).actions
    return _format_action_output(actions, discrete=True)


def _predict_eadream(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    algorithm = _load_dreamer_algorithm(config, checkpoint_state, device=device)
    obs_tensor = _prepare_observation(obs, device=device)
    with torch.no_grad():
        actions = algorithm.act(obs_tensor, deterministic=deterministic).actions
    return _format_action_output(actions, discrete=True)


def _predict_muzero(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    algorithm = _load_muzero_algorithm(config, checkpoint_state, device=device)
    with torch.no_grad():
        action_tensor = algorithm.act(obs, deterministic=deterministic).actions
    return int(action_tensor.squeeze(0).detach().cpu().item())


def _predict_gumbel_muzero(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    algorithm = _load_gumbel_muzero_algorithm(config, checkpoint_state, device=device)
    with torch.no_grad():
        action_tensor = algorithm.act(obs, deterministic=deterministic).actions
    return int(action_tensor.squeeze(0).detach().cpu().item())


def _predict_efficientzero(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    algorithm = _load_efficientzero_algorithm(config, checkpoint_state, device=device)
    with torch.no_grad():
        action_tensor = algorithm.act(obs, deterministic=deterministic).actions
    return int(action_tensor.squeeze(0).detach().cpu().item())


def _predict_scalezero(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    algorithm = _load_muzero_algorithm(config, checkpoint_state, device=device)
    with torch.no_grad():
        action_tensor = algorithm.act(obs, deterministic=deterministic).actions
    return int(action_tensor.squeeze(0).detach().cpu().item())


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


def _predict_fqf(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    del deterministic
    algorithm = _load_fqf_algorithm(config, checkpoint_state, device=device)
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


def _predict_mbpo(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    algorithm = _load_mbpo_algorithm(config, checkpoint_state, device=device)
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


def _predict_naf(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    del deterministic
    algorithm = _load_naf_algorithm(config, checkpoint_state, device=device)
    obs_tensor = _prepare_observation(obs, device=device)
    low, high = _continuous_action_bounds(config, device=device)
    with torch.no_grad():
        normalized_actions = algorithm.model.actor(obs_tensor)
        actions = _scale_continuous_actions(normalized_actions, low=low, high=high)
    return _format_action_output(actions, discrete=False)


def _predict_d4pg(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    del deterministic
    algorithm = _load_d4pg_algorithm(config, checkpoint_state, device=device)
    obs_tensor = _prepare_observation(obs, device=device)
    low, high = _continuous_action_bounds(config, device=device)
    with torch.no_grad():
        normalized_actions = algorithm.model.actor(obs_tensor)
        actions = _scale_continuous_actions(normalized_actions, low=low, high=high)
    return _format_action_output(actions, discrete=False)


def _predict_ppg(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    algorithm = _load_ppg_algorithm(config, checkpoint_state, device=device)
    obs_tensor = _prepare_observation(obs, device=device)
    with torch.no_grad():
        actions = algorithm.model.act(obs_tensor, deterministic=deterministic).actions
    return _format_action_output(actions, discrete=True)


def _predict_drq(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    algorithm = _load_drq_algorithm(config, checkpoint_state, device=device)
    obs_tensor = _prepare_observation(obs, device=device)
    low, high = _continuous_action_bounds(config, device=device)
    with torch.no_grad():
        normalized_actions = algorithm.model.sample_actions(obs_tensor, deterministic=deterministic).actions
        actions = _scale_continuous_actions(normalized_actions, low=low, high=high)
    return _format_action_output(actions, discrete=False)


def _predict_curl(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    algorithm = _load_curl_algorithm(config, checkpoint_state, device=device)
    obs_tensor = _prepare_observation(obs, device=device)
    low, high = _continuous_action_bounds(config, device=device)
    with torch.no_grad():
        normalized_actions = algorithm.model.sample_actions(obs_tensor, deterministic=deterministic).actions
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


def _predict_drqn(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    algorithm = _load_drqn_algorithm(config, checkpoint_state, device=device)
    algorithm.set_eval_mode()
    obs_tensor = _prepare_observation(obs, device=device)
    initial_state = algorithm.q_network.initial_state(int(obs_tensor.shape[0]), device=device)
    episode_starts = torch.ones(int(obs_tensor.shape[0]), dtype=torch.bool, device=device)
    with torch.no_grad():
        actions = algorithm.q_network.act(
            obs_tensor,
            state=initial_state,
            epsilon=0.0,
            deterministic=deterministic,
            episode_starts=episode_starts,
        ).actions
    return _format_action_output(actions, discrete=True)


def _predict_r2d2(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    algorithm = _load_r2d2_algorithm(config, checkpoint_state, device=device)
    algorithm.set_eval_mode()
    obs_tensor = _prepare_observation(obs, device=device)
    initial_state = algorithm.q_network.initial_state(int(obs_tensor.shape[0]), device=device)
    episode_starts = torch.ones(int(obs_tensor.shape[0]), dtype=torch.bool, device=device)
    with torch.no_grad():
        actions = algorithm.q_network.act(
            obs_tensor,
            state=initial_state,
            epsilon=0.0,
            deterministic=deterministic,
            episode_starts=episode_starts,
        ).actions
    return _format_action_output(actions, discrete=True)


def _predict_agent57(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    algorithm = _load_agent57_algorithm(config, checkpoint_state, device=device)
    algorithm.set_eval_mode()
    obs_tensor = _prepare_observation(obs, device=device)
    initial_state = algorithm.q_network.initial_state(int(obs_tensor.shape[0]), device=device)
    episode_starts = torch.ones(int(obs_tensor.shape[0]), dtype=torch.bool, device=device)
    with torch.no_grad():
        actions = algorithm.q_network.act(
            obs_tensor,
            state=initial_state,
            epsilon=0.0,
            deterministic=deterministic,
            episode_starts=episode_starts,
        ).actions
    return _format_action_output(actions, discrete=True)


_ALGORITHM_REGISTRY: dict[str, AlgorithmSpec] = {
    "a2c": AlgorithmSpec(
        name="a2c",
        train_fn=train_a2c,
        evaluate_fn=_evaluate_a2c,
        predict_fn=_predict_a2c,
    ),
    "ars": AlgorithmSpec(
        name="ars",
        train_fn=train_ars,
        evaluate_fn=_evaluate_ars,
        predict_fn=_predict_ars,
    ),
    "openai_es": AlgorithmSpec(
        name="openai_es",
        train_fn=train_openai_es,
        evaluate_fn=_evaluate_openai_es,
        predict_fn=_predict_openai_es,
    ),
    "impala": AlgorithmSpec(
        name="impala",
        train_fn=train_impala,
        evaluate_fn=_evaluate_impala,
        predict_fn=_predict_impala,
    ),
    "appo": AlgorithmSpec(
        name="appo",
        train_fn=train_appo,
        evaluate_fn=_evaluate_appo,
        predict_fn=_predict_appo,
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
    "decision_transformer": AlgorithmSpec(
        name="decision_transformer",
        train_fn=train_decision_transformer,
        evaluate_fn=_evaluate_decision_transformer,
        predict_fn=_predict_decision_transformer,
    ),
    "mopo": AlgorithmSpec(
        name="mopo",
        train_fn=train_mopo,
        evaluate_fn=_evaluate_mopo,
        predict_fn=_predict_mopo,
    ),
    "mbpo": AlgorithmSpec(
        name="mbpo",
        train_fn=train_mbpo,
        evaluate_fn=_evaluate_mbpo,
        predict_fn=_predict_mbpo,
    ),
    "pets": AlgorithmSpec(
        name="pets",
        train_fn=train_pets,
        evaluate_fn=_evaluate_pets,
        predict_fn=_predict_pets,
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
    "gail": AlgorithmSpec(
        name="gail",
        train_fn=train_gail,
        evaluate_fn=_evaluate_gail,
        predict_fn=_predict_gail,
    ),
    "dreamer": AlgorithmSpec(
        name="dreamer",
        train_fn=train_dreamer,
        evaluate_fn=_evaluate_dreamer,
        predict_fn=_predict_dreamer,
    ),
    "dreamerv3": AlgorithmSpec(
        name="dreamerv3",
        train_fn=train_dreamer,
        evaluate_fn=_evaluate_dreamerv3,
        predict_fn=_predict_dreamerv3,
    ),
    "diamond": AlgorithmSpec(
        name="diamond",
        train_fn=train_dreamer,
        evaluate_fn=_evaluate_diamond,
        predict_fn=_predict_diamond,
    ),
    "horizon_imagination": AlgorithmSpec(
        name="horizon_imagination",
        train_fn=train_dreamer,
        evaluate_fn=_evaluate_horizon_imagination,
        predict_fn=_predict_horizon_imagination,
    ),
    "po_dreamer": AlgorithmSpec(
        name="po_dreamer",
        train_fn=train_dreamer,
        evaluate_fn=_evaluate_po_dreamer,
        predict_fn=_predict_po_dreamer,
    ),
    "twisted": AlgorithmSpec(
        name="twisted",
        train_fn=train_dreamer,
        evaluate_fn=_evaluate_twisted,
        predict_fn=_predict_twisted,
    ),
    "mow": AlgorithmSpec(
        name="mow",
        train_fn=train_dreamer,
        evaluate_fn=_evaluate_mow,
        predict_fn=_predict_mow,
    ),
    "eadream": AlgorithmSpec(
        name="eadream",
        train_fn=train_dreamer,
        evaluate_fn=_evaluate_eadream,
        predict_fn=_predict_eadream,
    ),
    "muzero": AlgorithmSpec(
        name="muzero",
        train_fn=train_muzero,
        evaluate_fn=_evaluate_muzero,
        predict_fn=_predict_muzero,
    ),
    "gumbel_muzero": AlgorithmSpec(
        name="gumbel_muzero",
        train_fn=train_muzero,
        evaluate_fn=_evaluate_gumbel_muzero,
        predict_fn=_predict_gumbel_muzero,
    ),
    "efficientzero": AlgorithmSpec(
        name="efficientzero",
        train_fn=train_efficientzero,
        evaluate_fn=_evaluate_efficientzero,
        predict_fn=_predict_efficientzero,
    ),
    "scalezero": AlgorithmSpec(
        name="scalezero",
        train_fn=train_muzero,
        evaluate_fn=_evaluate_scalezero,
        predict_fn=_predict_scalezero,
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
    "jowa": AlgorithmSpec(
        name="jowa",
        train_fn=train_dqn,
        evaluate_fn=_evaluate_dqn,
        predict_fn=_predict_dqn,
    ),
    "spr": AlgorithmSpec(
        name="spr",
        train_fn=train_dqn,
        evaluate_fn=_evaluate_dqn,
        predict_fn=_predict_dqn,
    ),
    "apex_dqn": AlgorithmSpec(
        name="apex_dqn",
        train_fn=train_apex_dqn,
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
    "expected_sarsa": AlgorithmSpec(
        name="expected_sarsa",
        train_fn=train_dqn,
        evaluate_fn=_evaluate_dqn,
        predict_fn=_predict_dqn,
    ),
    "expected_double_dqn": AlgorithmSpec(
        name="expected_double_dqn",
        train_fn=train_dqn,
        evaluate_fn=_evaluate_dqn,
        predict_fn=_predict_dqn,
    ),
    "boltzmann_dqn": AlgorithmSpec(
        name="boltzmann_dqn",
        train_fn=train_dqn,
        evaluate_fn=_evaluate_dqn,
        predict_fn=_predict_dqn,
    ),
    "boltzmann_double_dqn": AlgorithmSpec(
        name="boltzmann_double_dqn",
        train_fn=train_dqn,
        evaluate_fn=_evaluate_dqn,
        predict_fn=_predict_dqn,
    ),
    "mellowmax_dqn": AlgorithmSpec(
        name="mellowmax_dqn",
        train_fn=train_dqn,
        evaluate_fn=_evaluate_dqn,
        predict_fn=_predict_dqn,
    ),
    "soft_dqn": AlgorithmSpec(
        name="soft_dqn",
        train_fn=train_dqn,
        evaluate_fn=_evaluate_dqn,
        predict_fn=_predict_dqn,
    ),
    "soft_double_dqn": AlgorithmSpec(
        name="soft_double_dqn",
        train_fn=train_dqn,
        evaluate_fn=_evaluate_dqn,
        predict_fn=_predict_dqn,
    ),
    "advantage_learning_dqn": AlgorithmSpec(
        name="advantage_learning_dqn",
        train_fn=train_dqn,
        evaluate_fn=_evaluate_dqn,
        predict_fn=_predict_dqn,
    ),
    "persistent_advantage_learning_dqn": AlgorithmSpec(
        name="persistent_advantage_learning_dqn",
        train_fn=train_dqn,
        evaluate_fn=_evaluate_dqn,
        predict_fn=_predict_dqn,
    ),
    "munchausen_dqn": AlgorithmSpec(
        name="munchausen_dqn",
        train_fn=train_dqn,
        evaluate_fn=_evaluate_dqn,
        predict_fn=_predict_dqn,
    ),
    "munchausen_double_dqn": AlgorithmSpec(
        name="munchausen_double_dqn",
        train_fn=train_dqn,
        evaluate_fn=_evaluate_dqn,
        predict_fn=_predict_dqn,
    ),
    "cql_dqn": AlgorithmSpec(
        name="cql_dqn",
        train_fn=train_dqn,
        evaluate_fn=_evaluate_dqn,
        predict_fn=_predict_dqn,
    ),
    "cql_double_dqn": AlgorithmSpec(
        name="cql_double_dqn",
        train_fn=train_dqn,
        evaluate_fn=_evaluate_dqn,
        predict_fn=_predict_dqn,
    ),
    "clipped_double_dqn": AlgorithmSpec(
        name="clipped_double_dqn",
        train_fn=train_dqn,
        evaluate_fn=_evaluate_dqn,
        predict_fn=_predict_dqn,
    ),
    "hysteretic_dqn": AlgorithmSpec(
        name="hysteretic_dqn",
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
    "fqf": AlgorithmSpec(
        name="fqf",
        train_fn=train_dqn,
        evaluate_fn=_evaluate_fqf,
        predict_fn=_predict_fqf,
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
    "naf": AlgorithmSpec(
        name="naf",
        train_fn=train_naf,
        evaluate_fn=_evaluate_naf,
        predict_fn=_predict_naf,
    ),
    "d4pg": AlgorithmSpec(
        name="d4pg",
        train_fn=train_d4pg,
        evaluate_fn=_evaluate_d4pg,
        predict_fn=_predict_d4pg,
    ),
    "drqn": AlgorithmSpec(
        name="drqn",
        train_fn=train_drqn,
        evaluate_fn=_evaluate_drqn,
        predict_fn=_predict_drqn,
    ),
    "r2d2": AlgorithmSpec(
        name="r2d2",
        train_fn=train_r2d2,
        evaluate_fn=_evaluate_r2d2,
        predict_fn=_predict_r2d2,
    ),
    "agent57": AlgorithmSpec(
        name="agent57",
        train_fn=train_agent57,
        evaluate_fn=_evaluate_agent57,
        predict_fn=_predict_agent57,
    ),
    "drq": AlgorithmSpec(
        name="drq",
        train_fn=train_drq,
        evaluate_fn=_evaluate_drq,
        predict_fn=_predict_drq,
    ),
    "curl": AlgorithmSpec(
        name="curl",
        train_fn=train_curl,
        evaluate_fn=_evaluate_curl,
        predict_fn=_predict_curl,
    ),
    "ppg": AlgorithmSpec(
        name="ppg",
        train_fn=train_ppg,
        evaluate_fn=_evaluate_ppg,
        predict_fn=_predict_ppg,
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
