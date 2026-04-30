from __future__ import annotations

import torch

from axiomrl.algorithms.awac import AWAC as AWACAlgorithm
from axiomrl.algorithms.bcq import BCQ as BCQAlgorithm
from axiomrl.algorithms.bear import BEAR as BEARAlgorithm
from axiomrl.algorithms.cal_ql import CalQL as CalQLAlgorithm
from axiomrl.algorithms.crr import CRR as CRRAlgorithm
from axiomrl.algorithms.rebrac import ReBRAC as ReBRACAlgorithm
from axiomrl.algorithms.xql import XQL as XQLAlgorithm
from axiomrl.experiment.checkpointing import CheckpointState
from axiomrl.experiment.config import TrainConfig
from axiomrl.experiment.registry_support import _infer_continuous_env_spaces
from axiomrl.models.mlp_bcq import MLPBCQModel
from axiomrl.models.mlp_bear import MLPBEARModel
from axiomrl.models.mlp_iql import MLPIQLModel
from axiomrl.models.mlp_sac import MLPSACModel
from axiomrl.models.mlp_td3 import MLPTD3Model


def _load_bcq_algorithm(
    config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device
) -> BCQAlgorithm:
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


def _load_bear_algorithm(
    config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device
) -> BEARAlgorithm:
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


def _load_awac_algorithm(
    config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device
) -> AWACAlgorithm:
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


def _load_crr_algorithm(
    config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device
) -> CRRAlgorithm:
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


def _load_xql_algorithm(
    config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device
) -> XQLAlgorithm:
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
