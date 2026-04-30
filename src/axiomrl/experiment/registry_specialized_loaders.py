from __future__ import annotations

import torch

from rl_training.algorithms.diamond import Diamond as DiamondAlgorithm
from rl_training.algorithms.discrete_sac import DiscreteSAC as DiscreteSACAlgorithm
from rl_training.algorithms.dreamer import Dreamer as DreamerAlgorithm
from rl_training.algorithms.dreamerv3 import DreamerV3 as DreamerV3Algorithm
from rl_training.algorithms.eadream import EADream as EADreamAlgorithm
from rl_training.algorithms.efficientzero import EfficientZero as EfficientZeroAlgorithm
from rl_training.algorithms.gumbel_muzero import GumbelMuZero as GumbelMuZeroAlgorithm
from rl_training.algorithms.her import HER as HERAlgorithm
from rl_training.algorithms.horizon_imagination import (
    HorizonImagination as HorizonImaginationAlgorithm,
)
from rl_training.algorithms.mbpo import MBPO as MBPOAlgorithm
from rl_training.algorithms.mopo import MOPO as MOPOAlgorithm
from rl_training.algorithms.mow import MoW as MoWAlgorithm
from rl_training.algorithms.muzero import MuZero as MuZeroAlgorithm
from rl_training.algorithms.muzero import MuZeroMCTSConfig
from rl_training.algorithms.pets import PETS as PETSAlgorithm
from rl_training.algorithms.po_dreamer import PODreamer as PODreamerAlgorithm
from rl_training.algorithms.scalezero import ScaleZero as ScaleZeroAlgorithm
from rl_training.algorithms.twisted import Twisted as TwistedAlgorithm
from rl_training.experiment.checkpointing import CheckpointState
from rl_training.experiment.config import TrainConfig
from rl_training.experiment.registry_support import (
    _infer_continuous_env_spaces,
    _infer_discrete_env_spaces,
)
from rl_training.models.dreamer import DreamerModel
from rl_training.models.eadream import EADreamModel
from rl_training.models.mlp_ddpg import MLPDDPGModel
from rl_training.models.mlp_discrete_sac import MLPDiscreteSACModel
from rl_training.models.mlp_mopo import MLPMOPOEnsembleModel
from rl_training.models.mlp_sac import MLPSACModel
from rl_training.models.mow import MoWModel
from rl_training.models.muzero import MuZeroModel
from rl_training.models.po_dreamer import PODreamerModel
from rl_training.models.scalezero import ScaleZeroModel
from rl_training.runtime.her_trainer import _infer_her_spaces


def _load_mopo_algorithm(
    config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device
) -> MOPOAlgorithm:
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


def _load_mbpo_algorithm(
    config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device
) -> MBPOAlgorithm:
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


def _load_pets_algorithm(
    config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device
) -> PETSAlgorithm:
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


def _load_her_algorithm(
    config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device
) -> HERAlgorithm:
    goal_spec, action_dim = _infer_her_spaces(config)
    hidden_sizes = tuple(config.algo_kwargs.get("hidden_sizes", (256, 256)))
    algorithm = HERAlgorithm(
        model=MLPDDPGModel(obs_dim=goal_spec.flat_observation_dim, action_dim=action_dim, hidden_sizes=hidden_sizes).to(
            device
        ),
        learning_rate=float(config.algo_kwargs.get("learning_rate", 3e-4)),
        gamma=float(config.algo_kwargs.get("gamma", 0.99)),
        tau=float(config.algo_kwargs.get("tau", 0.005)),
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
        algorithm_kwargs["transport_temperature"] = float(config.algo_kwargs.get("transport_temperature", 0.5))
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
