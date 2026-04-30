from __future__ import annotations

import torch
from torch import nn

from axiomrl.algorithms.c51_dqn import C51DQN as C51DQNAlgorithm
from axiomrl.algorithms.dqn import CQLDQN as CQLDQNAlgorithm
from axiomrl.algorithms.dqn import DQN as DQNAlgorithm
from axiomrl.algorithms.dqn import AdvantageLearningDQN as AdvantageLearningDQNAlgorithm
from axiomrl.algorithms.dqn import BoltzmannDoubleDQN as BoltzmannDoubleDQNAlgorithm
from axiomrl.algorithms.dqn import BoltzmannDQN as BoltzmannDQNAlgorithm
from axiomrl.algorithms.dqn import ClippedDoubleDQN as ClippedDoubleDQNAlgorithm
from axiomrl.algorithms.dqn import CQLDoubleDQN as CQLDoubleDQNAlgorithm
from axiomrl.algorithms.dqn import DoubleDQN as DoubleDQNAlgorithm
from axiomrl.algorithms.dqn import DuelingDQN as DuelingDQNAlgorithm
from axiomrl.algorithms.dqn import ExpectedDoubleDQN as ExpectedDoubleDQNAlgorithm
from axiomrl.algorithms.dqn import ExpectedSARSA as ExpectedSARSAAlgorithm
from axiomrl.algorithms.dqn import HystereticDQN as HystereticDQNAlgorithm
from axiomrl.algorithms.dqn import MellowmaxDQN as MellowmaxDQNAlgorithm
from axiomrl.algorithms.dqn import MunchausenDoubleDQN as MunchausenDoubleDQNAlgorithm
from axiomrl.algorithms.dqn import MunchausenDQN as MunchausenDQNAlgorithm
from axiomrl.algorithms.dqn import NoisyDQN as NoisyDQNAlgorithm
from axiomrl.algorithms.dqn import PersistentAdvantageLearningDQN as PersistentAdvantageLearningDQNAlgorithm
from axiomrl.algorithms.dqn import PrioritizedDQN as PrioritizedDQNAlgorithm
from axiomrl.algorithms.dqn import RainbowDQN as RainbowDQNAlgorithm
from axiomrl.algorithms.dqn import SoftDoubleDQN as SoftDoubleDQNAlgorithm
from axiomrl.algorithms.dqn import SoftDQN as SoftDQNAlgorithm
from axiomrl.algorithms.fqf import FQF as FQFAlgorithm
from axiomrl.algorithms.iqn import IQN as IQNAlgorithm
from axiomrl.algorithms.jowa import JOWA as JOWAAlgorithm
from axiomrl.algorithms.qr_dqn import QRDQN as QRDQNAlgorithm
from axiomrl.algorithms.spr import SPR as SPRAlgorithm
from axiomrl.experiment.checkpointing import CheckpointState
from axiomrl.experiment.config import TrainConfig
from axiomrl.experiment.registry_support import _infer_discrete_env_spaces
from axiomrl.models.cnn import (
    CNNC51QNetwork,
    CNNDuelingNoisyQNetwork,
    CNNDuelingQNetwork,
    CNNFQFNetwork,
    CNNIQNetwork,
    CNNJOWAQNetwork,
    CNNNoisyQNetwork,
    CNNQNetwork,
    CNNQRQNetwork,
    CNNSPRQNetwork,
)
from axiomrl.models.mlp_c51_q_network import MLPC51QNetwork
from axiomrl.models.mlp_dueling_noisy_q_network import MLPDuelingNoisyQNetwork
from axiomrl.models.mlp_dueling_q_network import MLPDuelingQNetwork
from axiomrl.models.mlp_fqf_network import MLPFQFNetwork
from axiomrl.models.mlp_iqn_network import MLPIQNetwork
from axiomrl.models.mlp_noisy_q_network import MLPNoisyQNetwork
from axiomrl.models.mlp_q_network import MLPQNetwork
from axiomrl.models.mlp_qr_q_network import MLPQRQNetwork


def _build_image_dqn_loader(
    config: TrainConfig,
    *,
    obs_shape: tuple[int, ...],
    action_dim: int,
    device: torch.device,
) -> tuple[
    CNNQNetwork | CNNSPRQNetwork | CNNJOWAQNetwork, type[DQNAlgorithm] | type[SPRAlgorithm] | type[JOWAAlgorithm]
]:
    head_hidden_sizes = tuple(
        config.algo_kwargs.get("head_hidden_sizes", config.algo_kwargs.get("hidden_sizes", (512,)))
    )
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
        q_network = MLPDuelingNoisyQNetwork(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=hidden_sizes).to(
            device
        )
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
        algorithm_kwargs["persistent_advantage_alpha"] = float(
            config.algo_kwargs.get("persistent_advantage_alpha", 0.9)
        )
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
        algorithm_kwargs["jowa_world_model_loss_coef"] = float(
            config.algo_kwargs.get("jowa_world_model_loss_coef", 1.0)
        )
        algorithm_kwargs["jowa_reward_loss_coef"] = float(config.algo_kwargs.get("jowa_reward_loss_coef", 1.0))
        algorithm_kwargs["jowa_reconstruction_loss_coef"] = float(
            config.algo_kwargs.get("jowa_reconstruction_loss_coef", 1.0)
        )
        algorithm_kwargs["jowa_consistency_loss_coef"] = float(
            config.algo_kwargs.get("jowa_consistency_loss_coef", 0.5)
        )
    return algorithm_kwargs


def _load_dqn_algorithm(
    config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device
) -> DQNAlgorithm:
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


def _load_c51_dqn_algorithm(
    config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device
) -> C51DQNAlgorithm:
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


def _load_qr_dqn_algorithm(
    config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device
) -> QRDQNAlgorithm:
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


def _load_iqn_algorithm(
    config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device
) -> IQNAlgorithm:
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


def _load_fqf_algorithm(
    config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device
) -> FQFAlgorithm:
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
        fraction_learning_rate=float(
            config.algo_kwargs.get("fraction_learning_rate", config.algo_kwargs.get("learning_rate", 1e-3))
        ),
        gamma=float(config.algo_kwargs.get("gamma", 0.99)),
        target_update_interval=int(config.algo_kwargs.get("target_update_interval", 250)),
        num_quantiles=num_quantiles,
        kappa=kappa,
        entropy_coef=entropy_coef,
    )
    algorithm.load_state_dict(checkpoint_state.algorithm_state)
    return algorithm
