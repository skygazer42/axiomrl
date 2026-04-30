from __future__ import annotations

import torch

from axiomrl.algorithms.a2c import A2C as A2CAlgorithm
from axiomrl.algorithms.appo import APPO as APPOAlgorithm
from axiomrl.algorithms.ars import ARS as ARSAlgorithm
from axiomrl.algorithms.bc import BC as BCAlgorithm
from axiomrl.algorithms.decision_transformer import (
    DecisionTransformer as DecisionTransformerAlgorithm,
)
from axiomrl.algorithms.gail import GAIL as GAILAlgorithm
from axiomrl.algorithms.impala import IMPALA as IMPALAAlgorithm
from axiomrl.algorithms.openai_es import OpenAIES as OpenAIESAlgorithm
from axiomrl.algorithms.ppg import PPG as PPGAlgorithm
from axiomrl.algorithms.ppo import PPO as PPOAlgorithm
from axiomrl.algorithms.trpo import TRPO as TRPOAlgorithm
from axiomrl.experiment.checkpointing import CheckpointState
from axiomrl.experiment.config import TrainConfig
from axiomrl.experiment.registry_support import (
    _infer_continuous_env_spaces,
    _infer_discrete_env_spaces,
)
from axiomrl.models.cnn import CNNActorCritic, CNNPPGModel
from axiomrl.models.decision_transformer import DecisionTransformerModel
from axiomrl.models.mlp_actor_critic import MLPActorCritic
from axiomrl.models.mlp_ars import MLPARSModel
from axiomrl.models.mlp_bc import MLPBCModel
from axiomrl.models.mlp_gail_discriminator import (
    CNNGAILDiscriminator,
    MLPGAILDiscriminator,
)
from axiomrl.models.mlp_ppg import MLPPPGModel


def _load_a2c_algorithm(
    config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device
) -> A2CAlgorithm:
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


def _load_ars_algorithm(
    config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device
) -> ARSAlgorithm:
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


def _load_ppo_algorithm(
    config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device
) -> PPOAlgorithm:
    obs_shape, action_dim = _infer_discrete_env_spaces(config)
    if len(obs_shape) == 1:
        hidden_sizes = tuple(config.algo_kwargs.get("hidden_sizes", (64, 64)))
        policy = MLPActorCritic(obs_dim=obs_shape[0], action_dim=action_dim, hidden_sizes=hidden_sizes).to(device)
    else:
        policy = CNNActorCritic(
            obs_shape=obs_shape,
            action_dim=action_dim,
            hidden_sizes=tuple(
                config.algo_kwargs.get("head_hidden_sizes", config.algo_kwargs.get("hidden_sizes", (512,)))
            ),
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


def _load_gail_algorithm(
    config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device
) -> GAILAlgorithm:
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
            hidden_sizes=tuple(
                config.algo_kwargs.get("head_hidden_sizes", config.algo_kwargs.get("hidden_sizes", (512,)))
            ),
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
            features_dim=int(
                config.algo_kwargs.get("discriminator_features_dim", config.algo_kwargs.get("features_dim", 512))
            ),
        ).to(device)

    algorithm = GAILAlgorithm(
        policy=policy,
        discriminator=discriminator,
        learning_rate=float(config.algo_kwargs.get("learning_rate", 3e-4)),
        clip_coef=float(config.algo_kwargs.get("clip_coef", 0.2)),
        ent_coef=float(config.algo_kwargs.get("ent_coef", 0.01)),
        vf_coef=float(config.algo_kwargs.get("vf_coef", 0.5)),
        discriminator_learning_rate=float(
            config.algo_kwargs.get("discriminator_learning_rate", config.algo_kwargs.get("learning_rate", 3e-4))
        ),
        max_grad_norm=float(config.algo_kwargs.get("max_grad_norm", 0.5)),
    )
    algorithm.load_state_dict(checkpoint_state.algorithm_state)
    return algorithm


def _load_ppg_algorithm(
    config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device
) -> PPGAlgorithm:
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
        aux_learning_rate=float(
            config.algo_kwargs.get("aux_learning_rate", config.algo_kwargs.get("learning_rate", 3e-4))
        ),
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


def _load_trpo_algorithm(
    config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device
) -> TRPOAlgorithm:
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
