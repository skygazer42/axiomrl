from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from axiomrl.algorithms.c51_dqn import C51DQN
from axiomrl.algorithms.dqn import (
    CQLDQN,
    DQN,
    AdvantageLearningDQN,
    BoltzmannDoubleDQN,
    BoltzmannDQN,
    ClippedDoubleDQN,
    CQLDoubleDQN,
    DoubleDQN,
    DuelingDQN,
    ExpectedDoubleDQN,
    ExpectedSARSA,
    HystereticDQN,
    MellowmaxDQN,
    MunchausenDoubleDQN,
    MunchausenDQN,
    PersistentAdvantageLearningDQN,
    PrioritizedDQN,
    RainbowDQN,
    SoftDoubleDQN,
    SoftDQN,
)
from axiomrl.algorithms.fqf import FQF
from axiomrl.algorithms.iqn import IQN
from axiomrl.algorithms.jowa import JOWA
from axiomrl.algorithms.qr_dqn import QRDQN
from axiomrl.algorithms.spr import SPR
from axiomrl.data.n_step import NStepAccumulator
from axiomrl.data.prioritized_replay_buffer import PrioritizedReplayBuffer
from axiomrl.data.replay_buffer import ReplayBuffer
from axiomrl.envs.factory import make_vector_env
from axiomrl.experiment.checkpointing import CheckpointState
from axiomrl.experiment.config import TrainConfig
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
from axiomrl.runtime.callbacks import Callback, CallbackList
from axiomrl.runtime.collector import CollectResult
from axiomrl.runtime.controls import (
    resolve_eval_interval,
    resolve_exploration_epsilon,
    should_run_evaluation,
)
from axiomrl.runtime.evaluation_support import evaluate_discrete_episodes
from axiomrl.runtime.resume_state import (
    capture_global_random_state,
    capture_vector_env_resume_state,
    restore_global_random_state,
    restore_vector_env_resume_state,
)
from axiomrl.runtime.run_utils import save_training_checkpoint, should_save_periodic_checkpoint
from axiomrl.runtime.session import create_training_session
from axiomrl.runtime.trainer import TrainerState, TrainResult
from axiomrl.runtime.types import MetricDict


@dataclass(frozen=True)
class _PrioritizedReplaySettings:
    total_timesteps: int
    beta_start: float
    beta_end: float
    beta_fraction: float


def _infer_spaces(envs: gym.vector.VectorEnv) -> tuple[tuple[int, ...], int]:
    obs_space = envs.single_observation_space
    action_space = envs.single_action_space

    if not isinstance(obs_space, gym.spaces.Box):
        raise TypeError(f"unsupported observation space for DQN trainer: {type(obs_space)!r}")
    if not isinstance(action_space, gym.spaces.Discrete):
        raise TypeError(f"unsupported action space for DQN trainer: {type(action_space)!r}")
    if obs_space.shape is None or len(obs_space.shape) not in (1, 3):
        raise ValueError(f"expected flat 1D or channel-first image observations, got shape={obs_space.shape!r}")

    return tuple(int(dim) for dim in obs_space.shape), int(action_space.n)


def _epsilon_at_step(
    step: int,
    *,
    total_timesteps: int,
    epsilon_start: float,
    epsilon_end: float,
    exploration_fraction: float,
) -> float:
    decay_steps = max(1, int(total_timesteps * exploration_fraction))
    progress = min(step / decay_steps, 1.0)
    return float(epsilon_start + progress * (epsilon_end - epsilon_start))


def _beta_at_step(
    step: int,
    *,
    total_timesteps: int,
    beta_start: float,
    beta_end: float,
    beta_fraction: float,
) -> float:
    if beta_fraction <= 0:
        return float(beta_end)
    decay_steps = max(1, int(total_timesteps * beta_fraction))
    progress = min(step / decay_steps, 1.0)
    return float(beta_start + progress * (beta_end - beta_start))


def _build_q_network(
    config: TrainConfig,
    *,
    obs_shape: tuple[int, ...],
    action_dim: int,
    hidden_sizes: tuple[int, ...],
) -> (
    CNNC51QNetwork
    | CNNDuelingNoisyQNetwork
    | CNNDuelingQNetwork
    | CNNFQFNetwork
    | CNNIQNetwork
    | CNNJOWAQNetwork
    | CNNNoisyQNetwork
    | CNNQNetwork
    | CNNQRQNetwork
    | CNNSPRQNetwork
    | MLPQNetwork
    | MLPDuelingQNetwork
    | MLPNoisyQNetwork
    | MLPDuelingNoisyQNetwork
    | MLPC51QNetwork
    | MLPQRQNetwork
    | MLPIQNetwork
    | MLPFQFNetwork
):
    if len(obs_shape) == 3:
        head_hidden_sizes = tuple(
            config.algo_kwargs.get(
                "head_hidden_sizes",
                config.algo_kwargs.get("hidden_sizes", hidden_sizes or (512,)),
            )
        )
        features_dim = int(config.algo_kwargs.get("features_dim", 512))

        if config.algo == "dueling_dqn":
            return CNNDuelingQNetwork(
                obs_shape=obs_shape,
                action_dim=action_dim,
                hidden_sizes=head_hidden_sizes,
                features_dim=features_dim,
            )
        if config.algo == "noisy_dqn":
            return CNNNoisyQNetwork(
                obs_shape=obs_shape,
                action_dim=action_dim,
                hidden_sizes=head_hidden_sizes,
                sigma_init=float(config.algo_kwargs.get("sigma_init", 0.5)),
                features_dim=features_dim,
            )
        if config.algo == "rainbow_dqn":
            return CNNDuelingNoisyQNetwork(
                obs_shape=obs_shape,
                action_dim=action_dim,
                hidden_sizes=head_hidden_sizes,
                sigma_init=float(config.algo_kwargs.get("sigma_init", 0.5)),
                features_dim=features_dim,
            )
        if config.algo == "c51_dqn":
            return CNNC51QNetwork(
                obs_shape=obs_shape,
                action_dim=action_dim,
                v_min=float(config.algo_kwargs.get("v_min", 0.0)),
                v_max=float(config.algo_kwargs.get("v_max", 200.0)),
                num_atoms=int(config.algo_kwargs.get("num_atoms", 51)),
                hidden_sizes=head_hidden_sizes,
                features_dim=features_dim,
            )
        if config.algo == "qr_dqn":
            return CNNQRQNetwork(
                obs_shape=obs_shape,
                action_dim=action_dim,
                num_quantiles=int(config.algo_kwargs.get("num_quantiles", 51)),
                hidden_sizes=head_hidden_sizes,
                features_dim=features_dim,
            )
        if config.algo == "iqn":
            return CNNIQNetwork(
                obs_shape=obs_shape,
                action_dim=action_dim,
                num_quantiles=int(config.algo_kwargs.get("num_quantiles", 32)),
                hidden_sizes=head_hidden_sizes,
                embedding_dim=int(config.algo_kwargs.get("embedding_dim", 64)),
                features_dim=features_dim,
            )
        if config.algo == "fqf":
            return CNNFQFNetwork(
                obs_shape=obs_shape,
                action_dim=action_dim,
                num_quantiles=int(config.algo_kwargs.get("num_quantiles", 32)),
                hidden_sizes=head_hidden_sizes,
                embedding_dim=int(config.algo_kwargs.get("embedding_dim", 64)),
                features_dim=features_dim,
            )
        if config.algo == "spr":
            return CNNSPRQNetwork(
                obs_shape=obs_shape,
                action_dim=action_dim,
                hidden_sizes=head_hidden_sizes,
                features_dim=features_dim,
                transition_hidden_size=int(config.algo_kwargs.get("spr_hidden_size", features_dim)),
                projection_dim=int(config.algo_kwargs.get("spr_projection_dim", 256)),
                action_embed_dim=int(config.algo_kwargs.get("spr_action_embed_dim", 64)),
            )
        if config.algo == "jowa":
            return CNNJOWAQNetwork(
                obs_shape=obs_shape,
                action_dim=action_dim,
                hidden_sizes=head_hidden_sizes,
                features_dim=features_dim,
                transition_hidden_size=int(config.algo_kwargs.get("jowa_transition_hidden_size", features_dim)),
                reward_hidden_size=int(config.algo_kwargs.get("jowa_reward_hidden_size", features_dim)),
                action_embed_dim=int(config.algo_kwargs.get("jowa_action_embed_dim", 64)),
            )

        return CNNQNetwork(
            obs_shape=obs_shape,
            action_dim=action_dim,
            hidden_sizes=head_hidden_sizes,
            features_dim=features_dim,
        )

    obs_dim = obs_shape[0]
    if config.algo in {"spr", "jowa"}:
        raise ValueError(f"{config.algo} currently supports channel-first image observations only")
    if config.algo == "dueling_dqn":
        return MLPDuelingQNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
        )
    if config.algo == "noisy_dqn":
        return MLPNoisyQNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
        )
    if config.algo == "rainbow_dqn":
        return MLPDuelingNoisyQNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
        )
    if config.algo == "c51_dqn":
        return MLPC51QNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            v_min=float(config.algo_kwargs.get("v_min", 0.0)),
            v_max=float(config.algo_kwargs.get("v_max", 200.0)),
            num_atoms=int(config.algo_kwargs.get("num_atoms", 51)),
            hidden_sizes=hidden_sizes,
        )
    if config.algo == "qr_dqn":
        return MLPQRQNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            num_quantiles=int(config.algo_kwargs.get("num_quantiles", 51)),
            hidden_sizes=hidden_sizes,
        )
    if config.algo == "iqn":
        return MLPIQNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            num_quantiles=int(config.algo_kwargs.get("num_quantiles", 32)),
            hidden_sizes=hidden_sizes,
            embedding_dim=int(config.algo_kwargs.get("embedding_dim", 64)),
        )
    if config.algo == "fqf":
        return MLPFQFNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            num_quantiles=int(config.algo_kwargs.get("num_quantiles", 32)),
            hidden_sizes=hidden_sizes,
            embedding_dim=int(config.algo_kwargs.get("embedding_dim", 64)),
        )
    return MLPQNetwork(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=hidden_sizes,
    )


def _build_distributional_dqn_algorithm(
    config: TrainConfig,
    *,
    q_network: nn.Module,
    learning_rate: float,
    gamma: float,
    target_update_interval: int,
) -> C51DQN | QRDQN | IQN | FQF | None:
    if config.algo == "c51_dqn":
        if not isinstance(q_network, CNNC51QNetwork | MLPC51QNetwork):
            raise TypeError(f"expected C51 q_network for c51_dqn, got {type(q_network)!r}")
        return C51DQN(
            q_network=q_network,
            learning_rate=learning_rate,
            gamma=gamma,
            target_update_interval=target_update_interval,
            v_min=float(config.algo_kwargs.get("v_min", q_network.v_min)),  # type: ignore[attr-defined]
            v_max=float(config.algo_kwargs.get("v_max", q_network.v_max)),  # type: ignore[attr-defined]
            num_atoms=int(config.algo_kwargs.get("num_atoms", q_network.num_atoms)),  # type: ignore[attr-defined]
        )
    if config.algo == "qr_dqn":
        if not isinstance(q_network, CNNQRQNetwork | MLPQRQNetwork):
            raise TypeError(f"expected QR q_network for qr_dqn, got {type(q_network)!r}")
        num_quantiles = int(config.algo_kwargs.get("num_quantiles", q_network.num_quantiles))  # type: ignore[attr-defined]
        return QRDQN(
            q_network=q_network,
            learning_rate=learning_rate,
            gamma=gamma,
            target_update_interval=target_update_interval,
            num_quantiles=num_quantiles,
            kappa=float(config.algo_kwargs.get("kappa", 1.0)),
        )
    if config.algo == "iqn":
        if not isinstance(q_network, CNNIQNetwork | MLPIQNetwork):
            raise TypeError(f"expected IQN q_network for iqn, got {type(q_network)!r}")
        num_quantiles = int(config.algo_kwargs.get("num_quantiles", q_network.num_quantiles))  # type: ignore[attr-defined]
        return IQN(
            q_network=q_network,
            learning_rate=learning_rate,
            gamma=gamma,
            target_update_interval=target_update_interval,
            num_quantiles=num_quantiles,
            kappa=float(config.algo_kwargs.get("kappa", 1.0)),
        )
    if config.algo == "fqf":
        if not isinstance(q_network, CNNFQFNetwork | MLPFQFNetwork):
            raise TypeError(f"expected FQF q_network for fqf, got {type(q_network)!r}")
        num_quantiles = int(config.algo_kwargs.get("num_quantiles", q_network.num_quantiles))  # type: ignore[attr-defined]
        return FQF(
            q_network=q_network,
            learning_rate=learning_rate,
            fraction_learning_rate=float(config.algo_kwargs.get("fraction_learning_rate", learning_rate)),
            gamma=gamma,
            target_update_interval=target_update_interval,
            num_quantiles=num_quantiles,
            kappa=float(config.algo_kwargs.get("kappa", 1.0)),
            entropy_coef=float(config.algo_kwargs.get("entropy_coef", 1e-3)),
        )
    return None


def _build_specialized_dqn_algorithm(
    config: TrainConfig,
    *,
    q_network: MLPQNetwork | MLPDuelingQNetwork | MLPNoisyQNetwork | MLPDuelingNoisyQNetwork,
    learning_rate: float,
    gamma: float,
    target_update_interval: int,
) -> (
    MellowmaxDQN
    | SoftDQN
    | ExpectedSARSA
    | ExpectedDoubleDQN
    | BoltzmannDQN
    | BoltzmannDoubleDQN
    | AdvantageLearningDQN
    | PersistentAdvantageLearningDQN
    | MunchausenDQN
    | CQLDQN
    | SoftDoubleDQN
    | MunchausenDoubleDQN
    | CQLDoubleDQN
    | ClippedDoubleDQN
    | HystereticDQN
    | None
):
    builders = {
        "expected_sarsa": lambda: ExpectedSARSA(
            q_network=q_network,
            learning_rate=learning_rate,
            gamma=gamma,
            target_update_interval=target_update_interval,
            target_epsilon=float(config.algo_kwargs.get("target_epsilon", 0.05)),
        ),
        "expected_double_dqn": lambda: ExpectedDoubleDQN(
            q_network=q_network,
            learning_rate=learning_rate,
            gamma=gamma,
            target_update_interval=target_update_interval,
            target_epsilon=float(config.algo_kwargs.get("target_epsilon", 0.05)),
        ),
        "boltzmann_dqn": lambda: BoltzmannDQN(
            q_network=q_network,
            learning_rate=learning_rate,
            gamma=gamma,
            target_update_interval=target_update_interval,
            boltzmann_temperature=float(config.algo_kwargs.get("boltzmann_temperature", 0.5)),
        ),
        "boltzmann_double_dqn": lambda: BoltzmannDoubleDQN(
            q_network=q_network,
            learning_rate=learning_rate,
            gamma=gamma,
            target_update_interval=target_update_interval,
            boltzmann_temperature=float(config.algo_kwargs.get("boltzmann_temperature", 0.5)),
        ),
        "mellowmax_dqn": lambda: MellowmaxDQN(
            q_network=q_network,
            learning_rate=learning_rate,
            gamma=gamma,
            target_update_interval=target_update_interval,
            mellowmax_omega=float(config.algo_kwargs.get("mellowmax_omega", 5.0)),
        ),
        "soft_dqn": lambda: SoftDQN(
            q_network=q_network,
            learning_rate=learning_rate,
            gamma=gamma,
            target_update_interval=target_update_interval,
            entropy_temperature=float(config.algo_kwargs.get("entropy_temperature", 0.03)),
        ),
        "soft_double_dqn": lambda: SoftDoubleDQN(
            q_network=q_network,
            learning_rate=learning_rate,
            gamma=gamma,
            target_update_interval=target_update_interval,
            entropy_temperature=float(config.algo_kwargs.get("entropy_temperature", 0.03)),
        ),
        "advantage_learning_dqn": lambda: AdvantageLearningDQN(
            q_network=q_network,
            learning_rate=learning_rate,
            gamma=gamma,
            target_update_interval=target_update_interval,
            advantage_alpha=float(config.algo_kwargs.get("advantage_alpha", 0.9)),
        ),
        "persistent_advantage_learning_dqn": lambda: PersistentAdvantageLearningDQN(
            q_network=q_network,
            learning_rate=learning_rate,
            gamma=gamma,
            target_update_interval=target_update_interval,
            persistent_advantage_alpha=float(config.algo_kwargs.get("persistent_advantage_alpha", 0.9)),
        ),
        "munchausen_dqn": lambda: MunchausenDQN(
            q_network=q_network,
            learning_rate=learning_rate,
            gamma=gamma,
            target_update_interval=target_update_interval,
            munchausen_alpha=float(config.algo_kwargs.get("munchausen_alpha", 0.9)),
            entropy_temperature=float(config.algo_kwargs.get("entropy_temperature", 0.03)),
            munchausen_clip_min=float(config.algo_kwargs.get("munchausen_clip_min", -1.0)),
        ),
        "munchausen_double_dqn": lambda: MunchausenDoubleDQN(
            q_network=q_network,
            learning_rate=learning_rate,
            gamma=gamma,
            target_update_interval=target_update_interval,
            munchausen_alpha=float(config.algo_kwargs.get("munchausen_alpha", 0.9)),
            entropy_temperature=float(config.algo_kwargs.get("entropy_temperature", 0.03)),
            munchausen_clip_min=float(config.algo_kwargs.get("munchausen_clip_min", -1.0)),
        ),
        "cql_dqn": lambda: CQLDQN(
            q_network=q_network,
            learning_rate=learning_rate,
            gamma=gamma,
            target_update_interval=target_update_interval,
            cql_alpha=float(config.algo_kwargs.get("cql_alpha", 1.0)),
        ),
        "cql_double_dqn": lambda: CQLDoubleDQN(
            q_network=q_network,
            learning_rate=learning_rate,
            gamma=gamma,
            target_update_interval=target_update_interval,
            cql_alpha=float(config.algo_kwargs.get("cql_alpha", 1.0)),
        ),
        "clipped_double_dqn": lambda: ClippedDoubleDQN(
            q_network=q_network,
            learning_rate=learning_rate,
            gamma=gamma,
            target_update_interval=target_update_interval,
        ),
        "hysteretic_dqn": lambda: HystereticDQN(
            q_network=q_network,
            learning_rate=learning_rate,
            gamma=gamma,
            target_update_interval=target_update_interval,
            hysteretic_beta=float(config.algo_kwargs.get("hysteretic_beta", 0.1)),
        ),
    }
    builder = builders.get(config.algo)
    return builder() if builder is not None else None


def _select_default_dqn_algorithm_class(algo: str) -> type[DQN]:
    if algo == "double_dqn":
        return DoubleDQN
    if algo == "dueling_dqn":
        return DuelingDQN
    if algo == "prioritized_dqn":
        return PrioritizedDQN
    if algo == "rainbow_dqn":
        return RainbowDQN
    return DQN


def _build_algorithm(
    config: TrainConfig,
    *,
    q_network: nn.Module,
    learning_rate: float,
    gamma: float,
    target_update_interval: int,
) -> (
    DQN
    | MellowmaxDQN
    | SoftDQN
    | ExpectedSARSA
    | ExpectedDoubleDQN
    | BoltzmannDQN
    | BoltzmannDoubleDQN
    | AdvantageLearningDQN
    | PersistentAdvantageLearningDQN
    | MunchausenDQN
    | CQLDQN
    | SoftDoubleDQN
    | MunchausenDoubleDQN
    | CQLDoubleDQN
    | ClippedDoubleDQN
    | HystereticDQN
    | C51DQN
    | QRDQN
    | IQN
    | FQF
    | SPR
    | JOWA
):
    if config.algo == "spr":
        if not isinstance(q_network, CNNSPRQNetwork):
            raise TypeError(f"expected SPR q_network for spr, got {type(q_network)!r}")
        return SPR(
            q_network=q_network,
            learning_rate=learning_rate,
            gamma=gamma,
            target_update_interval=target_update_interval,
            spr_loss_coef=float(config.algo_kwargs.get("spr_loss_coef", 1.0)),
        )
    if config.algo == "jowa":
        if not isinstance(q_network, CNNJOWAQNetwork):
            raise TypeError(f"expected JOWA q_network for jowa, got {type(q_network)!r}")
        return JOWA(
            q_network=q_network,
            learning_rate=learning_rate,
            gamma=gamma,
            target_update_interval=target_update_interval,
            jowa_world_model_loss_coef=float(config.algo_kwargs.get("jowa_world_model_loss_coef", 1.0)),
            jowa_reward_loss_coef=float(config.algo_kwargs.get("jowa_reward_loss_coef", 1.0)),
            jowa_reconstruction_loss_coef=float(config.algo_kwargs.get("jowa_reconstruction_loss_coef", 1.0)),
            jowa_consistency_loss_coef=float(config.algo_kwargs.get("jowa_consistency_loss_coef", 0.5)),
        )

    distributional_algorithm = _build_distributional_dqn_algorithm(
        config,
        q_network=q_network,
        learning_rate=learning_rate,
        gamma=gamma,
        target_update_interval=target_update_interval,
    )
    if distributional_algorithm is not None:
        return distributional_algorithm

    specialized_algorithm = _build_specialized_dqn_algorithm(
        config,
        q_network=q_network,
        learning_rate=learning_rate,
        gamma=gamma,
        target_update_interval=target_update_interval,
    )
    if specialized_algorithm is not None:
        return specialized_algorithm

    algorithm_cls = _select_default_dqn_algorithm_class(config.algo)

    return algorithm_cls(
        q_network=q_network,
        learning_rate=learning_rate,
        gamma=gamma,
        target_update_interval=target_update_interval,
    )


def _uses_n_step_returns(algo: str, *, n_step: int) -> bool:
    return algo in {"n_step_dqn", "rainbow_dqn"} and n_step > 1


def _uses_prioritized_replay(algo: str) -> bool:
    return algo in {"prioritized_dqn", "rainbow_dqn"}


def _resolve_algorithm_gamma(*, gamma: float, n_step: int, use_n_step_returns: bool) -> float:
    if not use_n_step_returns:
        return gamma
    return gamma**n_step


def _build_dqn_replay_buffer(
    *,
    use_prioritized_replay: bool,
    buffer_capacity: int,
    obs_shape: tuple[int, ...],
    device: torch.device,
    prioritized_alpha: float,
    prioritized_eps: float,
) -> ReplayBuffer | PrioritizedReplayBuffer:
    obs_dtype = torch.uint8 if len(obs_shape) == 3 else torch.float32
    if use_prioritized_replay:
        return PrioritizedReplayBuffer(
            capacity=buffer_capacity,
            obs_shape=obs_shape,
            action_shape=(),
            alpha=prioritized_alpha,
            priority_eps=prioritized_eps,
            device=device,
            obs_dtype=obs_dtype,
        )
    return ReplayBuffer(
        capacity=buffer_capacity,
        obs_shape=obs_shape,
        action_shape=(),
        device=device,
        obs_dtype=obs_dtype,
    )


def _build_n_step_accumulator(
    *,
    use_n_step_returns: bool,
    num_envs: int,
    n_step: int,
    gamma: float,
) -> NStepAccumulator | None:
    if not use_n_step_returns:
        return None
    return NStepAccumulator(num_envs=num_envs, n_step=n_step, gamma=gamma)


def _store_transitions(
    *,
    replay_buffer: ReplayBuffer | PrioritizedReplayBuffer,
    n_step_accumulator: NStepAccumulator | None,
    num_envs: int,
    obs: np.ndarray,
    actions: torch.Tensor,
    rewards: np.ndarray,
    next_obs: np.ndarray,
    dones: np.ndarray,
) -> None:
    for env_index in range(num_envs):
        action = int(actions[env_index].item())
        reward = float(rewards[env_index])
        done = bool(dones[env_index])
        if n_step_accumulator is None:
            replay_buffer.add(
                obs=obs[env_index],
                actions=action,
                rewards=reward,
                next_obs=next_obs[env_index],
                dones=float(done),
            )
            continue

        for transition in n_step_accumulator.add(
            env_index,
            obs[env_index],
            action,
            reward,
            next_obs[env_index],
            done,
        ):
            replay_buffer.add(**transition)


def _should_run_dqn_update(
    *,
    replay_size: int,
    batch_size: int,
    learning_starts: int,
    global_step: int,
    train_frequency: int,
) -> bool:
    return replay_size >= max(batch_size, learning_starts) and global_step % train_frequency == 0


def _emit_collect_event(
    callback_list: CallbackList,
    trainer_state: TrainerState,
    *,
    global_step: int,
    num_envs: int,
    dones: np.ndarray,
    replay_size: int,
    obs: np.ndarray,
) -> None:
    callback_list.on_collect_end(
        trainer_state,
        CollectResult(
            num_env_steps=num_envs,
            num_episodes=int(np.sum(dones)),
            metrics={"global_step": float(global_step), "buffer_size": float(replay_size)},
            last_obs=obs,
        ),
    )


def _update_with_prioritized_replay(
    *,
    algorithm: object,
    replay_buffer: PrioritizedReplayBuffer,
    batch_size: int,
    global_step: int,
    settings: _PrioritizedReplaySettings,
):
    beta = _beta_at_step(
        global_step,
        total_timesteps=settings.total_timesteps,
        beta_start=settings.beta_start,
        beta_end=settings.beta_end,
        beta_fraction=settings.beta_fraction,
    )
    batch = replay_buffer.sample(batch_size, beta=beta)
    result = algorithm.update(batch, global_step=global_step)  # type: ignore[attr-defined]
    td_errors = getattr(algorithm, "last_td_errors", None)
    if td_errors is not None:
        replay_buffer.update_priorities(batch["indices"], td_errors)
    return result


def _maybe_update_dqn(
    *,
    algorithm: object,
    replay_buffer: ReplayBuffer | PrioritizedReplayBuffer,
    use_prioritized_replay: bool,
    batch_size: int,
    learning_starts: int,
    train_frequency: int,
    global_step: int,
    prioritized_replay_settings: _PrioritizedReplaySettings,
    callback_list: CallbackList,
    trainer_state: TrainerState,
    latest_update_metrics: MetricDict,
    update_count: int,
) -> tuple[MetricDict, int]:
    if not _should_run_dqn_update(
        replay_size=len(replay_buffer),
        batch_size=batch_size,
        learning_starts=learning_starts,
        global_step=global_step,
        train_frequency=train_frequency,
    ):
        return latest_update_metrics, update_count

    if use_prioritized_replay:
        result = _update_with_prioritized_replay(
            algorithm=algorithm,
            replay_buffer=replay_buffer,  # type: ignore[arg-type]
            batch_size=batch_size,
            global_step=global_step,
            settings=prioritized_replay_settings,
        )
    else:
        result = algorithm.update(replay_buffer.sample(batch_size), global_step=global_step)  # type: ignore[attr-defined]

    callback_list.on_update_end(trainer_state, result)
    return result.metrics, update_count + result.num_gradient_steps


def _build_dqn_metrics(
    *,
    latest_update_metrics: MetricDict,
    epsilon: float,
    global_step: int,
    replay_size: int,
    update_count: int,
    include_beta: bool,
    total_timesteps: int,
    prioritized_beta_start: float,
    prioritized_beta_end: float,
    prioritized_beta_fraction: float,
) -> MetricDict:
    metrics: MetricDict = {
        **latest_update_metrics,
        "epsilon": epsilon,
        "global_step": float(global_step),
        "buffer_size": float(replay_size),
        "gradient_steps": float(update_count),
    }
    if include_beta:
        metrics["beta"] = float(
            _beta_at_step(
                global_step,
                total_timesteps=total_timesteps,
                beta_start=prioritized_beta_start,
                beta_end=prioritized_beta_end,
                beta_fraction=prioritized_beta_fraction,
            )
        )
    return metrics


def _maybe_run_dqn_evaluation(
    *,
    should_run_eval: bool,
    algorithm: object,
    q_network: CNNQNetwork
    | MLPQNetwork
    | MLPDuelingQNetwork
    | MLPNoisyQNetwork
    | MLPDuelingNoisyQNetwork
    | MLPC51QNetwork
    | MLPQRQNetwork
    | MLPIQNetwork,
    config: TrainConfig,
    device: torch.device,
    logger: object,
    callback_list: CallbackList,
    trainer_state: TrainerState,
    metrics: MetricDict,
    global_step: int,
) -> tuple[MetricDict, bool]:
    if not should_run_eval:
        return metrics, False

    algorithm.set_eval_mode()  # type: ignore[attr-defined]
    eval_metrics = _evaluate_q_policy(
        q_network,
        config,
        device=device,
        num_episodes=config.eval_episodes,
    )
    algorithm.set_train_mode()  # type: ignore[attr-defined]
    evaluated_metrics = {**metrics, **eval_metrics}
    logger.log_metrics(evaluated_metrics, step=global_step)
    callback_list.on_eval_end(trainer_state, evaluated_metrics)
    return evaluated_metrics, trainer_state.should_stop


def _evaluate_q_policy(
    q_network: CNNQNetwork
    | MLPQNetwork
    | MLPDuelingQNetwork
    | MLPNoisyQNetwork
    | MLPDuelingNoisyQNetwork
    | MLPC51QNetwork
    | MLPQRQNetwork
    | MLPIQNetwork,
    config: TrainConfig,
    *,
    device: torch.device,
    num_episodes: int,
) -> MetricDict:
    def action_fn(obs_tensor: torch.Tensor) -> int:
        with torch.no_grad():
            action = q_network.act(obs_tensor, epsilon=0.0).squeeze(0)
        return int(action.item())

    return evaluate_discrete_episodes(
        config,
        device=device,
        num_episodes=num_episodes,
        action_fn=action_fn,
    )


def train_dqn(
    config: TrainConfig,
    *,
    run_suffix: str | None = None,
    checkpoint_state: CheckpointState | None = None,
    callbacks: Sequence[Callback] | None = None,
) -> TrainResult:
    session = create_training_session(config, algorithm=config.algo, run_suffix=run_suffix, callbacks=callbacks)
    device = session.device
    run_context = session.run_context
    logger = session.logger
    callback_list = session.callback_list
    trainer_state = session.trainer_state

    buffer_capacity = int(config.algo_kwargs.get("buffer_capacity", 10000))
    batch_size = int(config.algo_kwargs.get("batch_size", 64))
    learning_starts = int(config.algo_kwargs.get("learning_starts", 1000))
    train_frequency = int(config.algo_kwargs.get("train_frequency", 1))
    target_update_interval = int(config.algo_kwargs.get("target_update_interval", 250))
    hidden_sizes = tuple(config.algo_kwargs.get("hidden_sizes", (64, 64)))
    learning_rate = float(config.algo_kwargs.get("learning_rate", 1e-3))
    gamma = float(config.algo_kwargs.get("gamma", 0.99))
    n_step = int(config.algo_kwargs.get("n_step", 1))
    prioritized_alpha = float(config.algo_kwargs.get("prioritized_alpha", 0.6))
    prioritized_beta_start = float(config.algo_kwargs.get("prioritized_beta_start", 0.4))
    prioritized_beta_end = float(config.algo_kwargs.get("prioritized_beta_end", 1.0))
    prioritized_beta_fraction = float(config.algo_kwargs.get("prioritized_beta_fraction", 1.0))
    prioritized_eps = float(config.algo_kwargs.get("prioritized_eps", 1e-6))
    eval_interval = resolve_eval_interval(config)
    prioritized_replay_settings = _PrioritizedReplaySettings(
        total_timesteps=config.total_timesteps,
        beta_start=prioritized_beta_start,
        beta_end=prioritized_beta_end,
        beta_fraction=prioritized_beta_fraction,
    )

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    envs = None
    checkpoint_path: Path | None = None
    metrics: MetricDict = {}

    try:
        envs = make_vector_env(config)
        if n_step <= 0:
            raise ValueError(f"n_step must be > 0, got {n_step}")

        obs_shape, action_dim = _infer_spaces(envs)
        use_n_step_returns = _uses_n_step_returns(config.algo, n_step=n_step)
        use_prioritized_replay = _uses_prioritized_replay(config.algo)
        q_network = _build_q_network(
            config,
            obs_shape=obs_shape,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
        ).to(device)
        algorithm_gamma = _resolve_algorithm_gamma(
            gamma=gamma,
            n_step=n_step,
            use_n_step_returns=use_n_step_returns,
        )

        algorithm = _build_algorithm(
            config,
            q_network=q_network,
            learning_rate=learning_rate,
            gamma=algorithm_gamma,
            target_update_interval=target_update_interval,
        )
        replay_buffer = _build_dqn_replay_buffer(
            use_prioritized_replay=use_prioritized_replay,
            buffer_capacity=buffer_capacity,
            obs_shape=obs_shape,
            device=device,
            prioritized_alpha=prioritized_alpha,
            prioritized_eps=prioritized_eps,
        )
        if checkpoint_state is not None:
            algorithm.load_state_dict(checkpoint_state.algorithm_state)
            if checkpoint_state.buffer_state is not None:
                replay_buffer.load_state_dict(checkpoint_state.buffer_state)

        n_step_accumulator = _build_n_step_accumulator(
            use_n_step_returns=use_n_step_returns,
            num_envs=config.num_envs,
            n_step=n_step,
            gamma=gamma,
        )

        obs, _ = envs.reset(seed=config.seed)
        global_step = int(checkpoint_state.trainer_state.get("global_step", 0)) if checkpoint_state is not None else 0
        update_count = int(checkpoint_state.trainer_state.get("update_count", 0)) if checkpoint_state is not None else 0
        latest_update_metrics: MetricDict = {}
        if checkpoint_state is not None:
            resume_context = checkpoint_state.trainer_state.get("resume_context")
            if isinstance(resume_context, dict):
                env_resume_state = resume_context.get("env_state")
                if isinstance(env_resume_state, dict):
                    restored_obs = restore_vector_env_resume_state(envs, env_resume_state)
                    if restored_obs is not None:
                        obs = restored_obs
                random_state = resume_context.get("random_state")
                if isinstance(random_state, dict):
                    restore_global_random_state(random_state)
                n_step_state = resume_context.get("n_step_accumulator")
                if n_step_accumulator is not None and isinstance(n_step_state, dict):
                    n_step_accumulator.load_state_dict(n_step_state)
        trainer_state.global_step = global_step
        trainer_state.update_count = update_count
        callback_list.on_train_start(trainer_state)
        last_checkpoint_step = global_step

        def _save_checkpoint() -> Path:
            return save_training_checkpoint(
                run_context=run_context,
                config=config,
                algorithm_state=algorithm.state_dict(),
                buffer_state=replay_buffer.state_dict(),
                trainer_state={
                    "global_step": global_step,
                    "update_count": update_count,
                    "should_stop": trainer_state.should_stop,
                    "stop_reason": trainer_state.stop_reason,
                    "resume_context": {
                        "env_state": capture_vector_env_resume_state(envs),
                        "random_state": capture_global_random_state(),
                        "n_step_accumulator": (
                            n_step_accumulator.state_dict() if n_step_accumulator is not None else None
                        ),
                    },
                },
                metrics=metrics,
            )

        while global_step < config.total_timesteps:
            epsilon = resolve_exploration_epsilon(config, step=global_step)

            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                actions = q_network.act(obs_tensor, epsilon=epsilon)

            next_obs, rewards, terminated, truncated, _ = envs.step(actions.cpu().numpy())
            dones = np.logical_or(terminated, truncated).astype(np.float32)

            _store_transitions(
                replay_buffer=replay_buffer,
                n_step_accumulator=n_step_accumulator,
                num_envs=config.num_envs,
                obs=obs,
                actions=actions,
                rewards=rewards,
                next_obs=next_obs,
                dones=dones,
            )

            obs = next_obs
            global_step += config.num_envs
            trainer_state.global_step = global_step
            _emit_collect_event(
                callback_list,
                trainer_state,
                global_step=global_step,
                num_envs=config.num_envs,
                dones=dones,
                replay_size=len(replay_buffer),
                obs=obs,
            )

            latest_update_metrics, update_count = _maybe_update_dqn(
                algorithm=algorithm,
                replay_buffer=replay_buffer,
                use_prioritized_replay=use_prioritized_replay,
                batch_size=batch_size,
                learning_starts=learning_starts,
                train_frequency=train_frequency,
                global_step=global_step,
                prioritized_replay_settings=prioritized_replay_settings,
                callback_list=callback_list,
                trainer_state=trainer_state,
                latest_update_metrics=latest_update_metrics,
                update_count=update_count,
            )
            trainer_state.update_count = update_count

            metrics = _build_dqn_metrics(
                latest_update_metrics=latest_update_metrics,
                epsilon=epsilon,
                global_step=global_step,
                replay_size=len(replay_buffer),
                update_count=update_count,
                include_beta=use_prioritized_replay,
                total_timesteps=config.total_timesteps,
                prioritized_beta_start=prioritized_beta_start,
                prioritized_beta_end=prioritized_beta_end,
                prioritized_beta_fraction=prioritized_beta_fraction,
            )
            metrics, should_stop = _maybe_run_dqn_evaluation(
                should_run_eval=should_run_evaluation(
                    global_step=global_step,
                    total_timesteps=config.total_timesteps,
                    eval_interval=eval_interval,
                ),
                algorithm=algorithm,
                q_network=q_network,
                config=config,
                device=device,
                logger=logger,
                callback_list=callback_list,
                trainer_state=trainer_state,
                metrics=metrics,
                global_step=global_step,
            )
            if should_save_periodic_checkpoint(
                global_step=global_step,
                last_checkpoint_step=last_checkpoint_step,
                checkpoint_interval=config.checkpoint_interval,
            ):
                checkpoint_path = _save_checkpoint()
                last_checkpoint_step = global_step
            if should_stop:
                break

        checkpoint_path = _save_checkpoint()
    finally:
        if envs is not None:
            envs.close()
        session.close()
    result = TrainResult(
        run_dir=run_context.run_dir,
        checkpoint_path=checkpoint_path,
        metrics=metrics,
    )
    callback_list.on_train_end(trainer_state, result)
    return result
