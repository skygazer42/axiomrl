import torch

from axiomrl.algorithms.agent57 import Agent57 as Agent57Algorithm
from axiomrl.algorithms.drqn import DRQN as DRQNAlgorithm
from axiomrl.algorithms.r2d2 import R2D2 as R2D2Algorithm
from axiomrl.contrib.recurrent_ppo import RecurrentPPOAlgorithm
from axiomrl.experiment.checkpointing import CheckpointState
from axiomrl.experiment.config import TrainConfig
from axiomrl.experiment.registry_support import _infer_discrete_env_spaces
from axiomrl.models.recurrent import LSTMActorCritic, LSTMQNetwork
from axiomrl.models.rnd import RNDModel


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


def _load_drqn_algorithm(
    config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device
) -> DRQNAlgorithm:
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


def _load_r2d2_algorithm(
    config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device
) -> R2D2Algorithm:
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
