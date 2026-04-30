from __future__ import annotations

import torch

from rl_training.algorithms.awr import AWR as AWRAlgorithm
from rl_training.algorithms.cql import CQL as CQLAlgorithm
from rl_training.algorithms.crossq import CrossQ as CrossQAlgorithm
from rl_training.algorithms.curl import CURL as CURLAlgorithm
from rl_training.algorithms.d4pg import D4PG as D4PGAlgorithm
from rl_training.algorithms.ddpg import DDPG as DDPGAlgorithm
from rl_training.algorithms.drq import DrQ as DrQAlgorithm
from rl_training.algorithms.drqv2 import DrQv2 as DrQv2Algorithm
from rl_training.algorithms.edac import EDAC as EDACAlgorithm
from rl_training.algorithms.iql import IQL as IQLAlgorithm
from rl_training.algorithms.marwil import MARWIL as MARWILAlgorithm
from rl_training.algorithms.naf import NAF as NAFAlgorithm
from rl_training.algorithms.redq import REDQ as REDQAlgorithm
from rl_training.algorithms.rlpd import RLPD as RLPDAlgorithm
from rl_training.algorithms.sac import SAC as SACAlgorithm
from rl_training.algorithms.td3 import TD3 as TD3Algorithm
from rl_training.algorithms.td3_bc import TD3BC as TD3BCAlgorithm
from rl_training.algorithms.tqc import TQC as TQCAlgorithm
from rl_training.experiment.checkpointing import CheckpointState
from rl_training.experiment.config import TrainConfig
from rl_training.experiment.registry_support import (
    _infer_continuous_env_spaces,
    _infer_image_continuous_env_spaces,
)
from rl_training.models.cnn import CNNCURLModel, CNNDrQModel, CNNDrQv2Model
from rl_training.models.mlp_crossq import MLPCrossQModel
from rl_training.models.mlp_d4pg import MLPD4PGModel
from rl_training.models.mlp_ddpg import MLPDDPGModel
from rl_training.models.mlp_iql import MLPIQLModel
from rl_training.models.mlp_naf import MLPNAFModel
from rl_training.models.mlp_redq import MLPREDQModel
from rl_training.models.mlp_sac import MLPSACModel
from rl_training.models.mlp_td3 import MLPTD3Model
from rl_training.models.mlp_tqc import MLPTQCModel


def _load_iql_algorithm(
    config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device
) -> IQLAlgorithm:
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


def _load_awr_algorithm(
    config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device
) -> AWRAlgorithm:
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
        moving_average_sqd_adv_norm_start=float(config.algo_kwargs.get("moving_average_sqd_adv_norm_start", 100.0)),
        moving_average_sqd_adv_norm_update_rate=float(
            config.algo_kwargs.get("moving_average_sqd_adv_norm_update_rate", 0.01)
        ),
    )
    algorithm.load_state_dict(checkpoint_state.algorithm_state)
    return algorithm


def _load_sac_algorithm(
    config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device
) -> SACAlgorithm:
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


def _load_rlpd_algorithm(
    config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device
) -> RLPDAlgorithm:
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


def _load_cql_algorithm(
    config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device
) -> CQLAlgorithm:
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


def _load_tqc_algorithm(
    config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device
) -> TQCAlgorithm:
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


def _load_redq_algorithm(
    config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device
) -> REDQAlgorithm:
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


def _load_edac_algorithm(
    config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device
) -> EDACAlgorithm:
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


def _load_ddpg_algorithm(
    config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device
) -> DDPGAlgorithm:
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


def _load_naf_algorithm(
    config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device
) -> NAFAlgorithm:
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


def _load_d4pg_algorithm(
    config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device
) -> D4PGAlgorithm:
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


def _load_drq_algorithm(
    config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device
) -> DrQAlgorithm:
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


def _load_curl_algorithm(
    config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device
) -> CURLAlgorithm:
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


def _load_drqv2_algorithm(
    config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device
) -> DrQv2Algorithm:
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


def _load_crossq_algorithm(
    config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device
) -> CrossQAlgorithm:
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


def _load_td3_algorithm(
    config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device
) -> TD3Algorithm:
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
