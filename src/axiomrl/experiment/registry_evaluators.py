import torch

from axiomrl.experiment.checkpointing import CheckpointState
from axiomrl.experiment.config import TrainConfig
from axiomrl.experiment.registry_core import (
    _load_a2c_algorithm,
    _load_agent57_algorithm,
    _load_appo_algorithm,
    _load_ars_algorithm,
    _load_awac_algorithm,
    _load_awr_algorithm,
    _load_bc_algorithm,
    _load_bcq_algorithm,
    _load_bear_algorithm,
    _load_c51_dqn_algorithm,
    _load_cal_ql_algorithm,
    _load_cql_algorithm,
    _load_crossq_algorithm,
    _load_crr_algorithm,
    _load_curl_algorithm,
    _load_d4pg_algorithm,
    _load_ddpg_algorithm,
    _load_decision_transformer_algorithm,
    _load_discrete_sac_algorithm,
    _load_dqn_algorithm,
    _load_dreamer_algorithm,
    _load_drq_algorithm,
    _load_drqn_algorithm,
    _load_drqv2_algorithm,
    _load_edac_algorithm,
    _load_efficientzero_algorithm,
    _load_fqf_algorithm,
    _load_gail_algorithm,
    _load_gumbel_muzero_algorithm,
    _load_her_algorithm,
    _load_impala_algorithm,
    _load_iql_algorithm,
    _load_iqn_algorithm,
    _load_marwil_algorithm,
    _load_mbpo_algorithm,
    _load_mopo_algorithm,
    _load_muzero_algorithm,
    _load_naf_algorithm,
    _load_openai_es_algorithm,
    _load_pets_algorithm,
    _load_ppg_algorithm,
    _load_ppo_algorithm,
    _load_qr_dqn_algorithm,
    _load_r2d2_algorithm,
    _load_rebrac_algorithm,
    _load_recurrent_ppo_algorithm,
    _load_redq_algorithm,
    _load_rlpd_algorithm,
    _load_sac_algorithm,
    _load_td3_algorithm,
    _load_td3_bc_algorithm,
    _load_tqc_algorithm,
    _load_trpo_algorithm,
    _load_xql_algorithm,
)
from axiomrl.runtime.a2c_trainer import _evaluate_policy as _evaluate_a2c_policy
from axiomrl.runtime.appo_trainer import _evaluate_appo_policy
from axiomrl.runtime.ars_trainer import _evaluate_ars_policy
from axiomrl.runtime.bc_trainer import _evaluate_bc_policy
from axiomrl.runtime.bcq_trainer import _evaluate_bcq_policy
from axiomrl.runtime.crossq_trainer import _evaluate_crossq_policy
from axiomrl.runtime.curl_trainer import _evaluate_curl_policy
from axiomrl.runtime.d4pg_trainer import _evaluate_d4pg_policy
from axiomrl.runtime.ddpg_trainer import _evaluate_ddpg_policy
from axiomrl.runtime.decision_transformer_trainer import _evaluate_decision_transformer_policy
from axiomrl.runtime.discrete_sac_trainer import _evaluate_discrete_sac_policy
from axiomrl.runtime.dqn_trainer import _evaluate_q_policy
from axiomrl.runtime.drq_trainer import _evaluate_drq_policy
from axiomrl.runtime.drqn_trainer import _evaluate_drqn_policy
from axiomrl.runtime.drqv2_trainer import _evaluate_drqv2_policy
from axiomrl.runtime.her_trainer import _evaluate_her_policy
from axiomrl.runtime.impala_trainer import _evaluate_impala_policy
from axiomrl.runtime.iql_trainer import _evaluate_iql_policy
from axiomrl.runtime.muzero_trainer import _evaluate_muzero_policy
from axiomrl.runtime.naf_trainer import _evaluate_naf_policy
from axiomrl.runtime.openai_es_trainer import _evaluate_openai_es_policy
from axiomrl.runtime.pets_trainer import _evaluate_pets_policy
from axiomrl.runtime.ppg_trainer import _evaluate_ppg_policy
from axiomrl.runtime.ppo_trainer import _evaluate_policy
from axiomrl.runtime.r2d2_trainer import _evaluate_r2d2_policy
from axiomrl.runtime.recurrent_ppo_trainer import _evaluate_recurrent_policy
from axiomrl.runtime.redq_trainer import _evaluate_redq_policy
from axiomrl.runtime.sac_trainer import _evaluate_sac_policy
from axiomrl.runtime.td3_trainer import _evaluate_td3_policy
from axiomrl.runtime.tqc_trainer import _evaluate_tqc_policy
from axiomrl.runtime.types import MetricDict


def _evaluate_a2c(
    config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int
) -> MetricDict:
    algorithm = _load_a2c_algorithm(config, checkpoint_state, device=device)
    return _evaluate_a2c_policy(algorithm.policy, config, device=device, num_episodes=num_episodes)


def _evaluate_ars(
    config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int
) -> MetricDict:
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


def _evaluate_bc(
    config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int
) -> MetricDict:
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


def _evaluate_bear(
    config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int
) -> MetricDict:
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


def _evaluate_mopo(
    config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int
) -> MetricDict:
    algorithm = _load_mopo_algorithm(config, checkpoint_state, device=device)
    return _evaluate_sac_policy(algorithm.policy_model, config, device=device, num_episodes=num_episodes)


def _evaluate_mbpo(
    config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int
) -> MetricDict:
    algorithm = _load_mbpo_algorithm(config, checkpoint_state, device=device)
    return _evaluate_sac_policy(algorithm.model, config, device=device, num_episodes=num_episodes)


def _evaluate_pets(
    config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int
) -> MetricDict:
    algorithm = _load_pets_algorithm(config, checkpoint_state, device=device)
    return _evaluate_pets_policy(algorithm, config, device=device, num_episodes=num_episodes)


def _evaluate_awac(
    config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int
) -> MetricDict:
    algorithm = _load_awac_algorithm(config, checkpoint_state, device=device)
    return _evaluate_sac_policy(algorithm.model, config, device=device, num_episodes=num_episodes)


def _evaluate_crr(
    config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int
) -> MetricDict:
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


def _evaluate_her(
    config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int
) -> MetricDict:
    algorithm = _load_her_algorithm(config, checkpoint_state, device=device)
    return _evaluate_her_policy(algorithm.model, config, device=device, num_episodes=num_episodes)


def _evaluate_ppo(
    config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int
) -> MetricDict:
    algorithm = _load_ppo_algorithm(config, checkpoint_state, device=device)
    return _evaluate_policy(algorithm.policy, config, device=device, num_episodes=num_episodes)


def _evaluate_gail(
    config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int
) -> MetricDict:
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


def _evaluate_trpo(
    config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int
) -> MetricDict:
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


def _evaluate_dqn(
    config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int
) -> MetricDict:
    algorithm = _load_dqn_algorithm(config, checkpoint_state, device=device)
    algorithm.set_eval_mode()
    return _evaluate_q_policy(algorithm.q_network, config, device=device, num_episodes=num_episodes)


def _evaluate_c51_dqn(
    config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int
) -> MetricDict:
    algorithm = _load_c51_dqn_algorithm(config, checkpoint_state, device=device)
    algorithm.set_eval_mode()
    return _evaluate_q_policy(algorithm.q_network, config, device=device, num_episodes=num_episodes)


def _evaluate_qr_dqn(
    config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int
) -> MetricDict:
    algorithm = _load_qr_dqn_algorithm(config, checkpoint_state, device=device)
    algorithm.set_eval_mode()
    return _evaluate_q_policy(algorithm.q_network, config, device=device, num_episodes=num_episodes)


def _evaluate_iqn(
    config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int
) -> MetricDict:
    algorithm = _load_iqn_algorithm(config, checkpoint_state, device=device)
    algorithm.set_eval_mode()
    return _evaluate_q_policy(algorithm.q_network, config, device=device, num_episodes=num_episodes)


def _evaluate_fqf(
    config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int
) -> MetricDict:
    algorithm = _load_fqf_algorithm(config, checkpoint_state, device=device)
    algorithm.set_eval_mode()
    return _evaluate_q_policy(algorithm.q_network, config, device=device, num_episodes=num_episodes)


def _evaluate_iql(
    config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int
) -> MetricDict:
    algorithm = _load_iql_algorithm(config, checkpoint_state, device=device)
    return _evaluate_iql_policy(algorithm.model, config, device=device, num_episodes=num_episodes)


def _evaluate_awr(
    config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int
) -> MetricDict:
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


def _evaluate_xql(
    config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int
) -> MetricDict:
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


def _evaluate_sac(
    config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int
) -> MetricDict:
    algorithm = _load_sac_algorithm(config, checkpoint_state, device=device)
    return _evaluate_sac_policy(algorithm.model, config, device=device, num_episodes=num_episodes)


def _evaluate_rlpd(
    config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int
) -> MetricDict:
    algorithm = _load_rlpd_algorithm(config, checkpoint_state, device=device)
    return _evaluate_sac_policy(algorithm.model, config, device=device, num_episodes=num_episodes)


def _evaluate_cql(
    config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int
) -> MetricDict:
    algorithm = _load_cql_algorithm(config, checkpoint_state, device=device)
    return _evaluate_sac_policy(algorithm.model, config, device=device, num_episodes=num_episodes)


def _evaluate_crossq(
    config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int
) -> MetricDict:
    algorithm = _load_crossq_algorithm(config, checkpoint_state, device=device)
    return _evaluate_crossq_policy(algorithm.model, config, device=device, num_episodes=num_episodes)


def _evaluate_tqc(
    config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int
) -> MetricDict:
    algorithm = _load_tqc_algorithm(config, checkpoint_state, device=device)
    return _evaluate_tqc_policy(algorithm.model, config, device=device, num_episodes=num_episodes)


def _evaluate_redq(
    config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int
) -> MetricDict:
    algorithm = _load_redq_algorithm(config, checkpoint_state, device=device)
    return _evaluate_redq_policy(algorithm.model, config, device=device, num_episodes=num_episodes)


def _evaluate_edac(
    config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int
) -> MetricDict:
    algorithm = _load_edac_algorithm(config, checkpoint_state, device=device)
    return _evaluate_redq_policy(algorithm.model, config, device=device, num_episodes=num_episodes)


def _evaluate_ddpg(
    config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int
) -> MetricDict:
    algorithm = _load_ddpg_algorithm(config, checkpoint_state, device=device)
    return _evaluate_ddpg_policy(algorithm.model, config, device=device, num_episodes=num_episodes)


def _evaluate_naf(
    config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int
) -> MetricDict:
    algorithm = _load_naf_algorithm(config, checkpoint_state, device=device)
    return _evaluate_naf_policy(algorithm.model, config, device=device, num_episodes=num_episodes)


def _evaluate_d4pg(
    config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int
) -> MetricDict:
    algorithm = _load_d4pg_algorithm(config, checkpoint_state, device=device)
    return _evaluate_d4pg_policy(algorithm.model, config, device=device, num_episodes=num_episodes)


def _evaluate_drqn(
    config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int
) -> MetricDict:
    algorithm = _load_drqn_algorithm(config, checkpoint_state, device=device)
    algorithm.set_eval_mode()
    return _evaluate_drqn_policy(algorithm.q_network, config, device=device, num_episodes=num_episodes)


def _evaluate_r2d2(
    config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int
) -> MetricDict:
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


def _evaluate_drq(
    config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int
) -> MetricDict:
    algorithm = _load_drq_algorithm(config, checkpoint_state, device=device)
    return _evaluate_drq_policy(algorithm.model, config, device=device, num_episodes=num_episodes)


def _evaluate_curl(
    config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int
) -> MetricDict:
    algorithm = _load_curl_algorithm(config, checkpoint_state, device=device)
    return _evaluate_curl_policy(algorithm.model, config, device=device, num_episodes=num_episodes)


def _evaluate_ppg(
    config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int
) -> MetricDict:
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


def _evaluate_td3(
    config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int
) -> MetricDict:
    algorithm = _load_td3_algorithm(config, checkpoint_state, device=device)
    return _evaluate_td3_policy(algorithm.model, config, device=device, num_episodes=num_episodes)


def _evaluate_td3_bc(
    config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int
) -> MetricDict:
    algorithm = _load_td3_bc_algorithm(config, checkpoint_state, device=device)
    return _evaluate_td3_policy(algorithm.model, config, device=device, num_episodes=num_episodes)
