from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch

from rl_training.envs.goals import flatten_goal_observation
from rl_training.experiment.checkpointing import CheckpointState
from rl_training.experiment.config import TrainConfig
from rl_training.experiment.registry_core import (
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
from rl_training.experiment.registry_support import (
    _continuous_action_bounds,
    _format_action_output,
    _prepare_observation,
    _scale_continuous_actions,
)
from rl_training.runtime.decision_transformer_trainer import _build_autoregressive_window


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
