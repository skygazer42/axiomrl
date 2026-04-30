from __future__ import annotations

from axiomrl.experiment.registry_continuous_loaders import (
    _load_awr_algorithm as _load_awr_algorithm,
)
from axiomrl.experiment.registry_continuous_loaders import (
    _load_cql_algorithm as _load_cql_algorithm,
)
from axiomrl.experiment.registry_continuous_loaders import (
    _load_crossq_algorithm as _load_crossq_algorithm,
)
from axiomrl.experiment.registry_continuous_loaders import (
    _load_curl_algorithm as _load_curl_algorithm,
)
from axiomrl.experiment.registry_continuous_loaders import (
    _load_d4pg_algorithm as _load_d4pg_algorithm,
)
from axiomrl.experiment.registry_continuous_loaders import (
    _load_ddpg_algorithm as _load_ddpg_algorithm,
)
from axiomrl.experiment.registry_continuous_loaders import (
    _load_drq_algorithm as _load_drq_algorithm,
)
from axiomrl.experiment.registry_continuous_loaders import (
    _load_drqv2_algorithm as _load_drqv2_algorithm,
)
from axiomrl.experiment.registry_continuous_loaders import (
    _load_edac_algorithm as _load_edac_algorithm,
)
from axiomrl.experiment.registry_continuous_loaders import (
    _load_iql_algorithm as _load_iql_algorithm,
)
from axiomrl.experiment.registry_continuous_loaders import (
    _load_marwil_algorithm as _load_marwil_algorithm,
)
from axiomrl.experiment.registry_continuous_loaders import (
    _load_naf_algorithm as _load_naf_algorithm,
)
from axiomrl.experiment.registry_continuous_loaders import (
    _load_redq_algorithm as _load_redq_algorithm,
)
from axiomrl.experiment.registry_continuous_loaders import (
    _load_rlpd_algorithm as _load_rlpd_algorithm,
)
from axiomrl.experiment.registry_continuous_loaders import (
    _load_sac_algorithm as _load_sac_algorithm,
)
from axiomrl.experiment.registry_continuous_loaders import (
    _load_td3_algorithm as _load_td3_algorithm,
)
from axiomrl.experiment.registry_continuous_loaders import (
    _load_td3_bc_algorithm as _load_td3_bc_algorithm,
)
from axiomrl.experiment.registry_continuous_loaders import (
    _load_tqc_algorithm as _load_tqc_algorithm,
)
from axiomrl.experiment.registry_dqn_loaders import (
    _build_dqn_algorithm_kwargs as _build_dqn_algorithm_kwargs,
)
from axiomrl.experiment.registry_dqn_loaders import (
    _build_image_dqn_loader as _build_image_dqn_loader,
)
from axiomrl.experiment.registry_dqn_loaders import (
    _build_vector_dqn_loader as _build_vector_dqn_loader,
)
from axiomrl.experiment.registry_dqn_loaders import (
    _load_c51_dqn_algorithm as _load_c51_dqn_algorithm,
)
from axiomrl.experiment.registry_dqn_loaders import (
    _load_dqn_algorithm as _load_dqn_algorithm,
)
from axiomrl.experiment.registry_dqn_loaders import (
    _load_fqf_algorithm as _load_fqf_algorithm,
)
from axiomrl.experiment.registry_dqn_loaders import (
    _load_iqn_algorithm as _load_iqn_algorithm,
)
from axiomrl.experiment.registry_dqn_loaders import (
    _load_qr_dqn_algorithm as _load_qr_dqn_algorithm,
)
from axiomrl.experiment.registry_dqn_loaders import (
    _resolve_vector_dqn_algorithm_class as _resolve_vector_dqn_algorithm_class,
)
from axiomrl.experiment.registry_offline_loaders import (
    _load_awac_algorithm as _load_awac_algorithm,
)
from axiomrl.experiment.registry_offline_loaders import (
    _load_bcq_algorithm as _load_bcq_algorithm,
)
from axiomrl.experiment.registry_offline_loaders import (
    _load_bear_algorithm as _load_bear_algorithm,
)
from axiomrl.experiment.registry_offline_loaders import (
    _load_cal_ql_algorithm as _load_cal_ql_algorithm,
)
from axiomrl.experiment.registry_offline_loaders import (
    _load_crr_algorithm as _load_crr_algorithm,
)
from axiomrl.experiment.registry_offline_loaders import (
    _load_rebrac_algorithm as _load_rebrac_algorithm,
)
from axiomrl.experiment.registry_offline_loaders import (
    _load_xql_algorithm as _load_xql_algorithm,
)
from axiomrl.experiment.registry_policy_loaders import (
    _load_a2c_algorithm as _load_a2c_algorithm,
)
from axiomrl.experiment.registry_policy_loaders import (
    _load_appo_algorithm as _load_appo_algorithm,
)
from axiomrl.experiment.registry_policy_loaders import (
    _load_ars_algorithm as _load_ars_algorithm,
)
from axiomrl.experiment.registry_policy_loaders import (
    _load_bc_algorithm as _load_bc_algorithm,
)
from axiomrl.experiment.registry_policy_loaders import (
    _load_decision_transformer_algorithm as _load_decision_transformer_algorithm,
)
from axiomrl.experiment.registry_policy_loaders import (
    _load_gail_algorithm as _load_gail_algorithm,
)
from axiomrl.experiment.registry_policy_loaders import (
    _load_impala_algorithm as _load_impala_algorithm,
)
from axiomrl.experiment.registry_policy_loaders import (
    _load_openai_es_algorithm as _load_openai_es_algorithm,
)
from axiomrl.experiment.registry_policy_loaders import (
    _load_ppg_algorithm as _load_ppg_algorithm,
)
from axiomrl.experiment.registry_policy_loaders import (
    _load_ppo_algorithm as _load_ppo_algorithm,
)
from axiomrl.experiment.registry_policy_loaders import (
    _load_trpo_algorithm as _load_trpo_algorithm,
)
from axiomrl.experiment.registry_recurrent_loaders import (
    _load_agent57_algorithm as _load_agent57_algorithm,
)
from axiomrl.experiment.registry_recurrent_loaders import (
    _load_drqn_algorithm as _load_drqn_algorithm,
)
from axiomrl.experiment.registry_recurrent_loaders import (
    _load_r2d2_algorithm as _load_r2d2_algorithm,
)
from axiomrl.experiment.registry_recurrent_loaders import (
    _load_recurrent_ppo_algorithm as _load_recurrent_ppo_algorithm,
)
from axiomrl.experiment.registry_specialized_loaders import (
    _load_discrete_sac_algorithm as _load_discrete_sac_algorithm,
)
from axiomrl.experiment.registry_specialized_loaders import (
    _load_dreamer_algorithm as _load_dreamer_algorithm,
)
from axiomrl.experiment.registry_specialized_loaders import (
    _load_efficientzero_algorithm as _load_efficientzero_algorithm,
)
from axiomrl.experiment.registry_specialized_loaders import (
    _load_gumbel_muzero_algorithm as _load_gumbel_muzero_algorithm,
)
from axiomrl.experiment.registry_specialized_loaders import (
    _load_her_algorithm as _load_her_algorithm,
)
from axiomrl.experiment.registry_specialized_loaders import (
    _load_mbpo_algorithm as _load_mbpo_algorithm,
)
from axiomrl.experiment.registry_specialized_loaders import (
    _load_mopo_algorithm as _load_mopo_algorithm,
)
from axiomrl.experiment.registry_specialized_loaders import (
    _load_muzero_algorithm as _load_muzero_algorithm,
)
from axiomrl.experiment.registry_specialized_loaders import (
    _load_pets_algorithm as _load_pets_algorithm,
)
from axiomrl.experiment.registry_support import (
    _format_action_output as _registry_format_action_output,
)
from axiomrl.experiment.registry_support import (
    _prepare_observation as _registry_prepare_observation,
)
from axiomrl.experiment.registry_types import AlgorithmSpec
from axiomrl.runtime.agent57_trainer import train_agent57
from axiomrl.runtime.apex_dqn_trainer import train_apex_dqn
from axiomrl.runtime.awac_trainer import train_awac
from axiomrl.runtime.awr_trainer import train_awr
from axiomrl.runtime.bc_trainer import train_bc
from axiomrl.runtime.bcq_trainer import train_bcq
from axiomrl.runtime.bear_trainer import train_bear
from axiomrl.runtime.cal_ql_trainer import train_cal_ql
from axiomrl.runtime.cql_trainer import train_cql
from axiomrl.runtime.crossq_trainer import train_crossq
from axiomrl.runtime.crr_trainer import train_crr
from axiomrl.runtime.curl_trainer import train_curl
from axiomrl.runtime.d4pg_trainer import train_d4pg
from axiomrl.runtime.ddpg_trainer import train_ddpg
from axiomrl.runtime.decision_transformer_trainer import (
    train_decision_transformer,
)
from axiomrl.runtime.discrete_sac_trainer import train_discrete_sac
from axiomrl.runtime.dqn_trainer import train_dqn
from axiomrl.runtime.dreamer_trainer import train_dreamer
from axiomrl.runtime.drq_trainer import train_drq
from axiomrl.runtime.drqn_trainer import train_drqn
from axiomrl.runtime.drqv2_trainer import train_drqv2
from axiomrl.runtime.edac_trainer import train_edac
from axiomrl.runtime.efficientzero_trainer import train_efficientzero
from axiomrl.runtime.her_trainer import train_her
from axiomrl.runtime.iql_trainer import train_iql
from axiomrl.runtime.marwil_trainer import train_marwil
from axiomrl.runtime.mbpo_trainer import train_mbpo
from axiomrl.runtime.mopo_trainer import train_mopo
from axiomrl.runtime.muzero_trainer import train_muzero
from axiomrl.runtime.naf_trainer import train_naf
from axiomrl.runtime.pets_trainer import train_pets
from axiomrl.runtime.r2d2_trainer import train_r2d2
from axiomrl.runtime.rebrac_trainer import train_rebrac
from axiomrl.runtime.recurrent_ppo_trainer import train_recurrent_ppo
from axiomrl.runtime.redq_trainer import train_redq
from axiomrl.runtime.rlpd_trainer import train_rlpd
from axiomrl.runtime.sac_trainer import train_sac
from axiomrl.runtime.td3_bc_trainer import train_td3_bc
from axiomrl.runtime.td3_trainer import train_td3
from axiomrl.runtime.tqc_trainer import train_tqc
from axiomrl.runtime.xql_trainer import train_xql

_format_action_output = _registry_format_action_output
_prepare_observation = _registry_prepare_observation

from axiomrl.experiment.registry_actor_critic_specs import ACTOR_CRITIC_SPECS
from axiomrl.experiment.registry_evaluators import (
    _evaluate_a2c,
    _evaluate_agent57,
    _evaluate_appo,
    _evaluate_ars,
    _evaluate_awac,
    _evaluate_awr,
    _evaluate_bc,
    _evaluate_bcq,
    _evaluate_bear,
    _evaluate_c51_dqn,
    _evaluate_cal_ql,
    _evaluate_cql,
    _evaluate_crossq,
    _evaluate_crr,
    _evaluate_curl,
    _evaluate_d4pg,
    _evaluate_ddpg,
    _evaluate_decision_transformer,
    _evaluate_diamond,
    _evaluate_discrete_sac,
    _evaluate_dqn,
    _evaluate_dreamer,
    _evaluate_dreamerv3,
    _evaluate_drq,
    _evaluate_drqn,
    _evaluate_drqv2,
    _evaluate_eadream,
    _evaluate_edac,
    _evaluate_efficientzero,
    _evaluate_fqf,
    _evaluate_gail,
    _evaluate_gumbel_muzero,
    _evaluate_her,
    _evaluate_horizon_imagination,
    _evaluate_impala,
    _evaluate_iql,
    _evaluate_iqn,
    _evaluate_marwil,
    _evaluate_mbpo,
    _evaluate_mopo,
    _evaluate_mow,
    _evaluate_muzero,
    _evaluate_naf,
    _evaluate_openai_es,
    _evaluate_pets,
    _evaluate_po_dreamer,
    _evaluate_ppg,
    _evaluate_ppo,
    _evaluate_qr_dqn,
    _evaluate_r2d2,
    _evaluate_rebrac,
    _evaluate_recurrent_ppo,
    _evaluate_redq,
    _evaluate_rlpd,
    _evaluate_sac,
    _evaluate_scalezero,
    _evaluate_td3,
    _evaluate_td3_bc,
    _evaluate_tqc,
    _evaluate_trpo,
    _evaluate_twisted,
    _evaluate_xql,
)
from axiomrl.experiment.registry_offline_specs import OFFLINE_SPECS
from axiomrl.experiment.registry_on_policy_specs import ON_POLICY_SPECS
from axiomrl.experiment.registry_predictors import (
    _predict_a2c,
    _predict_agent57,
    _predict_appo,
    _predict_ars,
    _predict_awac,
    _predict_awr,
    _predict_bc,
    _predict_bcq,
    _predict_bear,
    _predict_c51_dqn,
    _predict_cal_ql,
    _predict_cql,
    _predict_crossq,
    _predict_crr,
    _predict_curl,
    _predict_d4pg,
    _predict_ddpg,
    _predict_decision_transformer,
    _predict_diamond,
    _predict_discrete_sac,
    _predict_dqn,
    _predict_dreamer,
    _predict_dreamerv3,
    _predict_drq,
    _predict_drqn,
    _predict_drqv2,
    _predict_eadream,
    _predict_edac,
    _predict_efficientzero,
    _predict_fqf,
    _predict_gail,
    _predict_gumbel_muzero,
    _predict_her,
    _predict_horizon_imagination,
    _predict_impala,
    _predict_iql,
    _predict_iqn,
    _predict_marwil,
    _predict_mbpo,
    _predict_mopo,
    _predict_mow,
    _predict_muzero,
    _predict_naf,
    _predict_openai_es,
    _predict_pets,
    _predict_po_dreamer,
    _predict_ppg,
    _predict_ppo,
    _predict_qr_dqn,
    _predict_r2d2,
    _predict_rebrac,
    _predict_recurrent_ppo,
    _predict_redq,
    _predict_rlpd,
    _predict_sac,
    _predict_scalezero,
    _predict_td3,
    _predict_td3_bc,
    _predict_tqc,
    _predict_trpo,
    _predict_twisted,
    _predict_xql,
)
from axiomrl.experiment.registry_value_based_specs import VALUE_BASED_SPECS
from axiomrl.experiment.registry_world_model_specs import WORLD_MODEL_SPECS

# Preserve registry_core attribute compatibility for on-policy evaluator/predictor helpers
# that are now consumed through the split ON_POLICY_SPECS module.
_ON_POLICY_EVALUATOR_REEXPORTS = (
    _evaluate_a2c,
    _evaluate_appo,
    _evaluate_ars,
    _evaluate_gail,
    _evaluate_impala,
    _evaluate_openai_es,
    _evaluate_ppg,
    _evaluate_ppo,
    _evaluate_trpo,
)
_ON_POLICY_PREDICTOR_REEXPORTS = (
    _predict_a2c,
    _predict_appo,
    _predict_ars,
    _predict_gail,
    _predict_impala,
    _predict_openai_es,
    _predict_ppg,
    _predict_ppo,
    _predict_trpo,
)
del _ON_POLICY_EVALUATOR_REEXPORTS
del _ON_POLICY_PREDICTOR_REEXPORTS

# Preserve registry_core attribute compatibility for offline trainer/evaluator/predictor
# helpers that are now consumed through the split OFFLINE_SPECS module.
_OFFLINE_TRAIN_FN_REEXPORTS = (
    train_awac,
    train_awr,
    train_bc,
    train_bcq,
    train_bear,
    train_cal_ql,
    train_cql,
    train_crr,
    train_decision_transformer,
    train_iql,
    train_marwil,
    train_rebrac,
    train_xql,
)
_OFFLINE_EVALUATOR_REEXPORTS = (
    _evaluate_awac,
    _evaluate_awr,
    _evaluate_bc,
    _evaluate_bcq,
    _evaluate_bear,
    _evaluate_cal_ql,
    _evaluate_cql,
    _evaluate_crr,
    _evaluate_decision_transformer,
    _evaluate_iql,
    _evaluate_marwil,
    _evaluate_rebrac,
    _evaluate_xql,
)
_OFFLINE_PREDICTOR_REEXPORTS = (
    _predict_awac,
    _predict_awr,
    _predict_bc,
    _predict_bcq,
    _predict_bear,
    _predict_cal_ql,
    _predict_cql,
    _predict_crr,
    _predict_decision_transformer,
    _predict_iql,
    _predict_marwil,
    _predict_rebrac,
    _predict_xql,
)
del _OFFLINE_TRAIN_FN_REEXPORTS
del _OFFLINE_EVALUATOR_REEXPORTS
del _OFFLINE_PREDICTOR_REEXPORTS

# Preserve registry_core attribute compatibility for world-model trainer/evaluator/predictor
# helpers that are now consumed through the split WORLD_MODEL_SPECS module.
_WORLD_MODEL_TRAIN_FN_REEXPORTS = (
    train_dreamer,
    train_efficientzero,
    train_mbpo,
    train_mopo,
    train_muzero,
    train_pets,
)
_WORLD_MODEL_EVALUATOR_REEXPORTS = (
    _evaluate_diamond,
    _evaluate_dreamer,
    _evaluate_dreamerv3,
    _evaluate_eadream,
    _evaluate_efficientzero,
    _evaluate_gumbel_muzero,
    _evaluate_horizon_imagination,
    _evaluate_mbpo,
    _evaluate_mopo,
    _evaluate_mow,
    _evaluate_muzero,
    _evaluate_pets,
    _evaluate_po_dreamer,
    _evaluate_scalezero,
    _evaluate_twisted,
)
_WORLD_MODEL_PREDICTOR_REEXPORTS = (
    _predict_diamond,
    _predict_dreamer,
    _predict_dreamerv3,
    _predict_eadream,
    _predict_efficientzero,
    _predict_gumbel_muzero,
    _predict_horizon_imagination,
    _predict_mbpo,
    _predict_mopo,
    _predict_mow,
    _predict_muzero,
    _predict_pets,
    _predict_po_dreamer,
    _predict_scalezero,
    _predict_twisted,
)
del _WORLD_MODEL_TRAIN_FN_REEXPORTS
del _WORLD_MODEL_EVALUATOR_REEXPORTS
del _WORLD_MODEL_PREDICTOR_REEXPORTS

# Preserve registry_core attribute compatibility for actor-critic trainer/evaluator/predictor
# helpers that are now consumed through the split ACTOR_CRITIC_SPECS module.
_ACTOR_CRITIC_TRAIN_FN_REEXPORTS = (
    train_crossq,
    train_curl,
    train_d4pg,
    train_ddpg,
    train_discrete_sac,
    train_drq,
    train_drqv2,
    train_edac,
    train_naf,
    train_redq,
    train_rlpd,
    train_sac,
    train_td3,
    train_td3_bc,
    train_tqc,
)
_ACTOR_CRITIC_EVALUATOR_REEXPORTS = (
    _evaluate_crossq,
    _evaluate_curl,
    _evaluate_d4pg,
    _evaluate_ddpg,
    _evaluate_discrete_sac,
    _evaluate_drq,
    _evaluate_drqv2,
    _evaluate_edac,
    _evaluate_naf,
    _evaluate_redq,
    _evaluate_rlpd,
    _evaluate_sac,
    _evaluate_td3,
    _evaluate_td3_bc,
    _evaluate_tqc,
)
_ACTOR_CRITIC_PREDICTOR_REEXPORTS = (
    _predict_crossq,
    _predict_curl,
    _predict_d4pg,
    _predict_ddpg,
    _predict_discrete_sac,
    _predict_drq,
    _predict_drqv2,
    _predict_edac,
    _predict_naf,
    _predict_redq,
    _predict_rlpd,
    _predict_sac,
    _predict_td3,
    _predict_td3_bc,
    _predict_tqc,
)
del _ACTOR_CRITIC_TRAIN_FN_REEXPORTS
del _ACTOR_CRITIC_EVALUATOR_REEXPORTS
del _ACTOR_CRITIC_PREDICTOR_REEXPORTS

# Preserve registry_core attribute compatibility for value-based trainer/evaluator/
# predictor helpers that are now consumed through the split VALUE_BASED_SPECS
# module.
_VALUE_BASED_TRAIN_FN_REEXPORTS = (
    train_agent57,
    train_apex_dqn,
    train_dqn,
    train_drqn,
    train_r2d2,
)
_VALUE_BASED_EVALUATOR_REEXPORTS = (
    _evaluate_agent57,
    _evaluate_c51_dqn,
    _evaluate_dqn,
    _evaluate_drqn,
    _evaluate_fqf,
    _evaluate_iqn,
    _evaluate_qr_dqn,
    _evaluate_r2d2,
)
_VALUE_BASED_PREDICTOR_REEXPORTS = (
    _predict_agent57,
    _predict_c51_dqn,
    _predict_dqn,
    _predict_drqn,
    _predict_fqf,
    _predict_iqn,
    _predict_qr_dqn,
    _predict_r2d2,
)
del _VALUE_BASED_TRAIN_FN_REEXPORTS
del _VALUE_BASED_EVALUATOR_REEXPORTS
del _VALUE_BASED_PREDICTOR_REEXPORTS

_ALGORITHM_REGISTRY: dict[str, AlgorithmSpec] = {
    "a2c": ON_POLICY_SPECS["a2c"],
    "ars": ON_POLICY_SPECS["ars"],
    "openai_es": ON_POLICY_SPECS["openai_es"],
    "impala": ON_POLICY_SPECS["impala"],
    "appo": ON_POLICY_SPECS["appo"],
    "awac": OFFLINE_SPECS["awac"],
    "crr": OFFLINE_SPECS["crr"],
    "bc": OFFLINE_SPECS["bc"],
    "decision_transformer": OFFLINE_SPECS["decision_transformer"],
    "mopo": WORLD_MODEL_SPECS["mopo"],
    "mbpo": WORLD_MODEL_SPECS["mbpo"],
    "pets": WORLD_MODEL_SPECS["pets"],
    "bcq": OFFLINE_SPECS["bcq"],
    "bear": OFFLINE_SPECS["bear"],
    "her": AlgorithmSpec(
        name="her",
        train_fn=train_her,
        evaluate_fn=_evaluate_her,
        predict_fn=_predict_her,
    ),
    "ppo": ON_POLICY_SPECS["ppo"],
    "gail": ON_POLICY_SPECS["gail"],
    "dreamer": WORLD_MODEL_SPECS["dreamer"],
    "dreamerv3": WORLD_MODEL_SPECS["dreamerv3"],
    "diamond": WORLD_MODEL_SPECS["diamond"],
    "horizon_imagination": WORLD_MODEL_SPECS["horizon_imagination"],
    "po_dreamer": WORLD_MODEL_SPECS["po_dreamer"],
    "twisted": WORLD_MODEL_SPECS["twisted"],
    "mow": WORLD_MODEL_SPECS["mow"],
    "eadream": WORLD_MODEL_SPECS["eadream"],
    "muzero": WORLD_MODEL_SPECS["muzero"],
    "gumbel_muzero": WORLD_MODEL_SPECS["gumbel_muzero"],
    "efficientzero": WORLD_MODEL_SPECS["efficientzero"],
    "scalezero": WORLD_MODEL_SPECS["scalezero"],
    "trpo": ON_POLICY_SPECS["trpo"],
    "recurrent_ppo": AlgorithmSpec(
        name="recurrent_ppo",
        train_fn=train_recurrent_ppo,
        evaluate_fn=_evaluate_recurrent_ppo,
        predict_fn=_predict_recurrent_ppo,
    ),
    "dqn": VALUE_BASED_SPECS["dqn"],
    "jowa": VALUE_BASED_SPECS["jowa"],
    "spr": VALUE_BASED_SPECS["spr"],
    "apex_dqn": VALUE_BASED_SPECS["apex_dqn"],
    "c51_dqn": VALUE_BASED_SPECS["c51_dqn"],
    "n_step_dqn": VALUE_BASED_SPECS["n_step_dqn"],
    "expected_sarsa": VALUE_BASED_SPECS["expected_sarsa"],
    "expected_double_dqn": VALUE_BASED_SPECS["expected_double_dqn"],
    "boltzmann_dqn": VALUE_BASED_SPECS["boltzmann_dqn"],
    "boltzmann_double_dqn": VALUE_BASED_SPECS["boltzmann_double_dqn"],
    "mellowmax_dqn": VALUE_BASED_SPECS["mellowmax_dqn"],
    "soft_dqn": VALUE_BASED_SPECS["soft_dqn"],
    "soft_double_dqn": VALUE_BASED_SPECS["soft_double_dqn"],
    "advantage_learning_dqn": VALUE_BASED_SPECS["advantage_learning_dqn"],
    "persistent_advantage_learning_dqn": VALUE_BASED_SPECS["persistent_advantage_learning_dqn"],
    "munchausen_dqn": VALUE_BASED_SPECS["munchausen_dqn"],
    "munchausen_double_dqn": VALUE_BASED_SPECS["munchausen_double_dqn"],
    "cql_dqn": VALUE_BASED_SPECS["cql_dqn"],
    "cql_double_dqn": VALUE_BASED_SPECS["cql_double_dqn"],
    "clipped_double_dqn": VALUE_BASED_SPECS["clipped_double_dqn"],
    "hysteretic_dqn": VALUE_BASED_SPECS["hysteretic_dqn"],
    "noisy_dqn": VALUE_BASED_SPECS["noisy_dqn"],
    "prioritized_dqn": VALUE_BASED_SPECS["prioritized_dqn"],
    "rainbow_dqn": VALUE_BASED_SPECS["rainbow_dqn"],
    "qr_dqn": VALUE_BASED_SPECS["qr_dqn"],
    "iqn": VALUE_BASED_SPECS["iqn"],
    "fqf": VALUE_BASED_SPECS["fqf"],
    "cal_ql": OFFLINE_SPECS["cal_ql"],
    "awr": OFFLINE_SPECS["awr"],
    "marwil": OFFLINE_SPECS["marwil"],
    "iql": OFFLINE_SPECS["iql"],
    "xql": OFFLINE_SPECS["xql"],
    "ddpg": ACTOR_CRITIC_SPECS["ddpg"],
    "naf": ACTOR_CRITIC_SPECS["naf"],
    "d4pg": ACTOR_CRITIC_SPECS["d4pg"],
    "drqn": VALUE_BASED_SPECS["drqn"],
    "r2d2": VALUE_BASED_SPECS["r2d2"],
    "agent57": VALUE_BASED_SPECS["agent57"],
    "drq": ACTOR_CRITIC_SPECS["drq"],
    "curl": ACTOR_CRITIC_SPECS["curl"],
    "ppg": ON_POLICY_SPECS["ppg"],
    "drqv2": ACTOR_CRITIC_SPECS["drqv2"],
    "double_dqn": VALUE_BASED_SPECS["double_dqn"],
    "dueling_dqn": VALUE_BASED_SPECS["dueling_dqn"],
    "sac": ACTOR_CRITIC_SPECS["sac"],
    "rlpd": ACTOR_CRITIC_SPECS["rlpd"],
    "cql": OFFLINE_SPECS["cql"],
    "crossq": ACTOR_CRITIC_SPECS["crossq"],
    "discrete_sac": ACTOR_CRITIC_SPECS["discrete_sac"],
    "tqc": ACTOR_CRITIC_SPECS["tqc"],
    "redq": ACTOR_CRITIC_SPECS["redq"],
    "edac": ACTOR_CRITIC_SPECS["edac"],
    "td3": ACTOR_CRITIC_SPECS["td3"],
    "td3_bc": ACTOR_CRITIC_SPECS["td3_bc"],
    "rebrac": OFFLINE_SPECS["rebrac"],
}


def get_algorithm_spec(name: str) -> AlgorithmSpec:
    try:
        return _ALGORITHM_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"unknown algorithm: {name!r}") from exc


def list_algorithm_specs() -> tuple[AlgorithmSpec, ...]:
    return tuple(_ALGORITHM_REGISTRY.values())
