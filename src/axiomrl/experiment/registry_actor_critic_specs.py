from axiomrl.experiment.registry_evaluators import (
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
from axiomrl.experiment.registry_predictors import (
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
from axiomrl.experiment.registry_types import AlgorithmSpec
from axiomrl.runtime.crossq_trainer import train_crossq
from axiomrl.runtime.curl_trainer import train_curl
from axiomrl.runtime.d4pg_trainer import train_d4pg
from axiomrl.runtime.ddpg_trainer import train_ddpg
from axiomrl.runtime.discrete_sac_trainer import train_discrete_sac
from axiomrl.runtime.drq_trainer import train_drq
from axiomrl.runtime.drqv2_trainer import train_drqv2
from axiomrl.runtime.edac_trainer import train_edac
from axiomrl.runtime.naf_trainer import train_naf
from axiomrl.runtime.redq_trainer import train_redq
from axiomrl.runtime.rlpd_trainer import train_rlpd
from axiomrl.runtime.sac_trainer import train_sac
from axiomrl.runtime.td3_bc_trainer import train_td3_bc
from axiomrl.runtime.td3_trainer import train_td3
from axiomrl.runtime.tqc_trainer import train_tqc

ACTOR_CRITIC_SPECS: dict[str, AlgorithmSpec] = {
    "sac": AlgorithmSpec(
        name="sac",
        train_fn=train_sac,
        evaluate_fn=_evaluate_sac,
        predict_fn=_predict_sac,
    ),
    "rlpd": AlgorithmSpec(
        name="rlpd",
        train_fn=train_rlpd,
        evaluate_fn=_evaluate_rlpd,
        predict_fn=_predict_rlpd,
    ),
    "crossq": AlgorithmSpec(
        name="crossq",
        train_fn=train_crossq,
        evaluate_fn=_evaluate_crossq,
        predict_fn=_predict_crossq,
    ),
    "discrete_sac": AlgorithmSpec(
        name="discrete_sac",
        train_fn=train_discrete_sac,
        evaluate_fn=_evaluate_discrete_sac,
        predict_fn=_predict_discrete_sac,
    ),
    "tqc": AlgorithmSpec(
        name="tqc",
        train_fn=train_tqc,
        evaluate_fn=_evaluate_tqc,
        predict_fn=_predict_tqc,
    ),
    "redq": AlgorithmSpec(
        name="redq",
        train_fn=train_redq,
        evaluate_fn=_evaluate_redq,
        predict_fn=_predict_redq,
    ),
    "edac": AlgorithmSpec(
        name="edac",
        train_fn=train_edac,
        evaluate_fn=_evaluate_edac,
        predict_fn=_predict_edac,
    ),
    "ddpg": AlgorithmSpec(
        name="ddpg",
        train_fn=train_ddpg,
        evaluate_fn=_evaluate_ddpg,
        predict_fn=_predict_ddpg,
    ),
    "naf": AlgorithmSpec(
        name="naf",
        train_fn=train_naf,
        evaluate_fn=_evaluate_naf,
        predict_fn=_predict_naf,
    ),
    "d4pg": AlgorithmSpec(
        name="d4pg",
        train_fn=train_d4pg,
        evaluate_fn=_evaluate_d4pg,
        predict_fn=_predict_d4pg,
    ),
    "drq": AlgorithmSpec(
        name="drq",
        train_fn=train_drq,
        evaluate_fn=_evaluate_drq,
        predict_fn=_predict_drq,
    ),
    "curl": AlgorithmSpec(
        name="curl",
        train_fn=train_curl,
        evaluate_fn=_evaluate_curl,
        predict_fn=_predict_curl,
    ),
    "drqv2": AlgorithmSpec(
        name="drqv2",
        train_fn=train_drqv2,
        evaluate_fn=_evaluate_drqv2,
        predict_fn=_predict_drqv2,
    ),
    "td3": AlgorithmSpec(
        name="td3",
        train_fn=train_td3,
        evaluate_fn=_evaluate_td3,
        predict_fn=_predict_td3,
    ),
    "td3_bc": AlgorithmSpec(
        name="td3_bc",
        train_fn=train_td3_bc,
        evaluate_fn=_evaluate_td3_bc,
        predict_fn=_predict_td3_bc,
    ),
}
