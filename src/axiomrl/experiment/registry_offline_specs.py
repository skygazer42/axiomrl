from __future__ import annotations

from axiomrl.experiment.registry_evaluators import (
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
from axiomrl.experiment.registry_predictors import (
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
from axiomrl.experiment.registry_types import AlgorithmSpec
from axiomrl.runtime.awac_trainer import train_awac
from axiomrl.runtime.awr_trainer import train_awr
from axiomrl.runtime.bc_trainer import train_bc
from axiomrl.runtime.bcq_trainer import train_bcq
from axiomrl.runtime.bear_trainer import train_bear
from axiomrl.runtime.cal_ql_trainer import train_cal_ql
from axiomrl.runtime.cql_trainer import train_cql
from axiomrl.runtime.crr_trainer import train_crr
from axiomrl.runtime.decision_transformer_trainer import train_decision_transformer
from axiomrl.runtime.iql_trainer import train_iql
from axiomrl.runtime.marwil_trainer import train_marwil
from axiomrl.runtime.rebrac_trainer import train_rebrac
from axiomrl.runtime.xql_trainer import train_xql

OFFLINE_SPECS: dict[str, AlgorithmSpec] = {
    "bc": AlgorithmSpec(
        name="bc",
        train_fn=train_bc,
        evaluate_fn=_evaluate_bc,
        predict_fn=_predict_bc,
    ),
    "decision_transformer": AlgorithmSpec(
        name="decision_transformer",
        train_fn=train_decision_transformer,
        evaluate_fn=_evaluate_decision_transformer,
        predict_fn=_predict_decision_transformer,
    ),
    "bcq": AlgorithmSpec(
        name="bcq",
        train_fn=train_bcq,
        evaluate_fn=_evaluate_bcq,
        predict_fn=_predict_bcq,
    ),
    "bear": AlgorithmSpec(
        name="bear",
        train_fn=train_bear,
        evaluate_fn=_evaluate_bear,
        predict_fn=_predict_bear,
    ),
    "awac": AlgorithmSpec(
        name="awac",
        train_fn=train_awac,
        evaluate_fn=_evaluate_awac,
        predict_fn=_predict_awac,
    ),
    "crr": AlgorithmSpec(
        name="crr",
        train_fn=train_crr,
        evaluate_fn=_evaluate_crr,
        predict_fn=_predict_crr,
    ),
    "cal_ql": AlgorithmSpec(
        name="cal_ql",
        train_fn=train_cal_ql,
        evaluate_fn=_evaluate_cal_ql,
        predict_fn=_predict_cal_ql,
    ),
    "xql": AlgorithmSpec(
        name="xql",
        train_fn=train_xql,
        evaluate_fn=_evaluate_xql,
        predict_fn=_predict_xql,
    ),
    "iql": AlgorithmSpec(
        name="iql",
        train_fn=train_iql,
        evaluate_fn=_evaluate_iql,
        predict_fn=_predict_iql,
    ),
    "awr": AlgorithmSpec(
        name="awr",
        train_fn=train_awr,
        evaluate_fn=_evaluate_awr,
        predict_fn=_predict_awr,
    ),
    "marwil": AlgorithmSpec(
        name="marwil",
        train_fn=train_marwil,
        evaluate_fn=_evaluate_marwil,
        predict_fn=_predict_marwil,
    ),
    "cql": AlgorithmSpec(
        name="cql",
        train_fn=train_cql,
        evaluate_fn=_evaluate_cql,
        predict_fn=_predict_cql,
    ),
    "rebrac": AlgorithmSpec(
        name="rebrac",
        train_fn=train_rebrac,
        evaluate_fn=_evaluate_rebrac,
        predict_fn=_predict_rebrac,
    ),
}
