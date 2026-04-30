from __future__ import annotations

from axiomrl.experiment.registry_evaluators import (
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
from axiomrl.experiment.registry_predictors import (
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
from axiomrl.experiment.registry_types import AlgorithmSpec
from axiomrl.runtime.a2c_trainer import train_a2c
from axiomrl.runtime.appo_trainer import train_appo
from axiomrl.runtime.ars_trainer import train_ars
from axiomrl.runtime.gail_trainer import train_gail
from axiomrl.runtime.impala_trainer import train_impala
from axiomrl.runtime.openai_es_trainer import train_openai_es
from axiomrl.runtime.ppg_trainer import train_ppg
from axiomrl.runtime.ppo_trainer import train_ppo
from axiomrl.runtime.trpo_trainer import train_trpo

ON_POLICY_SPECS: dict[str, AlgorithmSpec] = {
    "a2c": AlgorithmSpec(
        name="a2c",
        train_fn=train_a2c,
        evaluate_fn=_evaluate_a2c,
        predict_fn=_predict_a2c,
    ),
    "ars": AlgorithmSpec(
        name="ars",
        train_fn=train_ars,
        evaluate_fn=_evaluate_ars,
        predict_fn=_predict_ars,
    ),
    "openai_es": AlgorithmSpec(
        name="openai_es",
        train_fn=train_openai_es,
        evaluate_fn=_evaluate_openai_es,
        predict_fn=_predict_openai_es,
    ),
    "impala": AlgorithmSpec(
        name="impala",
        train_fn=train_impala,
        evaluate_fn=_evaluate_impala,
        predict_fn=_predict_impala,
    ),
    "appo": AlgorithmSpec(
        name="appo",
        train_fn=train_appo,
        evaluate_fn=_evaluate_appo,
        predict_fn=_predict_appo,
    ),
    "ppo": AlgorithmSpec(
        name="ppo",
        train_fn=train_ppo,
        evaluate_fn=_evaluate_ppo,
        predict_fn=_predict_ppo,
    ),
    "gail": AlgorithmSpec(
        name="gail",
        train_fn=train_gail,
        evaluate_fn=_evaluate_gail,
        predict_fn=_predict_gail,
    ),
    "ppg": AlgorithmSpec(
        name="ppg",
        train_fn=train_ppg,
        evaluate_fn=_evaluate_ppg,
        predict_fn=_predict_ppg,
    ),
    "trpo": AlgorithmSpec(
        name="trpo",
        train_fn=train_trpo,
        evaluate_fn=_evaluate_trpo,
        predict_fn=_predict_trpo,
    ),
}
