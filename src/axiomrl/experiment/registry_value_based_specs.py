from __future__ import annotations

from axiomrl.experiment.registry_evaluators import (
    _evaluate_agent57,
    _evaluate_c51_dqn,
    _evaluate_dqn,
    _evaluate_drqn,
    _evaluate_fqf,
    _evaluate_iqn,
    _evaluate_qr_dqn,
    _evaluate_r2d2,
)
from axiomrl.experiment.registry_predictors import (
    _predict_agent57,
    _predict_c51_dqn,
    _predict_dqn,
    _predict_drqn,
    _predict_fqf,
    _predict_iqn,
    _predict_qr_dqn,
    _predict_r2d2,
)
from axiomrl.experiment.registry_types import AlgorithmSpec
from axiomrl.runtime.agent57_trainer import train_agent57
from axiomrl.runtime.apex_dqn_trainer import train_apex_dqn
from axiomrl.runtime.dqn_trainer import train_dqn
from axiomrl.runtime.drqn_trainer import train_drqn
from axiomrl.runtime.r2d2_trainer import train_r2d2


def _standard_dqn_spec(name: str) -> AlgorithmSpec:
    return AlgorithmSpec(
        name=name,
        train_fn=train_dqn,
        evaluate_fn=_evaluate_dqn,
        predict_fn=_predict_dqn,
    )


VALUE_BASED_SPECS: dict[str, AlgorithmSpec] = {
    "dqn": _standard_dqn_spec("dqn"),
    "jowa": _standard_dqn_spec("jowa"),
    "spr": _standard_dqn_spec("spr"),
    "apex_dqn": AlgorithmSpec(
        name="apex_dqn",
        train_fn=train_apex_dqn,
        evaluate_fn=_evaluate_dqn,
        predict_fn=_predict_dqn,
    ),
    "c51_dqn": AlgorithmSpec(
        name="c51_dqn",
        train_fn=train_dqn,
        evaluate_fn=_evaluate_c51_dqn,
        predict_fn=_predict_c51_dqn,
    ),
    "n_step_dqn": _standard_dqn_spec("n_step_dqn"),
    "expected_sarsa": _standard_dqn_spec("expected_sarsa"),
    "expected_double_dqn": _standard_dqn_spec("expected_double_dqn"),
    "boltzmann_dqn": _standard_dqn_spec("boltzmann_dqn"),
    "boltzmann_double_dqn": _standard_dqn_spec("boltzmann_double_dqn"),
    "mellowmax_dqn": _standard_dqn_spec("mellowmax_dqn"),
    "soft_dqn": _standard_dqn_spec("soft_dqn"),
    "soft_double_dqn": _standard_dqn_spec("soft_double_dqn"),
    "advantage_learning_dqn": _standard_dqn_spec("advantage_learning_dqn"),
    "persistent_advantage_learning_dqn": _standard_dqn_spec("persistent_advantage_learning_dqn"),
    "munchausen_dqn": _standard_dqn_spec("munchausen_dqn"),
    "munchausen_double_dqn": _standard_dqn_spec("munchausen_double_dqn"),
    "cql_dqn": _standard_dqn_spec("cql_dqn"),
    "cql_double_dqn": _standard_dqn_spec("cql_double_dqn"),
    "clipped_double_dqn": _standard_dqn_spec("clipped_double_dqn"),
    "hysteretic_dqn": _standard_dqn_spec("hysteretic_dqn"),
    "noisy_dqn": _standard_dqn_spec("noisy_dqn"),
    "prioritized_dqn": _standard_dqn_spec("prioritized_dqn"),
    "rainbow_dqn": _standard_dqn_spec("rainbow_dqn"),
    "qr_dqn": AlgorithmSpec(
        name="qr_dqn",
        train_fn=train_dqn,
        evaluate_fn=_evaluate_qr_dqn,
        predict_fn=_predict_qr_dqn,
    ),
    "iqn": AlgorithmSpec(
        name="iqn",
        train_fn=train_dqn,
        evaluate_fn=_evaluate_iqn,
        predict_fn=_predict_iqn,
    ),
    "fqf": AlgorithmSpec(
        name="fqf",
        train_fn=train_dqn,
        evaluate_fn=_evaluate_fqf,
        predict_fn=_predict_fqf,
    ),
    "double_dqn": _standard_dqn_spec("double_dqn"),
    "dueling_dqn": _standard_dqn_spec("dueling_dqn"),
    "drqn": AlgorithmSpec(
        name="drqn",
        train_fn=train_drqn,
        evaluate_fn=_evaluate_drqn,
        predict_fn=_predict_drqn,
    ),
    "r2d2": AlgorithmSpec(
        name="r2d2",
        train_fn=train_r2d2,
        evaluate_fn=_evaluate_r2d2,
        predict_fn=_predict_r2d2,
    ),
    "agent57": AlgorithmSpec(
        name="agent57",
        train_fn=train_agent57,
        evaluate_fn=_evaluate_agent57,
        predict_fn=_predict_agent57,
    ),
}
