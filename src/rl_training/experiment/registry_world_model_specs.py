from __future__ import annotations

from rl_training.experiment.registry_evaluators import (
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
from rl_training.experiment.registry_predictors import (
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
from rl_training.experiment.registry_types import AlgorithmSpec
from rl_training.runtime.dreamer_trainer import train_dreamer
from rl_training.runtime.efficientzero_trainer import train_efficientzero
from rl_training.runtime.mbpo_trainer import train_mbpo
from rl_training.runtime.mopo_trainer import train_mopo
from rl_training.runtime.muzero_trainer import train_muzero
from rl_training.runtime.pets_trainer import train_pets

WORLD_MODEL_SPECS: dict[str, AlgorithmSpec] = {
    "dreamer": AlgorithmSpec(
        name="dreamer",
        train_fn=train_dreamer,
        evaluate_fn=_evaluate_dreamer,
        predict_fn=_predict_dreamer,
    ),
    "dreamerv3": AlgorithmSpec(
        name="dreamerv3",
        train_fn=train_dreamer,
        evaluate_fn=_evaluate_dreamerv3,
        predict_fn=_predict_dreamerv3,
    ),
    "diamond": AlgorithmSpec(
        name="diamond",
        train_fn=train_dreamer,
        evaluate_fn=_evaluate_diamond,
        predict_fn=_predict_diamond,
    ),
    "horizon_imagination": AlgorithmSpec(
        name="horizon_imagination",
        train_fn=train_dreamer,
        evaluate_fn=_evaluate_horizon_imagination,
        predict_fn=_predict_horizon_imagination,
    ),
    "po_dreamer": AlgorithmSpec(
        name="po_dreamer",
        train_fn=train_dreamer,
        evaluate_fn=_evaluate_po_dreamer,
        predict_fn=_predict_po_dreamer,
    ),
    "twisted": AlgorithmSpec(
        name="twisted",
        train_fn=train_dreamer,
        evaluate_fn=_evaluate_twisted,
        predict_fn=_predict_twisted,
    ),
    "mow": AlgorithmSpec(
        name="mow",
        train_fn=train_dreamer,
        evaluate_fn=_evaluate_mow,
        predict_fn=_predict_mow,
    ),
    "eadream": AlgorithmSpec(
        name="eadream",
        train_fn=train_dreamer,
        evaluate_fn=_evaluate_eadream,
        predict_fn=_predict_eadream,
    ),
    "muzero": AlgorithmSpec(
        name="muzero",
        train_fn=train_muzero,
        evaluate_fn=_evaluate_muzero,
        predict_fn=_predict_muzero,
    ),
    "gumbel_muzero": AlgorithmSpec(
        name="gumbel_muzero",
        train_fn=train_muzero,
        evaluate_fn=_evaluate_gumbel_muzero,
        predict_fn=_predict_gumbel_muzero,
    ),
    "efficientzero": AlgorithmSpec(
        name="efficientzero",
        train_fn=train_efficientzero,
        evaluate_fn=_evaluate_efficientzero,
        predict_fn=_predict_efficientzero,
    ),
    "scalezero": AlgorithmSpec(
        name="scalezero",
        train_fn=train_muzero,
        evaluate_fn=_evaluate_scalezero,
        predict_fn=_predict_scalezero,
    ),
    "mopo": AlgorithmSpec(
        name="mopo",
        train_fn=train_mopo,
        evaluate_fn=_evaluate_mopo,
        predict_fn=_predict_mopo,
    ),
    "mbpo": AlgorithmSpec(
        name="mbpo",
        train_fn=train_mbpo,
        evaluate_fn=_evaluate_mbpo,
        predict_fn=_predict_mbpo,
    ),
    "pets": AlgorithmSpec(
        name="pets",
        train_fn=train_pets,
        evaluate_fn=_evaluate_pets,
        predict_fn=_predict_pets,
    ),
}
