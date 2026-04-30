from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import sqrt

from axiomrl.runtime.types import MetricDict

_HUMAN_RANDOM_SCORE_REFERENCES: dict[str, tuple[float, float]] = {
    "atari_breakout_reference": (1.7, 30.5),
    "breakout": (1.7, 30.5),
}


@dataclass(frozen=True, slots=True)
class ScoreNormalizationConfig:
    strategy: str
    random_score: float
    human_score: float
    scale: float = 100.0


@dataclass(frozen=True, slots=True)
class BestCheckpointConfig:
    metric_name: str = "eval_return_mean"
    metric_mode: str = "max"


def resolve_score_normalization_settings(requested: Mapping[str, object]) -> dict[str, object]:
    resolved = dict(requested)
    if "random_score" in resolved and "human_score" in resolved:
        return resolved

    source = str(resolved.get("source", "")).strip().lower()
    game = str(resolved.get("game", "")).strip().lower()
    reference_key = source or game
    if not reference_key:
        raise KeyError("score normalization requires 'random_score'/'human_score' or a named 'source'/'game' reference")

    reference = _HUMAN_RANDOM_SCORE_REFERENCES.get(reference_key)
    if reference is None:
        raise KeyError(f"unknown score normalization reference {reference_key!r}")

    random_score, human_score = reference
    resolved.setdefault("random_score", random_score)
    resolved.setdefault("human_score", human_score)
    return resolved


def resolve_score_normalization_config(benchmark: Mapping[str, object]) -> ScoreNormalizationConfig | None:
    requested = benchmark.get("score_normalization")
    if requested in (None, False):
        return None
    if not isinstance(requested, Mapping):
        raise TypeError(f"expected benchmark['score_normalization'] to be a mapping, got {type(requested)!r}")

    resolved = resolve_score_normalization_settings(requested)

    strategy = str(resolved.get("type", "human_random")).strip().lower()
    if strategy != "human_random":
        raise ValueError(f"unsupported score normalization strategy {strategy!r}; expected 'human_random'")

    random_score = float(resolved["random_score"])
    human_score = float(resolved["human_score"])
    if human_score == random_score:
        raise ValueError("benchmark['score_normalization'] requires human_score != random_score")

    return ScoreNormalizationConfig(
        strategy=strategy,
        random_score=random_score,
        human_score=human_score,
        scale=float(resolved.get("scale", 100.0)),
    )


def resolve_best_checkpoint_config(benchmark: Mapping[str, object]) -> BestCheckpointConfig:
    metric_name = str(benchmark.get("best_metric", "eval_return_mean")).strip() or "eval_return_mean"
    metric_mode = str(benchmark.get("best_metric_mode", "max")).strip().lower() or "max"
    if metric_mode not in {"max", "min"}:
        raise ValueError(f"unsupported benchmark['best_metric_mode'] {metric_mode!r}; expected 'max' or 'min'")
    return BestCheckpointConfig(metric_name=metric_name, metric_mode=metric_mode)


def compute_human_normalized_score(score: float, config: ScoreNormalizationConfig) -> float:
    return (float(score) - config.random_score) / (config.human_score - config.random_score) * config.scale


def augment_metrics_with_benchmark(metrics: MetricDict, benchmark: Mapping[str, object]) -> MetricDict:
    normalization = resolve_score_normalization_config(benchmark)
    if normalization is None:
        return metrics

    eval_return_mean = metrics.get("eval_return_mean")
    if eval_return_mean is None:
        return metrics

    metrics["eval_human_normalized_score"] = compute_human_normalized_score(float(eval_return_mean), normalization)
    return metrics


def aggregate_numeric_metrics(rows: Sequence[Mapping[str, object]]) -> MetricDict:
    """Aggregate only numeric keys shared by every row using population standard deviation."""
    if not rows:
        return {}

    # Keep aggregation semantics strict: a metric must exist in every row to be aggregated.
    shared_keys = set(rows[0].keys())
    for row in rows[1:]:
        shared_keys.intersection_update(row.keys())

    aggregated: MetricDict = {}
    for key in sorted(shared_keys):
        values: list[float] = []
        for row in rows:
            value = row[key]
            if isinstance(value, bool) or not isinstance(value, int | float):
                values = []
                break
            values.append(float(value))
        if not values:
            continue

        mean = sum(values) / len(values)
        # Use population variance/std because benchmark rows represent the full sweep being summarized.
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        aggregated[f"{key}_mean"] = mean
        aggregated[f"{key}_std"] = sqrt(variance)
        aggregated[f"{key}_min"] = min(values)
        aggregated[f"{key}_max"] = max(values)
    return aggregated
