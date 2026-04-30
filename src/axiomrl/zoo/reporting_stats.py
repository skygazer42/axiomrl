from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

from axiomrl.zoo.manifests import _build_manifest_alignment_severity
from axiomrl.zoo.reporting_runs import difference, sort_records


def seed_sort_key(seed: object) -> tuple[int, int | str]:
    if isinstance(seed, int):
        return (0, seed)
    seed_text = str(seed)
    try:
        return (0, int(seed_text))
    except ValueError:
        return (1, seed_text)


def mean(values: Sequence[object]) -> float | None:
    numeric_values = [float(cast(Any, value)) for value in values if value is not None]
    if not numeric_values:
        return None
    return sum(numeric_values) / len(numeric_values)


def sorted_numeric_values(values: Sequence[object]) -> list[float]:
    return sorted(float(cast(Any, value)) for value in values if value is not None)


def max_value(values: Sequence[object]) -> float | None:
    numeric_values = sorted_numeric_values(values)
    if not numeric_values:
        return None
    return max(numeric_values)


def min_value(values: Sequence[object]) -> float | None:
    numeric_values = sorted_numeric_values(values)
    if not numeric_values:
        return None
    return min(numeric_values)


def median(values: Sequence[object]) -> float | None:
    numeric_values = sorted_numeric_values(values)
    if not numeric_values:
        return None
    midpoint = len(numeric_values) // 2
    if len(numeric_values) % 2 == 1:
        return numeric_values[midpoint]
    return (numeric_values[midpoint - 1] + numeric_values[midpoint]) / 2.0


def quantile_inclusive(values: Sequence[object], quantile: float) -> float | None:
    numeric_values = sorted_numeric_values(values)
    if not numeric_values:
        return None
    if len(numeric_values) == 1:
        return numeric_values[0]
    position = (len(numeric_values) - 1) * quantile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return numeric_values[lower_index]
    weight = position - lower_index
    lower_value = numeric_values[lower_index]
    upper_value = numeric_values[upper_index]
    return lower_value + (upper_value - lower_value) * weight


def iqr(values: Sequence[object]) -> float | None:
    numeric_values = sorted_numeric_values(values)
    if len(numeric_values) < 2:
        return None
    q1 = quantile_inclusive(numeric_values, 0.25)
    q3 = quantile_inclusive(numeric_values, 0.75)
    if q1 is None or q3 is None:
        return None
    return q3 - q1


def stddev(values: Sequence[object]) -> float | None:
    numeric_values = sorted_numeric_values(values)
    if len(numeric_values) < 2:
        return None
    mean_value = sum(numeric_values) / len(numeric_values)
    variance = sum((value - mean_value) ** 2 for value in numeric_values) / (len(numeric_values) - 1)
    return math.sqrt(variance)


def stderr(values: Sequence[object]) -> float | None:
    numeric_values = [float(cast(Any, value)) for value in values if value is not None]
    if len(numeric_values) < 2:
        return None
    stddev_value = stddev(numeric_values)
    if stddev_value is None:
        return None
    return stddev_value / math.sqrt(len(numeric_values))


def ci95(values: Sequence[object]) -> float | None:
    stderr_value = stderr(values)
    if stderr_value is None:
        return None
    return 1.96 * stderr_value


def ratio(numerator: object, denominator: object) -> float | None:
    if numerator is None or denominator is None:
        return None
    denominator_value = float(cast(Any, denominator))
    if denominator_value == 0.0:
        return None
    return float(cast(Any, numerator)) / denominator_value


def single_or_multiple(values: list[object]) -> object | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    first = present[0]
    if all(value == first for value in present[1:]):
        return first
    return "multiple"


def attach_aggregate_rank_fields(aggregates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rank_specs = (
        ("best_eval_return_mean_max", "rank_best_eval_return_mean"),
        ("latest_eval_return_mean_mean", "rank_latest_eval_return_mean"),
        ("best_eval_human_normalized_score_max", "rank_best_eval_human_normalized_score"),
        ("latest_eval_human_normalized_score_mean", "rank_latest_eval_human_normalized_score"),
    )

    for _, rank_field in rank_specs:
        for aggregate in aggregates:
            aggregate[rank_field] = None

    for metric_field, rank_field in rank_specs:
        ranked_records = [
            (index, aggregate, float(aggregate[metric_field]))
            for index, aggregate in enumerate(aggregates)
            if aggregate.get(metric_field) is not None
        ]
        for rank, (_, aggregate, _) in enumerate(
            sorted(ranked_records, key=lambda item: item[2], reverse=True),
            start=1,
        ):
            aggregate[rank_field] = rank
    return aggregates


def attach_baseline_comparison_fields(
    aggregates: list[dict[str, Any]],
    *,
    baseline_preset: str,
) -> list[dict[str, Any]]:
    baseline_aggregate = next(
        (aggregate for aggregate in aggregates if str(aggregate.get("preset_name")) == baseline_preset),
        None,
    )
    if baseline_aggregate is None:
        raise ValueError(f"--baseline-preset {baseline_preset!r} was not found in the aggregate preset groups")

    baseline_return_mean = baseline_aggregate.get("latest_eval_return_mean_mean")
    baseline_normalized_mean = baseline_aggregate.get("latest_eval_human_normalized_score_mean")

    for aggregate in aggregates:
        aggregate["baseline_preset"] = baseline_preset
        aggregate["baseline_latest_eval_return_mean_mean"] = baseline_return_mean
        aggregate["baseline_latest_eval_human_normalized_score_mean"] = baseline_normalized_mean
        aggregate["delta_vs_baseline_latest_eval_return_mean_mean"] = difference(
            aggregate.get("latest_eval_return_mean_mean"),
            baseline_return_mean,
        )
        aggregate["delta_vs_baseline_latest_eval_human_normalized_score_mean"] = difference(
            aggregate.get("latest_eval_human_normalized_score_mean"),
            baseline_normalized_mean,
        )
        aggregate["ratio_vs_baseline_latest_eval_return_mean_mean"] = ratio(
            aggregate.get("latest_eval_return_mean_mean"),
            baseline_return_mean,
        )
        aggregate["ratio_vs_baseline_latest_eval_human_normalized_score_mean"] = ratio(
            aggregate.get("latest_eval_human_normalized_score_mean"),
            baseline_normalized_mean,
        )
    return aggregates


def baseline_summary_entry(aggregate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "group": aggregate.get("group"),
        "preset_name": aggregate.get("preset_name"),
        "preset_metadata": aggregate.get("preset_metadata"),
        "preset_config": aggregate.get("preset_config"),
        "preset_description": aggregate.get("preset_description"),
        "algo": aggregate.get("algo"),
        "env_id": aggregate.get("env_id"),
        "baseline_preset": aggregate.get("baseline_preset"),
        "latest_eval_return_mean_mean": aggregate.get("latest_eval_return_mean_mean"),
        "latest_eval_human_normalized_score_mean": aggregate.get("latest_eval_human_normalized_score_mean"),
        "baseline_latest_eval_return_mean_mean": aggregate.get("baseline_latest_eval_return_mean_mean"),
        "baseline_latest_eval_human_normalized_score_mean": aggregate.get(
            "baseline_latest_eval_human_normalized_score_mean"
        ),
        "delta_vs_baseline_latest_eval_return_mean_mean": aggregate.get(
            "delta_vs_baseline_latest_eval_return_mean_mean"
        ),
        "delta_vs_baseline_latest_eval_human_normalized_score_mean": aggregate.get(
            "delta_vs_baseline_latest_eval_human_normalized_score_mean"
        ),
        "ratio_vs_baseline_latest_eval_return_mean_mean": aggregate.get(
            "ratio_vs_baseline_latest_eval_return_mean_mean"
        ),
        "ratio_vs_baseline_latest_eval_human_normalized_score_mean": aggregate.get(
            "ratio_vs_baseline_latest_eval_human_normalized_score_mean"
        ),
        "manifest_alignment_status": aggregate.get("manifest_alignment_status"),
        "manifest_alignment_all_runs": aggregate.get("manifest_alignment_all_runs"),
        "manifest_alignment_total_runs": aggregate.get("manifest_alignment_total_runs"),
        "manifest_alignment_aligned_runs": aggregate.get("manifest_alignment_aligned_runs"),
        "manifest_alignment_drifted_runs": aggregate.get("manifest_alignment_drifted_runs"),
        "manifest_alignment_unknown_preset_runs": aggregate.get("manifest_alignment_unknown_preset_runs"),
        "manifest_alignment_protocol_mismatch_runs": aggregate.get("manifest_alignment_protocol_mismatch_runs"),
    }


def build_baseline_summary(
    aggregates: list[dict[str, Any]],
    *,
    baseline_preset: str,
    limit: int = 3,
) -> dict[str, Any]:
    comparison_candidates = [
        aggregate for aggregate in aggregates if str(aggregate.get("preset_name")) != baseline_preset
    ]

    def top_entries(metric_field: str, *, descending: bool) -> list[dict[str, Any]]:
        ordered = sort_records(comparison_candidates, sort_by=metric_field, descending=descending)
        return [baseline_summary_entry(entry) for entry in ordered[:limit]]

    return {
        "baseline_preset": baseline_preset,
        "top_movers_by_return_delta": top_entries(
            "delta_vs_baseline_latest_eval_return_mean_mean",
            descending=True,
        ),
        "top_regressions_by_return_delta": top_entries(
            "delta_vs_baseline_latest_eval_return_mean_mean",
            descending=False,
        ),
        "top_movers_by_normalized_delta": top_entries(
            "delta_vs_baseline_latest_eval_human_normalized_score_mean",
            descending=True,
        ),
        "top_regressions_by_normalized_delta": top_entries(
            "delta_vs_baseline_latest_eval_human_normalized_score_mean",
            descending=False,
        ),
    }


def aggregate_run_reports(
    reports: list[dict[str, Any]],
    *,
    group_by: str = "algo-env",
    min_seeds: int | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[object, ...], list[dict[str, Any]]] = {}
    for report in reports:
        key: tuple[object, ...]
        if group_by == "preset":
            preset_name = report.get("preset_name")
            key = (report.get("suite"), preset_name, report.get("protocol_name"))
        else:
            key = (str(report["algo"]), str(report["env_id"]))
        grouped.setdefault(key, []).append(report)

    aggregates: list[dict[str, Any]] = []
    for key, group in sorted(
        grouped.items(), key=lambda item: tuple("" if value is None else str(value) for value in item[0])
    ):
        ordered_group = sorted(group, key=lambda report: (seed_sort_key(report["seed"]), str(report["run_id"])))
        if min_seeds is not None and len(ordered_group) < min_seeds:
            continue
        seeds = ",".join(str(report["seed"]) for report in ordered_group)
        latest_return_values = [report["latest_eval_return_mean"] for report in ordered_group]
        latest_normalized_values = [report["latest_eval_human_normalized_score"] for report in ordered_group]
        aggregate = {
            "group_by": group_by,
            "suite": single_or_multiple([report.get("suite") for report in ordered_group]),
            "preset_name": single_or_multiple([report.get("preset_name") for report in ordered_group]),
            "protocol_name": single_or_multiple([report.get("protocol_name") for report in ordered_group]),
            "algo": single_or_multiple([report["algo"] for report in ordered_group]),
            "env_id": single_or_multiple([report["env_id"] for report in ordered_group]),
            "runs": len(ordered_group),
            "seed_count": len(ordered_group),
            "seeds": seeds,
            "latest_eval_return_mean_mean": mean(latest_return_values),
            "latest_eval_return_mean_median": median(latest_return_values),
            "latest_eval_return_mean_min": min_value(latest_return_values),
            "latest_eval_return_mean_max": max_value(latest_return_values),
            "latest_eval_return_mean_iqr": iqr(latest_return_values),
            "latest_eval_return_mean_std": stddev(latest_return_values),
            "latest_eval_return_mean_stderr": stderr(latest_return_values),
            "latest_eval_return_mean_ci95": ci95(latest_return_values),
            "latest_eval_human_normalized_score_mean": mean(latest_normalized_values),
            "latest_eval_human_normalized_score_median": median(latest_normalized_values),
            "latest_eval_human_normalized_score_min": min_value(latest_normalized_values),
            "latest_eval_human_normalized_score_max": max_value(latest_normalized_values),
            "latest_eval_human_normalized_score_iqr": iqr(latest_normalized_values),
            "latest_eval_human_normalized_score_std": stddev(latest_normalized_values),
            "latest_eval_human_normalized_score_stderr": stderr(latest_normalized_values),
            "latest_eval_human_normalized_score_ci95": ci95(latest_normalized_values),
            "best_eval_return_mean_max": max_value([report["best_eval_return_mean"] for report in ordered_group]),
            "best_eval_human_normalized_score_max": max_value(
                [report["best_eval_human_normalized_score"] for report in ordered_group]
            ),
        }
        manifest_alignment_total_runs = len(ordered_group)
        manifest_alignment_aligned_runs = sum(
            1 for report in ordered_group if report.get("manifest_alignment_status") == "aligned"
        )
        aggregate["manifest_alignment_total_runs"] = manifest_alignment_total_runs
        aggregate["manifest_alignment_aligned_runs"] = manifest_alignment_aligned_runs
        aggregate["manifest_alignment_drifted_runs"] = manifest_alignment_total_runs - manifest_alignment_aligned_runs
        aggregate["manifest_alignment_unknown_preset_runs"] = sum(
            1 for report in ordered_group if report.get("manifest_preset_known") is False
        )
        aggregate["manifest_alignment_protocol_mismatch_runs"] = sum(
            1 for report in ordered_group if report.get("manifest_protocol_matches_manifest") is False
        )
        aggregate["manifest_alignment_all_runs"] = aggregate["manifest_alignment_drifted_runs"] == 0
        aggregate["manifest_alignment_status"] = "aligned" if aggregate["manifest_alignment_all_runs"] else "mixed"
        aggregate["manifest_alignment_severity"] = _build_manifest_alignment_severity(
            drifted_runs=aggregate["manifest_alignment_drifted_runs"],
            unknown_preset_runs=aggregate["manifest_alignment_unknown_preset_runs"],
            protocol_mismatch_runs=aggregate["manifest_alignment_protocol_mismatch_runs"],
        )
        aggregate["best_minus_latest_eval_return_mean_gap"] = difference(
            aggregate["best_eval_return_mean_max"],
            aggregate["latest_eval_return_mean_mean"],
        )
        aggregate["best_minus_latest_eval_human_normalized_score_gap"] = difference(
            aggregate["best_eval_human_normalized_score_max"],
            aggregate["latest_eval_human_normalized_score_mean"],
        )
        aggregate["best_over_latest_eval_return_ratio"] = ratio(
            aggregate["best_eval_return_mean_max"],
            aggregate["latest_eval_return_mean_mean"],
        )
        aggregate["best_over_latest_eval_human_normalized_score_ratio"] = ratio(
            aggregate["best_eval_human_normalized_score_max"],
            aggregate["latest_eval_human_normalized_score_mean"],
        )
        if group_by == "preset":
            aggregate["group"] = aggregate["preset_name"]
        else:
            aggregate["group"] = f"{aggregate['algo']}::{aggregate['env_id']}"
        aggregates.append(aggregate)
    return attach_aggregate_rank_fields(aggregates)
