from __future__ import annotations

import csv
from collections.abc import Mapping
import io
import json
import math
from pathlib import Path
from typing import Any

from rl_training.zoo.manifests import (
    _attach_manifest_metadata,
    _build_manifest_alignment_severity,
    _build_manifest_alignment_summary,
    _build_manifest_metadata,
    _build_manifest_preset_lookup,
    _build_payload_metadata_fields,
    _copy_metadata_value,
    _resolve_manifest_protocol_metadata,
    _resolve_manifest_protocol_name,
    _resolve_manifest_score_normalization_metadata,
)


def _load_run_metadata(metadata_path: Path) -> dict[str, Any]:
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object in {metadata_path}, got {type(payload)!r}")
    return payload


def _iter_run_reports(runs_dir: Path) -> list[dict[str, Any]]:
    if not runs_dir.exists():
        return []

    reports: list[dict[str, Any]] = []
    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        metadata_path = run_dir / "metadata.json"
        if not metadata_path.exists():
            continue
        payload = _load_run_metadata(metadata_path)
        latest_metrics = payload.get("latest_metrics", {})
        best_checkpoint = payload.get("best_checkpoint", {})
        benchmark = payload.get("benchmark", {})
        if not isinstance(latest_metrics, dict):
            latest_metrics = {}
        if not isinstance(best_checkpoint, dict):
            best_checkpoint = {}
        if not isinstance(benchmark, dict):
            benchmark = {}
        reports.append(
            {
                "run_id": run_dir.name,
                "algo": payload.get("algo", "unknown"),
                "env_id": payload.get("env_id", "unknown"),
                "seed": payload.get("seed", "unknown"),
                "suite": benchmark.get("suite"),
                "preset_name": benchmark.get("preset_name"),
                "protocol_name": benchmark.get("protocol_name"),
                "latest_eval_return_mean": latest_metrics.get("eval_return_mean"),
                "latest_eval_human_normalized_score": latest_metrics.get("eval_human_normalized_score"),
                "best_eval_return_mean": latest_metrics.get("best_eval_return_mean", best_checkpoint.get("metric_value")),
                "best_eval_human_normalized_score": latest_metrics.get(
                    "best_eval_human_normalized_score",
                    best_checkpoint.get("eval_human_normalized_score"),
                ),
                "best_minus_latest_eval_return_mean": _difference(
                    latest_metrics.get("best_eval_return_mean", best_checkpoint.get("metric_value")),
                    latest_metrics.get("eval_return_mean"),
                ),
                "best_minus_latest_eval_human_normalized_score": _difference(
                    latest_metrics.get(
                        "best_eval_human_normalized_score",
                        best_checkpoint.get("eval_human_normalized_score"),
                    ),
                    latest_metrics.get("eval_human_normalized_score"),
                ),
                "best_checkpoint_path": best_checkpoint.get("path"),
            }
        )
    return reports


def _seed_sort_key(seed: object) -> tuple[int, int | str]:
    if isinstance(seed, int):
        return (0, seed)
    seed_text = str(seed)
    try:
        return (0, int(seed_text))
    except ValueError:
        return (1, seed_text)


def _mean(values: list[object]) -> float | None:
    numeric_values = [float(value) for value in values if value is not None]
    if not numeric_values:
        return None
    return sum(numeric_values) / len(numeric_values)


def _sorted_numeric_values(values: list[object]) -> list[float]:
    return sorted(float(value) for value in values if value is not None)


def _max_value(values: list[object]) -> float | None:
    numeric_values = _sorted_numeric_values(values)
    if not numeric_values:
        return None
    return max(numeric_values)


def _min_value(values: list[object]) -> float | None:
    numeric_values = _sorted_numeric_values(values)
    if not numeric_values:
        return None
    return min(numeric_values)


def _median(values: list[object]) -> float | None:
    numeric_values = _sorted_numeric_values(values)
    if not numeric_values:
        return None
    midpoint = len(numeric_values) // 2
    if len(numeric_values) % 2 == 1:
        return numeric_values[midpoint]
    return (numeric_values[midpoint - 1] + numeric_values[midpoint]) / 2.0


def _quantile_inclusive(values: list[object], quantile: float) -> float | None:
    numeric_values = _sorted_numeric_values(values)
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


def _iqr(values: list[object]) -> float | None:
    numeric_values = _sorted_numeric_values(values)
    if len(numeric_values) < 2:
        return None
    q1 = _quantile_inclusive(numeric_values, 0.25)
    q3 = _quantile_inclusive(numeric_values, 0.75)
    if q1 is None or q3 is None:
        return None
    return q3 - q1


def _stddev(values: list[object]) -> float | None:
    numeric_values = _sorted_numeric_values(values)
    if len(numeric_values) < 2:
        return None
    mean_value = sum(numeric_values) / len(numeric_values)
    variance = sum((value - mean_value) ** 2 for value in numeric_values) / (len(numeric_values) - 1)
    return math.sqrt(variance)


def _stderr(values: list[object]) -> float | None:
    numeric_values = [float(value) for value in values if value is not None]
    if len(numeric_values) < 2:
        return None
    stddev = _stddev(numeric_values)
    if stddev is None:
        return None
    return stddev / math.sqrt(len(numeric_values))


def _ci95(values: list[object]) -> float | None:
    stderr = _stderr(values)
    if stderr is None:
        return None
    return 1.96 * stderr


def _difference(lhs: object, rhs: object) -> float | None:
    if lhs is None or rhs is None:
        return None
    return float(lhs) - float(rhs)


def _ratio(numerator: object, denominator: object) -> float | None:
    if numerator is None or denominator is None:
        return None
    denominator_value = float(denominator)
    if denominator_value == 0.0:
        return None
    return float(numerator) / denominator_value


def _single_or_multiple(values: list[object]) -> object | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    first = present[0]
    if all(value == first for value in present[1:]):
        return first
    return "multiple"


def _attach_aggregate_rank_fields(aggregates: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


def _attach_baseline_comparison_fields(
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
        aggregate["delta_vs_baseline_latest_eval_return_mean_mean"] = _difference(
            aggregate.get("latest_eval_return_mean_mean"),
            baseline_return_mean,
        )
        aggregate["delta_vs_baseline_latest_eval_human_normalized_score_mean"] = _difference(
            aggregate.get("latest_eval_human_normalized_score_mean"),
            baseline_normalized_mean,
        )
        aggregate["ratio_vs_baseline_latest_eval_return_mean_mean"] = _ratio(
            aggregate.get("latest_eval_return_mean_mean"),
            baseline_return_mean,
        )
        aggregate["ratio_vs_baseline_latest_eval_human_normalized_score_mean"] = _ratio(
            aggregate.get("latest_eval_human_normalized_score_mean"),
            baseline_normalized_mean,
        )
    return aggregates


def _baseline_summary_entry(aggregate: Mapping[str, Any]) -> dict[str, Any]:
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
        "baseline_latest_eval_human_normalized_score_mean": aggregate.get("baseline_latest_eval_human_normalized_score_mean"),
        "delta_vs_baseline_latest_eval_return_mean_mean": aggregate.get("delta_vs_baseline_latest_eval_return_mean_mean"),
        "delta_vs_baseline_latest_eval_human_normalized_score_mean": aggregate.get(
            "delta_vs_baseline_latest_eval_human_normalized_score_mean"
        ),
        "ratio_vs_baseline_latest_eval_return_mean_mean": aggregate.get("ratio_vs_baseline_latest_eval_return_mean_mean"),
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


def _build_baseline_summary(
    aggregates: list[dict[str, Any]],
    *,
    baseline_preset: str,
    limit: int = 3,
) -> dict[str, Any]:
    comparison_candidates = [
        aggregate for aggregate in aggregates if str(aggregate.get("preset_name")) != baseline_preset
    ]

    def top_entries(metric_field: str, *, descending: bool) -> list[dict[str, Any]]:
        ordered = _sort_records(comparison_candidates, sort_by=metric_field, descending=descending)
        return [_baseline_summary_entry(entry) for entry in ordered[:limit]]

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


def _aggregate_run_reports(
    reports: list[dict[str, Any]],
    *,
    group_by: str = "algo-env",
    min_seeds: int | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[object, ...], list[dict[str, Any]]] = {}
    for report in reports:
        if group_by == "preset":
            preset_name = report.get("preset_name")
            key = (report.get("suite"), preset_name, report.get("protocol_name"))
        else:
            key = (str(report["algo"]), str(report["env_id"]))
        grouped.setdefault(key, []).append(report)

    aggregates: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items(), key=lambda item: tuple("" if value is None else str(value) for value in item[0])):
        ordered_group = sorted(group, key=lambda report: (_seed_sort_key(report["seed"]), str(report["run_id"])))
        if min_seeds is not None and len(ordered_group) < min_seeds:
            continue
        seeds = ",".join(str(report["seed"]) for report in ordered_group)
        latest_return_values = [report["latest_eval_return_mean"] for report in ordered_group]
        latest_normalized_values = [report["latest_eval_human_normalized_score"] for report in ordered_group]
        aggregate = {
            "group_by": group_by,
            "suite": _single_or_multiple([report.get("suite") for report in ordered_group]),
            "preset_name": _single_or_multiple([report.get("preset_name") for report in ordered_group]),
            "protocol_name": _single_or_multiple([report.get("protocol_name") for report in ordered_group]),
            "algo": _single_or_multiple([report["algo"] for report in ordered_group]),
            "env_id": _single_or_multiple([report["env_id"] for report in ordered_group]),
            "runs": len(ordered_group),
            "seed_count": len(ordered_group),
            "seeds": seeds,
            "latest_eval_return_mean_mean": _mean(latest_return_values),
            "latest_eval_return_mean_median": _median(latest_return_values),
            "latest_eval_return_mean_min": _min_value(latest_return_values),
            "latest_eval_return_mean_max": _max_value(latest_return_values),
            "latest_eval_return_mean_iqr": _iqr(latest_return_values),
            "latest_eval_return_mean_std": _stddev(latest_return_values),
            "latest_eval_return_mean_stderr": _stderr(latest_return_values),
            "latest_eval_return_mean_ci95": _ci95(latest_return_values),
            "latest_eval_human_normalized_score_mean": _mean(latest_normalized_values),
            "latest_eval_human_normalized_score_median": _median(latest_normalized_values),
            "latest_eval_human_normalized_score_min": _min_value(latest_normalized_values),
            "latest_eval_human_normalized_score_max": _max_value(latest_normalized_values),
            "latest_eval_human_normalized_score_iqr": _iqr(latest_normalized_values),
            "latest_eval_human_normalized_score_std": _stddev(latest_normalized_values),
            "latest_eval_human_normalized_score_stderr": _stderr(latest_normalized_values),
            "latest_eval_human_normalized_score_ci95": _ci95(latest_normalized_values),
            "best_eval_return_mean_max": _max_value([report["best_eval_return_mean"] for report in ordered_group]),
            "best_eval_human_normalized_score_max": _max_value(
                [report["best_eval_human_normalized_score"] for report in ordered_group]
            ),
        }
        manifest_alignment_total_runs = len(ordered_group)
        manifest_alignment_aligned_runs = sum(
            1 for report in ordered_group if report.get("manifest_alignment_status") == "aligned"
        )
        aggregate["manifest_alignment_total_runs"] = manifest_alignment_total_runs
        aggregate["manifest_alignment_aligned_runs"] = manifest_alignment_aligned_runs
        aggregate["manifest_alignment_drifted_runs"] = (
            manifest_alignment_total_runs - manifest_alignment_aligned_runs
        )
        aggregate["manifest_alignment_unknown_preset_runs"] = sum(
            1 for report in ordered_group if report.get("manifest_preset_known") is False
        )
        aggregate["manifest_alignment_protocol_mismatch_runs"] = sum(
            1 for report in ordered_group if report.get("manifest_protocol_matches_manifest") is False
        )
        aggregate["manifest_alignment_all_runs"] = aggregate["manifest_alignment_drifted_runs"] == 0
        aggregate["manifest_alignment_status"] = (
            "aligned" if aggregate["manifest_alignment_all_runs"] else "mixed"
        )
        aggregate["manifest_alignment_severity"] = _build_manifest_alignment_severity(
            drifted_runs=aggregate["manifest_alignment_drifted_runs"],
            unknown_preset_runs=aggregate["manifest_alignment_unknown_preset_runs"],
            protocol_mismatch_runs=aggregate["manifest_alignment_protocol_mismatch_runs"],
        )
        aggregate["best_minus_latest_eval_return_mean_gap"] = _difference(
            aggregate["best_eval_return_mean_max"],
            aggregate["latest_eval_return_mean_mean"],
        )
        aggregate["best_minus_latest_eval_human_normalized_score_gap"] = _difference(
            aggregate["best_eval_human_normalized_score_max"],
            aggregate["latest_eval_human_normalized_score_mean"],
        )
        aggregate["best_over_latest_eval_return_ratio"] = _ratio(
            aggregate["best_eval_return_mean_max"],
            aggregate["latest_eval_return_mean_mean"],
        )
        aggregate["best_over_latest_eval_human_normalized_score_ratio"] = _ratio(
            aggregate["best_eval_human_normalized_score_max"],
            aggregate["latest_eval_human_normalized_score_mean"],
        )
        if group_by == "preset":
            aggregate["group"] = aggregate["preset_name"]
        else:
            aggregate["group"] = f"{aggregate['algo']}::{aggregate['env_id']}"
        aggregates.append(aggregate)
    return _attach_aggregate_rank_fields(aggregates)


def _filter_run_reports(
    reports: list[dict[str, Any]],
    *,
    algo: str | None = None,
    env_id: str | None = None,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for report in reports:
        if algo is not None and str(report["algo"]) != algo:
            continue
        if env_id is not None and str(report["env_id"]) != env_id:
            continue
        filtered.append(report)
    return filtered


def _resolve_sort_value(record: Mapping[str, object], sort_by: str) -> object | None:
    for candidate in (sort_by, f"{sort_by}_mean", f"{sort_by}_max"):
        if candidate in record and record[candidate] is not None:
            return record[candidate]
    return None


def _sortable_value(value: object) -> tuple[int, float | str]:
    if isinstance(value, (int, float)):
        return (0, float(value))

    value_text = str(value)
    try:
        return (0, float(value_text))
    except ValueError:
        return (1, value_text.lower())


def _sort_records(
    records: list[dict[str, Any]],
    *,
    sort_by: str | None = None,
    descending: bool = False,
) -> list[dict[str, Any]]:
    if not sort_by:
        return list(records)

    with_values: list[tuple[int, dict[str, Any], tuple[int, float | str]]] = []
    without_values: list[tuple[int, dict[str, Any]]] = []
    for index, record in enumerate(records):
        resolved = _resolve_sort_value(record, sort_by)
        if resolved is None:
            without_values.append((index, record))
            continue
        with_values.append((index, record, _sortable_value(resolved)))

    ordered = sorted(with_values, key=lambda item: (item[2], item[0]), reverse=descending)
    return [record for _, record, _ in ordered] + [record for _, record in without_values]


def _apply_top_k(records: list[dict[str, Any]], *, top_k: int | None = None) -> list[dict[str, Any]]:
    if top_k is None:
        return list(records)
    return list(records[:top_k])


def build_report_payload(
    manifest: dict[str, Any],
    *,
    runs_dir: Path,
    manifest_source: Mapping[str, object] | None = None,
    algo: str | None = None,
    env_id: str | None = None,
    group_by: str = "algo-env",
    min_seeds: int | None = None,
    top_k: int | None = None,
    baseline_preset: str | None = None,
    sort_by: str | None = None,
    descending: bool = False,
) -> dict[str, Any]:
    suite = manifest.get("suite", "unknown")
    protocol = manifest.get("protocol", {})
    score_normalization = manifest.get("score_normalization", {})
    manifest_metadata = _build_manifest_metadata(manifest)
    manifest_protocol_name = _resolve_manifest_protocol_name(manifest)
    protocol_metadata = _resolve_manifest_protocol_metadata(manifest)
    score_normalization_metadata = _resolve_manifest_score_normalization_metadata(manifest)
    preset_lookup = _build_manifest_preset_lookup(manifest)
    sorted_reports = _sort_records(
        _filter_run_reports(_iter_run_reports(runs_dir), algo=algo, env_id=env_id),
        sort_by=sort_by,
        descending=descending,
    )
    enriched_sorted_reports = [
        _attach_manifest_metadata(
            report,
            protocol_metadata=protocol_metadata,
            manifest_protocol_name=manifest_protocol_name,
            score_normalization_metadata=score_normalization_metadata,
            preset_lookup=preset_lookup,
        )
        for report in sorted_reports
    ]
    reports = _apply_top_k(enriched_sorted_reports, top_k=top_k)
    aggregate_records = [
        _attach_manifest_metadata(
            aggregate,
            protocol_metadata=protocol_metadata,
            manifest_protocol_name=manifest_protocol_name,
            score_normalization_metadata=score_normalization_metadata,
            preset_lookup=preset_lookup,
        )
        for aggregate in _aggregate_run_reports(enriched_sorted_reports, group_by=group_by, min_seeds=min_seeds)
    ]
    baseline_summary: dict[str, Any] | None = None
    manifest_alignment_summary = _build_manifest_alignment_summary(enriched_sorted_reports)
    if baseline_preset is not None:
        if group_by != "preset":
            raise ValueError("--baseline-preset requires --group-by preset")
        aggregate_records = _attach_baseline_comparison_fields(
            aggregate_records,
            baseline_preset=baseline_preset,
        )
        baseline_summary = _build_baseline_summary(
            aggregate_records,
            baseline_preset=baseline_preset,
        )
    aggregates = _apply_top_k(
        _sort_records(
            aggregate_records,
            sort_by=sort_by,
            descending=descending,
        ),
        top_k=top_k,
    )

    payload: dict[str, Any] = {
        "suite": suite,
        "protocol": protocol.get("name", "unknown") if isinstance(protocol, dict) else "unknown",
        "score_normalization": score_normalization.get("type", "none") if isinstance(score_normalization, dict) else "none",
        "runs_dir": str(runs_dir),
        "group_by": group_by,
        "manifest_metadata": manifest_metadata,
        "manifest_alignment_summary": manifest_alignment_summary,
        "runs": reports,
        "aggregates": aggregates,
    }
    if manifest_source is not None:
        payload["manifest_source"] = _copy_metadata_value(manifest_source)
    if protocol_metadata is not None:
        payload["protocol_metadata"] = _copy_metadata_value(protocol_metadata)
    if score_normalization_metadata is not None:
        payload["score_normalization_metadata"] = _copy_metadata_value(score_normalization_metadata)
    if baseline_preset is not None:
        payload["baseline_preset"] = baseline_preset
    if baseline_summary is not None:
        payload["baseline_summary"] = baseline_summary
    if algo is not None or env_id is not None:
        payload["filters"] = {"algo": algo, "env_id": env_id}
    if min_seeds is not None:
        payload["min_seeds"] = min_seeds
    if sort_by is not None:
        payload["sort_by"] = sort_by
        payload["descending"] = descending
    if top_k is not None:
        payload["top_k"] = top_k
    return payload


def _render_text_report(payload: Mapping[str, Any]) -> str:
    reports = payload["runs"]
    aggregates = payload["aggregates"]
    lines = [
        f"suite={payload['suite']}",
        f"protocol={payload['protocol']}",
        f"score_normalization={payload['score_normalization']}",
        f"runs_dir={payload['runs_dir']}",
    ]
    metadata_fields = _build_payload_metadata_fields(payload)
    if metadata_fields["manifest_requested_path"] is not None:
        lines.append(f"manifest_requested_path={metadata_fields['manifest_requested_path']}")
    if metadata_fields["manifest_resolved_path"] is not None:
        lines.append(f"manifest_resolved_path={metadata_fields['manifest_resolved_path']}")
    if metadata_fields["manifest_source_kind"] is not None:
        lines.append(f"manifest_source_kind={metadata_fields['manifest_source_kind']}")
    if metadata_fields["manifest_fingerprint"] is not None:
        lines.append(f"manifest_fingerprint={metadata_fields['manifest_fingerprint']}")
    if metadata_fields["manifest_preset_count"] is not None:
        lines.append(f"manifest_preset_count={metadata_fields['manifest_preset_count']}")
    if metadata_fields["manifest_alignment_total_runs"] is not None:
        lines.append(f"manifest_alignment_total_runs={metadata_fields['manifest_alignment_total_runs']}")
    if metadata_fields["manifest_alignment_aligned_runs"] is not None:
        lines.append(f"manifest_alignment_aligned_runs={metadata_fields['manifest_alignment_aligned_runs']}")
    if metadata_fields["manifest_alignment_drifted_runs"] is not None:
        lines.append(f"manifest_alignment_drifted_runs={metadata_fields['manifest_alignment_drifted_runs']}")
    if metadata_fields["manifest_alignment_unknown_preset_runs"] is not None:
        lines.append(
            f"manifest_alignment_unknown_preset_runs={metadata_fields['manifest_alignment_unknown_preset_runs']}"
        )
    if metadata_fields["manifest_alignment_protocol_mismatch_runs"] is not None:
        lines.append(
            "manifest_alignment_protocol_mismatch_runs="
            f"{metadata_fields['manifest_alignment_protocol_mismatch_runs']}"
        )
    if metadata_fields["manifest_alignment_all_runs"] is not None:
        lines.append(f"manifest_alignment_all_runs={metadata_fields['manifest_alignment_all_runs']}")
    if metadata_fields["manifest_alignment_severity"] is not None:
        lines.append(f"manifest_alignment_severity={metadata_fields['manifest_alignment_severity']}")
    if metadata_fields["manifest_alignment_fail_reasons"] is not None:
        lines.append(f"manifest_alignment_fail_reasons={metadata_fields['manifest_alignment_fail_reasons']}")
    if metadata_fields["protocol_description"] is not None:
        lines.append(f"protocol_description={metadata_fields['protocol_description']}")
    if metadata_fields["protocol_training"] is not None:
        lines.append(f"protocol_training={metadata_fields['protocol_training']}")
    if metadata_fields["protocol_evaluation"] is not None:
        lines.append(f"protocol_evaluation={metadata_fields['protocol_evaluation']}")
    if metadata_fields["score_normalization_game"] is not None:
        lines.append(f"score_normalization_game={metadata_fields['score_normalization_game']}")
    if metadata_fields["score_normalization_source"] is not None:
        lines.append(f"score_normalization_source={metadata_fields['score_normalization_source']}")
    if metadata_fields["score_normalization_random_score"] is not None:
        lines.append(f"score_normalization_random_score={metadata_fields['score_normalization_random_score']}")
    if metadata_fields["score_normalization_human_score"] is not None:
        lines.append(f"score_normalization_human_score={metadata_fields['score_normalization_human_score']}")
    if metadata_fields["score_normalization_scale"] is not None:
        lines.append(f"score_normalization_scale={metadata_fields['score_normalization_scale']}")
    if payload.get("baseline_preset") is not None:
        lines.append(f"baseline_preset={payload['baseline_preset']}")

    for report in reports:
        lines.append(
            " ".join(
                [
                    f"run_id={report['run_id']}",
                    f"algo={report['algo']}",
                    f"env_id={report['env_id']}",
                    f"seed={report['seed']}",
                    f"preset_name={report['preset_name']}",
                    f"preset_config={report.get('preset_config')}",
                    f"preset_description={report.get('preset_description')}",
                    f"protocol_name={report['protocol_name']}",
                    f"manifest_preset_known={report.get('manifest_preset_known')}",
                    f"manifest_protocol_matches_manifest={report.get('manifest_protocol_matches_manifest')}",
                    f"manifest_alignment_status={report.get('manifest_alignment_status')}",
                    f"manifest_alignment_severity={report.get('manifest_alignment_severity')}",
                    f"latest_eval_return_mean={report['latest_eval_return_mean']}",
                    f"latest_eval_human_normalized_score={report['latest_eval_human_normalized_score']}",
                    f"best_eval_return_mean={report['best_eval_return_mean']}",
                    f"best_eval_human_normalized_score={report['best_eval_human_normalized_score']}",
                    f"best_minus_latest_eval_return_mean={report['best_minus_latest_eval_return_mean']}",
                    f"best_minus_latest_eval_human_normalized_score={report['best_minus_latest_eval_human_normalized_score']}",
                    f"best_checkpoint_path={report['best_checkpoint_path']}",
                ]
            )
        )

    for aggregate in aggregates:
        lines.append(
            " ".join(
                [
                    "aggregate",
                    f"algo={aggregate['algo']}",
                    f"env_id={aggregate['env_id']}",
                    f"runs={aggregate['runs']}",
                    f"seeds={aggregate['seeds']}",
                    f"seed_count={aggregate['seed_count']}",
                    f"group_by={aggregate['group_by']}",
                    f"group={aggregate['group']}",
                    f"preset_name={aggregate['preset_name']}",
                    f"preset_config={aggregate.get('preset_config')}",
                    f"preset_description={aggregate.get('preset_description')}",
                    f"baseline_preset={aggregate.get('baseline_preset')}",
                    f"protocol_name={aggregate['protocol_name']}",
                    f"manifest_alignment_status={aggregate.get('manifest_alignment_status')}",
                    f"manifest_alignment_severity={aggregate.get('manifest_alignment_severity')}",
                    f"manifest_alignment_all_runs={aggregate.get('manifest_alignment_all_runs')}",
                    f"manifest_alignment_total_runs={aggregate.get('manifest_alignment_total_runs')}",
                    f"manifest_alignment_aligned_runs={aggregate.get('manifest_alignment_aligned_runs')}",
                    f"manifest_alignment_drifted_runs={aggregate.get('manifest_alignment_drifted_runs')}",
                    f"manifest_alignment_unknown_preset_runs={aggregate.get('manifest_alignment_unknown_preset_runs')}",
                    "manifest_alignment_protocol_mismatch_runs="
                    f"{aggregate.get('manifest_alignment_protocol_mismatch_runs')}",
                    f"latest_eval_return_mean_mean={aggregate['latest_eval_return_mean_mean']}",
                    f"latest_eval_return_mean_median={aggregate['latest_eval_return_mean_median']}",
                    f"latest_eval_return_mean_min={aggregate['latest_eval_return_mean_min']}",
                    f"latest_eval_return_mean_max={aggregate['latest_eval_return_mean_max']}",
                    f"latest_eval_return_mean_iqr={aggregate['latest_eval_return_mean_iqr']}",
                    f"latest_eval_return_mean_std={aggregate['latest_eval_return_mean_std']}",
                    f"latest_eval_return_mean_stderr={aggregate['latest_eval_return_mean_stderr']}",
                    f"latest_eval_return_mean_ci95={aggregate['latest_eval_return_mean_ci95']}",
                    f"latest_eval_human_normalized_score_mean={aggregate['latest_eval_human_normalized_score_mean']}",
                    f"latest_eval_human_normalized_score_median={aggregate['latest_eval_human_normalized_score_median']}",
                    f"latest_eval_human_normalized_score_min={aggregate['latest_eval_human_normalized_score_min']}",
                    f"latest_eval_human_normalized_score_max={aggregate['latest_eval_human_normalized_score_max']}",
                    f"latest_eval_human_normalized_score_iqr={aggregate['latest_eval_human_normalized_score_iqr']}",
                    f"latest_eval_human_normalized_score_std={aggregate['latest_eval_human_normalized_score_std']}",
                    f"latest_eval_human_normalized_score_stderr={aggregate['latest_eval_human_normalized_score_stderr']}",
                    f"latest_eval_human_normalized_score_ci95={aggregate['latest_eval_human_normalized_score_ci95']}",
                    f"best_eval_return_mean_max={aggregate['best_eval_return_mean_max']}",
                    f"best_eval_human_normalized_score_max={aggregate['best_eval_human_normalized_score_max']}",
                    f"best_minus_latest_eval_return_mean_gap={aggregate['best_minus_latest_eval_return_mean_gap']}",
                    f"best_minus_latest_eval_human_normalized_score_gap={aggregate['best_minus_latest_eval_human_normalized_score_gap']}",
                    f"best_over_latest_eval_return_ratio={aggregate['best_over_latest_eval_return_ratio']}",
                    f"best_over_latest_eval_human_normalized_score_ratio={aggregate['best_over_latest_eval_human_normalized_score_ratio']}",
                    f"baseline_latest_eval_return_mean_mean={aggregate.get('baseline_latest_eval_return_mean_mean')}",
                    f"baseline_latest_eval_human_normalized_score_mean={aggregate.get('baseline_latest_eval_human_normalized_score_mean')}",
                    f"delta_vs_baseline_latest_eval_return_mean_mean={aggregate.get('delta_vs_baseline_latest_eval_return_mean_mean')}",
                    f"delta_vs_baseline_latest_eval_human_normalized_score_mean={aggregate.get('delta_vs_baseline_latest_eval_human_normalized_score_mean')}",
                    f"ratio_vs_baseline_latest_eval_return_mean_mean={aggregate.get('ratio_vs_baseline_latest_eval_return_mean_mean')}",
                    f"ratio_vs_baseline_latest_eval_human_normalized_score_mean={aggregate.get('ratio_vs_baseline_latest_eval_human_normalized_score_mean')}",
                    f"rank_best_eval_return_mean={aggregate['rank_best_eval_return_mean']}",
                    f"rank_latest_eval_return_mean={aggregate['rank_latest_eval_return_mean']}",
                    f"rank_best_eval_human_normalized_score={aggregate['rank_best_eval_human_normalized_score']}",
                    f"rank_latest_eval_human_normalized_score={aggregate['rank_latest_eval_human_normalized_score']}",
                ]
            )
        )

    summary = payload.get("baseline_summary")
    if isinstance(summary, Mapping):
        for summary_kind in (
            "top_movers_by_return_delta",
            "top_regressions_by_return_delta",
            "top_movers_by_normalized_delta",
            "top_regressions_by_normalized_delta",
        ):
            entries = summary.get(summary_kind, [])
            if not isinstance(entries, list):
                continue
            for rank, entry in enumerate(entries, start=1):
                lines.append(
                    " ".join(
                        [
                            "summary",
                            f"baseline_preset={summary.get('baseline_preset')}",
                            f"summary_kind={summary_kind}",
                            f"summary_rank={rank}",
                            f"preset_name={entry.get('preset_name')}",
                            f"preset_config={entry.get('preset_config')}",
                            f"preset_description={entry.get('preset_description')}",
                            f"algo={entry.get('algo')}",
                            f"env_id={entry.get('env_id')}",
                            f"manifest_alignment_status={entry.get('manifest_alignment_status')}",
                            f"manifest_alignment_severity={entry.get('manifest_alignment_severity')}",
                            f"manifest_alignment_all_runs={entry.get('manifest_alignment_all_runs')}",
                            f"manifest_alignment_total_runs={entry.get('manifest_alignment_total_runs')}",
                            f"manifest_alignment_aligned_runs={entry.get('manifest_alignment_aligned_runs')}",
                            f"manifest_alignment_drifted_runs={entry.get('manifest_alignment_drifted_runs')}",
                            f"manifest_alignment_unknown_preset_runs={entry.get('manifest_alignment_unknown_preset_runs')}",
                            "manifest_alignment_protocol_mismatch_runs="
                            f"{entry.get('manifest_alignment_protocol_mismatch_runs')}",
                            f"delta_vs_baseline_latest_eval_return_mean_mean={entry.get('delta_vs_baseline_latest_eval_return_mean_mean')}",
                            f"delta_vs_baseline_latest_eval_human_normalized_score_mean={entry.get('delta_vs_baseline_latest_eval_human_normalized_score_mean')}",
                            f"ratio_vs_baseline_latest_eval_return_mean_mean={entry.get('ratio_vs_baseline_latest_eval_return_mean_mean')}",
                            f"ratio_vs_baseline_latest_eval_human_normalized_score_mean={entry.get('ratio_vs_baseline_latest_eval_human_normalized_score_mean')}",
                        ]
                    )
                )
    return "\n".join(lines) + "\n"


def _render_json_report(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2) + "\n"


def _csv_report_rows(payload: Mapping[str, Any]) -> list[dict[str, object]]:
    common = {
        "suite": payload["suite"],
        "protocol": payload["protocol"],
        "score_normalization": payload["score_normalization"],
        "runs_dir": payload["runs_dir"],
        "baseline_preset": payload.get("baseline_preset"),
    }
    common.update(_build_payload_metadata_fields(payload))
    rows: list[dict[str, object]] = []
    for report in payload["runs"]:
        row = dict(common)
        row["kind"] = "run"
        row.update(report)
        rows.append(row)
    for aggregate in payload["aggregates"]:
        row = dict(common)
        row["kind"] = "aggregate"
        row.update(aggregate)
        rows.append(row)
    summary = payload.get("baseline_summary")
    if isinstance(summary, Mapping):
        for summary_kind in (
            "top_movers_by_return_delta",
            "top_regressions_by_return_delta",
            "top_movers_by_normalized_delta",
            "top_regressions_by_normalized_delta",
        ):
            entries = summary.get(summary_kind, [])
            if not isinstance(entries, list):
                continue
            for rank, entry in enumerate(entries, start=1):
                row = dict(common)
                row["kind"] = "summary"
                row["summary_kind"] = summary_kind
                row["summary_rank"] = rank
                row.update(entry)
                rows.append(row)
    return rows


def _render_csv_report(payload: Mapping[str, Any]) -> str:
    rows = _csv_report_rows(payload)
    fieldnames = [
        "kind",
        "summary_kind",
        "summary_rank",
        "suite",
        "manifest_requested_path",
        "manifest_resolved_path",
        "manifest_source_kind",
        "manifest_fingerprint",
        "manifest_preset_count",
        "manifest_preset_names",
        "manifest_alignment_total_runs",
        "manifest_alignment_aligned_runs",
        "manifest_alignment_drifted_runs",
        "manifest_alignment_unknown_preset_runs",
        "manifest_alignment_protocol_mismatch_runs",
        "manifest_alignment_all_runs",
        "manifest_alignment_severity",
        "manifest_alignment_drifted_presets",
        "manifest_alignment_fail_reasons",
        "protocol",
        "protocol_description",
        "protocol_training",
        "protocol_evaluation",
        "score_normalization",
        "score_normalization_game",
        "score_normalization_source",
        "score_normalization_random_score",
        "score_normalization_human_score",
        "score_normalization_scale",
        "runs_dir",
        "baseline_preset",
        "run_id",
        "algo",
        "env_id",
        "seed",
        "preset_name",
        "preset_config",
        "preset_description",
        "protocol_name",
        "manifest_preset_known",
        "manifest_protocol_matches_manifest",
        "manifest_alignment_status",
        "manifest_alignment_severity",
        "latest_eval_return_mean",
        "latest_eval_human_normalized_score",
        "best_eval_return_mean",
        "best_eval_human_normalized_score",
        "best_minus_latest_eval_return_mean",
        "best_minus_latest_eval_human_normalized_score",
        "best_checkpoint_path",
        "group_by",
        "group",
        "runs",
        "seed_count",
        "seeds",
        "latest_eval_return_mean_mean",
        "latest_eval_return_mean_median",
        "latest_eval_return_mean_min",
        "latest_eval_return_mean_max",
        "latest_eval_return_mean_iqr",
        "latest_eval_return_mean_std",
        "latest_eval_return_mean_stderr",
        "latest_eval_return_mean_ci95",
        "latest_eval_human_normalized_score_mean",
        "latest_eval_human_normalized_score_median",
        "latest_eval_human_normalized_score_min",
        "latest_eval_human_normalized_score_max",
        "latest_eval_human_normalized_score_iqr",
        "latest_eval_human_normalized_score_std",
        "latest_eval_human_normalized_score_stderr",
        "latest_eval_human_normalized_score_ci95",
        "best_eval_return_mean_max",
        "best_eval_human_normalized_score_max",
        "best_minus_latest_eval_return_mean_gap",
        "best_minus_latest_eval_human_normalized_score_gap",
        "best_over_latest_eval_return_ratio",
        "best_over_latest_eval_human_normalized_score_ratio",
        "baseline_latest_eval_return_mean_mean",
        "baseline_latest_eval_human_normalized_score_mean",
        "delta_vs_baseline_latest_eval_return_mean_mean",
        "delta_vs_baseline_latest_eval_human_normalized_score_mean",
        "ratio_vs_baseline_latest_eval_return_mean_mean",
        "ratio_vs_baseline_latest_eval_human_normalized_score_mean",
        "rank_best_eval_return_mean",
        "rank_latest_eval_return_mean",
        "rank_best_eval_human_normalized_score",
        "rank_latest_eval_human_normalized_score",
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _render_commands_output(manifest: Mapping[str, Any]) -> str:
    presets = manifest.get("presets", [])
    if not isinstance(presets, list):
        raise TypeError("manifest 'presets' must be a list")
    lines = [f"axiomrl train --config {preset['config']}" for preset in presets]
    return "\n".join(lines) + ("\n" if lines else "")


def _render_table_output(manifest: Mapping[str, Any]) -> str:
    presets = manifest.get("presets", [])
    if not isinstance(presets, list):
        raise TypeError("manifest 'presets' must be a list")
    lines = [f"suite={manifest.get('suite', 'unknown')}"]
    for preset in presets:
        lines.append(f"{preset['name']}: {preset['config']}")
    return "\n".join(lines) + "\n"


def _emit_output(content: str, *, output_path: str | Path | None = None) -> None:
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
    print(content, end="")


__all__ = ["build_report_payload"]

