from __future__ import annotations

import csv
from collections.abc import Mapping
import io
import json
from pathlib import Path
from typing import Any

from rl_training.zoo.manifests import _build_payload_metadata_fields
from rl_training.zoo.reporting import build_report_payload


LEADERBOARD_METRIC_CHOICES = (
    "best-return",
    "latest-return",
    "gap-return",
    "stability-return",
    "confidence-return",
    "median-return",
    "iqr-return",
    "delta-vs-baseline-return",
    "ratio-vs-baseline-return",
    "best-normalized",
    "latest-normalized",
    "gap-normalized",
    "stability-normalized",
    "confidence-normalized",
    "median-normalized",
    "iqr-normalized",
    "delta-vs-baseline-normalized",
    "ratio-vs-baseline-normalized",
)
COMPARE_TO_CHOICES = ("best", "latest")
SCORE_VIEW_CHOICES = ("return", "normalized")


def _resolve_leaderboard_metric_alias(
    manifest: Mapping[str, Any],
    *,
    leaderboard_metric: str | None = None,
    baseline_preset: str | None = None,
    compare_to: str | None = None,
    score_view: str | None = None,
    sort_by: str | None = None,
    descending: bool = False,
) -> tuple[str | None, str | None, str | None, str, bool]:
    metric_to_sort_by = {
        "best-return": ("best_eval_return_mean", True),
        "latest-return": ("latest_eval_return_mean", True),
        "gap-return": ("best_minus_latest_eval_return_mean_gap", True),
        "stability-return": ("latest_eval_return_mean_std", False),
        "confidence-return": ("latest_eval_return_mean_ci95", False),
        "median-return": ("latest_eval_return_mean_median", True),
        "iqr-return": ("latest_eval_return_mean_iqr", False),
        "delta-vs-baseline-return": ("delta_vs_baseline_latest_eval_return_mean_mean", True),
        "ratio-vs-baseline-return": ("ratio_vs_baseline_latest_eval_return_mean_mean", True),
        "best-normalized": ("best_eval_human_normalized_score", True),
        "latest-normalized": ("latest_eval_human_normalized_score", True),
        "gap-normalized": ("best_minus_latest_eval_human_normalized_score_gap", True),
        "stability-normalized": ("latest_eval_human_normalized_score_std", False),
        "confidence-normalized": ("latest_eval_human_normalized_score_ci95", False),
        "median-normalized": ("latest_eval_human_normalized_score_median", True),
        "iqr-normalized": ("latest_eval_human_normalized_score_iqr", False),
        "delta-vs-baseline-normalized": ("delta_vs_baseline_latest_eval_human_normalized_score_mean", True),
        "ratio-vs-baseline-normalized": ("ratio_vs_baseline_latest_eval_human_normalized_score_mean", True),
    }
    baseline_metric_choices = {
        "delta-vs-baseline-return",
        "ratio-vs-baseline-return",
        "delta-vs-baseline-normalized",
        "ratio-vs-baseline-normalized",
    }
    if leaderboard_metric is not None:
        if leaderboard_metric in baseline_metric_choices and baseline_preset is None:
            raise ValueError(f"--leaderboard-metric {leaderboard_metric} requires --baseline-preset")
        resolved_sort_by, resolved_descending = metric_to_sort_by[leaderboard_metric]
        return None, None, leaderboard_metric, resolved_sort_by, resolved_descending

    if sort_by is not None:
        return None, None, None, sort_by, descending

    score_normalization = manifest.get("score_normalization", {})
    prefers_normalized_scores = (
        isinstance(score_normalization, dict)
        and str(score_normalization.get("type", "none")).lower() != "none"
    )
    resolved_compare_to = compare_to if compare_to is not None else "best"
    if score_view == "normalized" and not prefers_normalized_scores:
        raise ValueError("--score-view normalized requires score normalization in the benchmark manifest")
    resolved_score_view = score_view if score_view is not None else ("normalized" if prefers_normalized_scores else "return")
    resolved_metric = f"{resolved_compare_to}-{resolved_score_view}"
    resolved_sort_by, resolved_descending = metric_to_sort_by[resolved_metric]
    return resolved_compare_to, resolved_score_view, resolved_metric, resolved_sort_by, resolved_descending


def build_leaderboard_payload(
    manifest: dict[str, Any],
    *,
    runs_dir: Path,
    manifest_source: Mapping[str, object] | None = None,
    algo: str | None = None,
    env_id: str | None = None,
    group_by: str = "algo-env",
    min_seeds: int | None = None,
    top_k: int | None = None,
    leaderboard_metric: str | None = None,
    baseline_preset: str | None = None,
    compare_to: str | None = None,
    score_view: str | None = None,
    sort_by: str | None = None,
    descending: bool = False,
) -> dict[str, Any]:
    resolved_compare_to, resolved_score_view, resolved_metric, resolved_sort_by, resolved_descending = (
        _resolve_leaderboard_metric_alias(
        manifest,
        leaderboard_metric=leaderboard_metric,
        baseline_preset=baseline_preset,
        compare_to=compare_to,
        score_view=score_view,
        sort_by=sort_by,
        descending=descending,
    ))
    report_payload = build_report_payload(
        manifest,
        runs_dir=runs_dir,
        manifest_source=manifest_source,
        algo=algo,
        env_id=env_id,
        group_by=group_by,
        min_seeds=min_seeds,
        top_k=top_k,
        baseline_preset=baseline_preset,
        sort_by=resolved_sort_by,
        descending=resolved_descending,
    )
    payload: dict[str, Any] = {
        "format": "leaderboard",
        "suite": report_payload["suite"],
        "protocol": report_payload["protocol"],
        "score_normalization": report_payload["score_normalization"],
        "runs_dir": report_payload["runs_dir"],
        "group_by": report_payload["group_by"],
        "sort_by": resolved_sort_by,
        "descending": resolved_descending,
        "entries": report_payload["aggregates"],
    }
    if "manifest_source" in report_payload:
        payload["manifest_source"] = report_payload["manifest_source"]
    if "manifest_metadata" in report_payload:
        payload["manifest_metadata"] = report_payload["manifest_metadata"]
    if "manifest_alignment_summary" in report_payload:
        payload["manifest_alignment_summary"] = report_payload["manifest_alignment_summary"]
    if "protocol_metadata" in report_payload:
        payload["protocol_metadata"] = report_payload["protocol_metadata"]
    if "score_normalization_metadata" in report_payload:
        payload["score_normalization_metadata"] = report_payload["score_normalization_metadata"]
    if "baseline_preset" in report_payload:
        payload["baseline_preset"] = report_payload["baseline_preset"]
    if "baseline_summary" in report_payload:
        payload["baseline_summary"] = report_payload["baseline_summary"]
    if resolved_compare_to is not None:
        payload["compare_to"] = resolved_compare_to
    if resolved_score_view is not None:
        payload["score_view"] = resolved_score_view
    if resolved_metric is not None:
        payload["leaderboard_metric"] = resolved_metric
    if "filters" in report_payload:
        payload["filters"] = report_payload["filters"]
    if "min_seeds" in report_payload:
        payload["min_seeds"] = report_payload["min_seeds"]
    if top_k is not None:
        payload["top_k"] = top_k
    return payload


def _render_text_leaderboard(payload: Mapping[str, Any]) -> str:
    lines = [
        "leaderboard",
        f"suite={payload['suite']}",
        f"protocol={payload['protocol']}",
        f"score_normalization={payload['score_normalization']}",
        f"runs_dir={payload['runs_dir']}",
        f"group_by={payload['group_by']}",
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
    if payload.get("compare_to") is not None:
        lines.append(f"compare_to={payload['compare_to']}")
    if payload.get("score_view") is not None:
        lines.append(f"score_view={payload['score_view']}")
    if payload.get("leaderboard_metric") is not None:
        lines.append(f"leaderboard_metric={payload['leaderboard_metric']}")
    lines.extend(
        [
            f"sort_by={payload['sort_by']}",
            f"descending={payload['descending']}",
        ]
    )
    if "top_k" in payload:
        lines.append(f"top_k={payload['top_k']}")

    for index, entry in enumerate(payload["entries"], start=1):
        lines.append(
            " ".join(
                [
                    f"rank={index}",
                    f"group={entry['group']}",
                    f"preset_name={entry['preset_name']}",
                    f"preset_config={entry.get('preset_config')}",
                    f"preset_description={entry.get('preset_description')}",
                    f"protocol_name={entry['protocol_name']}",
                    f"manifest_alignment_status={entry.get('manifest_alignment_status')}",
                    f"manifest_alignment_severity={entry.get('manifest_alignment_severity')}",
                    f"manifest_alignment_all_runs={entry.get('manifest_alignment_all_runs')}",
                    f"manifest_alignment_total_runs={entry.get('manifest_alignment_total_runs')}",
                    f"manifest_alignment_aligned_runs={entry.get('manifest_alignment_aligned_runs')}",
                    f"manifest_alignment_drifted_runs={entry.get('manifest_alignment_drifted_runs')}",
                    f"manifest_alignment_unknown_preset_runs={entry.get('manifest_alignment_unknown_preset_runs')}",
                    "manifest_alignment_protocol_mismatch_runs="
                    f"{entry.get('manifest_alignment_protocol_mismatch_runs')}",
                    f"algo={entry['algo']}",
                    f"env_id={entry['env_id']}",
                    f"runs={entry['runs']}",
                    f"seeds={entry['seeds']}",
                    f"seed_count={entry['seed_count']}",
                    f"baseline_preset={entry.get('baseline_preset')}",
                    f"best_eval_return_mean_max={entry['best_eval_return_mean_max']}",
                    f"latest_eval_return_mean_mean={entry['latest_eval_return_mean_mean']}",
                    f"latest_eval_return_mean_median={entry['latest_eval_return_mean_median']}",
                    f"latest_eval_return_mean_min={entry['latest_eval_return_mean_min']}",
                    f"latest_eval_return_mean_max={entry['latest_eval_return_mean_max']}",
                    f"latest_eval_return_mean_iqr={entry['latest_eval_return_mean_iqr']}",
                    f"latest_eval_return_mean_std={entry['latest_eval_return_mean_std']}",
                    f"latest_eval_return_mean_stderr={entry['latest_eval_return_mean_stderr']}",
                    f"latest_eval_return_mean_ci95={entry['latest_eval_return_mean_ci95']}",
                    f"best_minus_latest_eval_return_mean_gap={entry['best_minus_latest_eval_return_mean_gap']}",
                    f"best_over_latest_eval_return_ratio={entry['best_over_latest_eval_return_ratio']}",
                    f"baseline_latest_eval_return_mean_mean={entry.get('baseline_latest_eval_return_mean_mean')}",
                    f"delta_vs_baseline_latest_eval_return_mean_mean={entry.get('delta_vs_baseline_latest_eval_return_mean_mean')}",
                    f"ratio_vs_baseline_latest_eval_return_mean_mean={entry.get('ratio_vs_baseline_latest_eval_return_mean_mean')}",
                    f"rank_best_eval_return_mean={entry['rank_best_eval_return_mean']}",
                    f"rank_latest_eval_return_mean={entry['rank_latest_eval_return_mean']}",
                    f"best_eval_human_normalized_score_max={entry['best_eval_human_normalized_score_max']}",
                    f"latest_eval_human_normalized_score_mean={entry['latest_eval_human_normalized_score_mean']}",
                    f"latest_eval_human_normalized_score_median={entry['latest_eval_human_normalized_score_median']}",
                    f"latest_eval_human_normalized_score_min={entry['latest_eval_human_normalized_score_min']}",
                    f"latest_eval_human_normalized_score_max={entry['latest_eval_human_normalized_score_max']}",
                    f"latest_eval_human_normalized_score_iqr={entry['latest_eval_human_normalized_score_iqr']}",
                    f"latest_eval_human_normalized_score_std={entry['latest_eval_human_normalized_score_std']}",
                    f"latest_eval_human_normalized_score_stderr={entry['latest_eval_human_normalized_score_stderr']}",
                    f"latest_eval_human_normalized_score_ci95={entry['latest_eval_human_normalized_score_ci95']}",
                    f"best_minus_latest_eval_human_normalized_score_gap={entry['best_minus_latest_eval_human_normalized_score_gap']}",
                    f"best_over_latest_eval_human_normalized_score_ratio={entry['best_over_latest_eval_human_normalized_score_ratio']}",
                    f"baseline_latest_eval_human_normalized_score_mean={entry.get('baseline_latest_eval_human_normalized_score_mean')}",
                    f"delta_vs_baseline_latest_eval_human_normalized_score_mean={entry.get('delta_vs_baseline_latest_eval_human_normalized_score_mean')}",
                    f"ratio_vs_baseline_latest_eval_human_normalized_score_mean={entry.get('ratio_vs_baseline_latest_eval_human_normalized_score_mean')}",
                    f"rank_best_eval_human_normalized_score={entry['rank_best_eval_human_normalized_score']}",
                    f"rank_latest_eval_human_normalized_score={entry['rank_latest_eval_human_normalized_score']}",
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


def _render_json_leaderboard(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2) + "\n"


def _render_csv_leaderboard(payload: Mapping[str, Any]) -> str:
    rows: list[dict[str, object]] = []
    common = {
        "suite": payload["suite"],
        "protocol": payload["protocol"],
        "score_normalization": payload["score_normalization"],
        "runs_dir": payload["runs_dir"],
        "baseline_preset": payload.get("baseline_preset"),
        "group_by": payload["group_by"],
        "compare_to": payload.get("compare_to"),
        "score_view": payload.get("score_view"),
        "leaderboard_metric": payload.get("leaderboard_metric"),
        "sort_by": payload["sort_by"],
        "descending": payload["descending"],
    }
    common.update(_build_payload_metadata_fields(payload))
    for index, entry in enumerate(payload["entries"], start=1):
        row = dict(common)
        row["kind"] = "leaderboard"
        row["rank"] = index
        row.update(entry)
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
                row["rank"] = rank
                row["summary_kind"] = summary_kind
                row.update(entry)
                rows.append(row)

    fieldnames = [
        "kind",
        "rank",
        "summary_kind",
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
        "group_by",
        "compare_to",
        "score_view",
        "leaderboard_metric",
        "sort_by",
        "descending",
        "group",
        "preset_name",
        "preset_config",
        "preset_description",
        "protocol_name",
        "manifest_preset_known",
        "manifest_protocol_matches_manifest",
        "manifest_alignment_status",
        "manifest_alignment_severity",
        "algo",
        "env_id",
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


__all__ = [
    "COMPARE_TO_CHOICES",
    "LEADERBOARD_METRIC_CHOICES",
    "SCORE_VIEW_CHOICES",
    "build_leaderboard_payload",
]
