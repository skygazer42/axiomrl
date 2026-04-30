from __future__ import annotations

import csv
import io
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from axiomrl.zoo.manifests import _build_payload_metadata_fields


def render_text_report(payload: Mapping[str, Any]) -> str:
    reports = payload["runs"]
    aggregates = payload["aggregates"]
    lines = [
        f"suite={payload['suite']}",
        f"protocol={payload['protocol']}",
        f"score_normalization={payload['score_normalization']}",
        f"runs_dir={payload['runs_dir']}",
    ]
    metadata_fields = _build_payload_metadata_fields(payload)
    for field in (
        "manifest_requested_path",
        "manifest_resolved_path",
        "manifest_source_kind",
        "manifest_fingerprint",
        "manifest_preset_count",
        "manifest_alignment_total_runs",
        "manifest_alignment_aligned_runs",
        "manifest_alignment_drifted_runs",
        "manifest_alignment_unknown_preset_runs",
        "manifest_alignment_protocol_mismatch_runs",
        "manifest_alignment_all_runs",
        "manifest_alignment_severity",
        "manifest_alignment_fail_reasons",
        "protocol_description",
        "protocol_training",
        "protocol_evaluation",
        "score_normalization_game",
        "score_normalization_source",
        "score_normalization_random_score",
        "score_normalization_human_score",
        "score_normalization_scale",
    ):
        value = metadata_fields.get(field)
        if value is not None:
            lines.append(f"{field}={value}")
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
                    f"manifest_alignment_protocol_mismatch_runs={aggregate.get('manifest_alignment_protocol_mismatch_runs')}",
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
                            f"manifest_alignment_protocol_mismatch_runs={entry.get('manifest_alignment_protocol_mismatch_runs')}",
                            f"delta_vs_baseline_latest_eval_return_mean_mean={entry.get('delta_vs_baseline_latest_eval_return_mean_mean')}",
                            "delta_vs_baseline_latest_eval_human_normalized_score_mean="
                            f"{entry.get('delta_vs_baseline_latest_eval_human_normalized_score_mean')}",
                            f"ratio_vs_baseline_latest_eval_return_mean_mean={entry.get('ratio_vs_baseline_latest_eval_return_mean_mean')}",
                            "ratio_vs_baseline_latest_eval_human_normalized_score_mean="
                            f"{entry.get('ratio_vs_baseline_latest_eval_human_normalized_score_mean')}",
                        ]
                    )
                )
    return "\n".join(lines) + "\n"


def render_json_report(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2) + "\n"


def csv_report_rows(payload: Mapping[str, Any]) -> list[dict[str, object]]:
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


def render_csv_report(payload: Mapping[str, Any]) -> str:
    rows = csv_report_rows(payload)
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


def render_commands_output(manifest: Mapping[str, Any]) -> str:
    presets = manifest.get("presets", [])
    if not isinstance(presets, list):
        raise TypeError("manifest 'presets' must be a list")
    lines = [f"axiomrl train --config {preset['config']}" for preset in presets]
    return "\n".join(lines) + ("\n" if lines else "")


def render_table_output(manifest: Mapping[str, Any]) -> str:
    presets = manifest.get("presets", [])
    if not isinstance(presets, list):
        raise TypeError("manifest 'presets' must be a list")
    lines = [f"suite={manifest.get('suite', 'unknown')}"]
    for preset in presets:
        lines.append(f"{preset['name']}: {preset['config']}")
    return "\n".join(lines) + "\n"


def emit_output(content: str, *, output_path: str | Path | None = None) -> None:
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
    print(content, end="")
